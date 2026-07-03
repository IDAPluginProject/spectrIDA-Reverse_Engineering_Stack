"""
Need-driven growth + the anti-stagnation logic ("never stays dumb").

Two independent concerns:

1. ``CompetenceTracker`` / ``GrowthController`` — decide, per behavior family,
   whether the agent is (a) still learning, (b) mastered it, (c) stuck on
   something *learnable* (→ grow capacity), or (d) stuck on *noise* (→ do NOT
   grow; decay its curiosity so it stops wasting effort — the noisy-TV guard).

   The discriminator is the key idea:
     - error decreasing            → learning, leave it alone
     - error low + stable          → mastered
     - error HIGH + LOW variance + not improving  → deterministic but underfit
                                                    → CAPACITY problem → GROW
     - error HIGH + HIGH variance + not improving → aleatoric noise (date, RNG,
                                                    net jitter) → noisy-TV → decay

2. ``GrowableCorrector`` — function-preserving capacity growth. New capacity is a
   zero-initialised residual block, so at the instant it's added the model's
   output is unchanged (net2net-style identity), then it trains. Growth strictly
   increases parameters and never regresses what was already learned.
"""

from __future__ import annotations

from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn as nn


# ── function-preserving capacity growth ──────────────────────────────────────
class _ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim),
        )
        # zero-init the last layer → block outputs 0 at creation (identity growth)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class GrowableCorrector(nn.Module):
    """A learned correction on the world model's predicted next latent.

    Starts as an exact zero (function-preserving w.r.t. the base model). Each
    ``grow()`` appends a zero-initialised residual block: capacity goes up, output
    is unchanged at that instant. This is the honest, robust variant of net2net —
    identity-preserving growth that works regardless of the base net's internals.
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.in_dim = latent_dim + action_dim
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.blocks = nn.ModuleList()

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if not self.blocks:
            return torch.zeros_like(latent)
        x = torch.cat([latent, action], dim=-1)
        out = self.blocks[0](x)
        for b in self.blocks[1:]:
            out = out + b(x)
        return out

    def grow(self) -> int:
        block = _ResidualBlock(self.in_dim, self.latent_dim, self.hidden)
        block.to(next(self.parameters()).device if list(self.parameters()) else "cpu")
        self.blocks.append(block)
        return len(self.blocks)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── competence tracking + grow/decay/stall decisions ─────────────────────────
class CompetenceTracker:
    """Per-family prediction-error history and its interpretation."""

    def __init__(self, window: int = 60,
                 master_thresh: float = 0.02,
                 high_thresh: float = 0.08,
                 progress_eps: float = 0.05,
                 noise_var_ratio: float = 0.35):
        self.window = window
        self.master_thresh = master_thresh
        self.high_thresh = high_thresh
        self.progress_eps = progress_eps
        self.noise_var_ratio = noise_var_ratio
        self.err: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, family: str, error: float) -> None:
        self.err[family].append(float(error))

    # summary stats
    def mean_error(self, family: str) -> float:
        h = self.err[family]
        return float(np.mean(h)) if h else 1.0

    def _halves(self, family: str):
        h = list(self.err[family])
        if len(h) < self.window:
            return None, None
        k = len(h) // 2
        return np.mean(h[:k]), np.mean(h[k:])

    def learning_progress(self, family: str) -> float:
        """Fractional error reduction recent-vs-older (positive = improving)."""
        old, new = self._halves(family)
        if old is None or old <= 1e-8:
            return 0.0
        return float((old - new) / old)

    def noise_ratio(self, family: str) -> float:
        """std/mean of recent error — high => unpredictable (aleatoric)."""
        h = self.err[family]
        if len(h) < self.window:
            return 0.0
        m = np.mean(h)
        return float(np.std(h) / (m + 1e-8))

    # interpretations
    def is_mastered(self, family: str) -> bool:
        return (len(self.err[family]) >= self.window
                and self.mean_error(family) < self.master_thresh)

    def is_improving(self, family: str) -> bool:
        return self.learning_progress(family) > self.progress_eps

    def is_noisy(self, family: str) -> bool:
        """High, non-improving error with high variance = noisy-TV (unlearnable)."""
        return (len(self.err[family]) >= self.window
                and self.mean_error(family) > self.high_thresh
                and not self.is_improving(family)
                and self.noise_ratio(family) > self.noise_var_ratio)

    def needs_capacity(self, family: str) -> bool:
        """High, non-improving error with LOW variance = deterministic but
        underfit → a capacity problem the model should grow to solve."""
        return (len(self.err[family]) >= self.window
                and self.mean_error(family) > self.high_thresh
                and not self.is_improving(family)
                and self.noise_ratio(family) <= self.noise_var_ratio)


class GrowthController:
    """Decides grow/decay/stall and applies function-preserving growth."""

    def __init__(self, corrector: GrowableCorrector, tracker: CompetenceTracker,
                 log=print, max_blocks: int = 24, cooldown: int = 300):
        self.corrector = corrector
        self.tracker = tracker
        self.log = log
        self.max_blocks = max_blocks
        self.cooldown = cooldown
        self._last_grow_step = -10**9
        self.events: list[dict] = []

    def curiosity_weight(self, family: str) -> float:
        """Multiplier applied to a family's curiosity value. Noisy-TV families
        get decayed toward zero so the agent stops chasing unlearnable noise."""
        return 0.1 if self.tracker.is_noisy(family) else 1.0

    def maybe_grow(self, step: int) -> bool:
        """Grow iff some family genuinely needs capacity (learnable + underfit)."""
        if len(self.corrector.blocks) >= self.max_blocks:
            return False
        if step - self._last_grow_step < self.cooldown:
            return False
        stuck = [f for f in self.tracker.err if self.tracker.needs_capacity(f)]
        if not stuck:
            return False
        family = max(stuck, key=self.tracker.mean_error)
        n = self.corrector.grow()
        self._last_grow_step = step
        err = self.tracker.mean_error(family)
        ev = {"step": step, "blocks": n, "family": family, "error": round(err, 4),
              "params": self.corrector.num_params()}
        self.events.append(ev)
        self.log(f"[grow] +capacity (block #{n}) — family '{family}' error "
                 f"plateaued at {err:.3f}, low-variance/learnable → "
                 f"corrector params={ev['params']:,}")
        return True

    def stall_report(self) -> str | None:
        """If NOTHING is improving and everything unmastered is noise/at-ceiling,
        say so loudly instead of faking progress."""
        fams = [f for f in self.tracker.err if len(self.tracker.err[f]) >= self.tracker.window]
        if not fams:
            return None
        unmastered = [f for f in fams if not self.tracker.is_mastered(f)]
        if not unmastered:
            return None
        if any(self.tracker.is_improving(f) or self.tracker.needs_capacity(f)
               for f in unmastered):
            return None  # still making or able-to-make progress
        noisy = [f for f in unmastered if self.tracker.is_noisy(f)]
        if len(self.corrector.blocks) >= self.max_blocks:
            return (f"STALL: at max capacity and no family improving; "
                    f"unmastered={unmastered}. Real ceiling — needs a bigger/"
                    f"different architecture, not more of the same.")
        if set(noisy) == set(unmastered):
            return (f"STALL(benign): remaining unmastered families are all "
                    f"unlearnable noise {noisy} — correctly not chasing them.")
        return None

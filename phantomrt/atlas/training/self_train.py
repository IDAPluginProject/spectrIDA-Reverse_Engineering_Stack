"""
SelfTrainer — the genuine self-directed learning loop.

Every step:
  1. The proposer offers candidate commands.
  2. The model scores them by CURIOSITY (expected prediction error on families it
     hasn't mastered + novelty), with noisy-TV families decayed. It picks the best.
     -> The chosen command is what actually runs. THE LOOP IS CLOSED. (The old
        scripts chose randomly and the network output did nothing.)
  3. The command runs in the real VM; we get the real outcome.
  4. SURPRISE = real prediction error between what the world model predicted and
     what actually happened. We take a gradient step to reduce it (surprise-gated),
     with experience replay so new learning doesn't erase old (anti-forgetting).
  5. Growth fires only when a family is provably learnable-but-underfit.

Understanding vs. memorization is measured explicitly: a held-out set of probe
commands is NEVER trained on; their prediction error is reported every interval.
Falling held-out error = it generalized (understood the structure); flat held-out
error while seen-command error falls = memorization (and the metric shows it).
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..core.world_model import WorldModel
from ..agents.command_space import embed_command, command_family
from .growth import GrowableCorrector, CompetenceTracker, GrowthController


# ── replay (anti-forgetting via rehearsal) ───────────────────────────────────
class TransitionReplay:
    """Stores (obs, action, next_obs, error). Sampling mixes recent experience
    with error-prioritised old experience so old skills keep getting rehearsed."""

    def __init__(self, capacity: int = 50000, rng: random.Random | None = None):
        self.capacity = capacity
        self.buf: list = []
        self.pos = 0
        self.rng = rng or random.Random(0)

    def add(self, obs, action, next_obs, error):
        item = (obs, action, next_obs, float(error))
        if len(self.buf) < self.capacity:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, old_ratio: float = 0.4):
        n = len(self.buf)
        if n <= batch_size:
            return list(self.buf)
        recent_n = int(batch_size * (1 - old_ratio))
        recent_start = max(0, n - 500)
        recent = [self.rng.randrange(recent_start, n) for _ in range(recent_n)]
        # prioritise old high-error transitions
        old_n = batch_size - recent_n
        errs = np.array([self.buf[i][3] for i in range(n)]) + 1e-3
        probs = errs / errs.sum()
        old = np.random.choice(n, size=old_n, replace=False, p=probs)
        idx = list(recent) + list(old)
        return [self.buf[i] for i in idx]

    def __len__(self):
        return len(self.buf)


def _bucket(vec: np.ndarray, prec: int = 1) -> tuple:
    return tuple(np.round(vec, prec).tolist())


class SelfTrainer:
    def __init__(self, env, proposer, device: str = "cpu",
                 latent_dim: int = 64, hidden: int = 128, lr: float = 1e-3,
                 batch_size: int = 64, epsilon: float = 0.15,
                 n_candidates: int = 16, log=print, seed: int = 0,
                 checkpoint_dir: str = "experiments/atlas_self",
                 embed_fn=embed_command, family_fn=command_family):
        self.env = env
        self.proposer = proposer
        self.device = device
        # pluggable action encoding: defaults to shell-command space, but the
        # binary-fuzz env injects its own input embedding / family functions.
        self.embed_fn = embed_fn
        self.family_fn = family_fn
        self.log = log
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.n_candidates = n_candidates
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        self.obs_dim = env.get_observation_dim()      # adapt to whatever env we drive
        self.wm = WorldModel(
            obs_dim=self.obs_dim, action_dim=env.get_action_dim(),
            latent_dim=latent_dim, hidden_dim=hidden,
            dynamics_solver="euler", dynamics_dt=1.0, dropout=0.0,
            surprise_threshold=0.05,
        ).to(device)
        self.corrector = GrowableCorrector(latent_dim, env.get_action_dim(), hidden=64).to(device)

        self.tracker = CompetenceTracker()
        self.growth = GrowthController(self.corrector, self.tracker, log=log)
        self.replay = TransitionReplay(rng=self.rng)

        self.opt = torch.optim.AdamW(
            list(self.wm.parameters()) + list(self.corrector.parameters()),
            lr=lr, weight_decay=1e-5,
        )

        self.pred_state_counts: Counter = Counter()
        self.action_region: Counter = Counter()
        self.probes: list[str] = []
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.history = {"step": [], "train_err": [], "heldout_err": [],
                        "coverage": [], "params": [], "learning_progress": []}

    # ── prediction path ──────────────────────────────────────────────────────
    def _predict_next(self, obs_t: torch.Tensor, act_t: torch.Tensor,
                      sample: bool = False) -> torch.Tensor:
        """Predict next-obs features for (obs, action). Deterministic unless sample."""
        mean, log_var, z = self.wm.encode(obs_t)
        latent = z if sample else mean
        nxt = self.wm.step_dynamics(latent, act_t) + self.corrector(latent, act_t)
        return self.wm.predict(nxt)

    @torch.no_grad()
    def _prediction_error(self, obs, action, next_obs) -> float:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        act_t = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        pred = self._predict_next(obs_t, act_t)
        tgt = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return F.mse_loss(pred, tgt).item()

    # ── curiosity-driven selection (closed loop) ─────────────────────────────
    @torch.no_grad()
    def _select(self, obs, candidates: list[str]) -> str:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(candidates)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, _, _ = self.wm.encode(obs_t)
        best, best_score = candidates[0], -1e9
        for cand in candidates:
            a = self.embed_fn(cand)
            act_t = torch.tensor(a, dtype=torch.float32, device=self.device).unsqueeze(0)
            nxt = self.wm.step_dynamics(mean, act_t) + self.corrector(mean, act_t)
            pred = self.wm.predict(nxt).squeeze(0).cpu().numpy()

            fam = self.family_fn(cand)
            fam_err = self.tracker.mean_error(fam)          # expected surprise
            w = self.growth.curiosity_weight(fam)           # noisy-TV decay
            pstate_nov = 1.0 / math.sqrt(1 + self.pred_state_counts[_bucket(pred)])
            areg_nov = 1.0 / math.sqrt(1 + self.action_region[_bucket(a)])
            score = w * fam_err + 0.5 * pstate_nov + 0.3 * areg_nov
            if score > best_score:
                best, best_score = cand, score
        return best

    # ── training ─────────────────────────────────────────────────────────────
    def _train_batch(self) -> float:
        if len(self.replay) < self.batch_size:
            return 0.0
        batch = self.replay.sample(self.batch_size)
        obs = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        act = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32, device=self.device)
        nxt = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32, device=self.device)

        mean, log_var, z = self.wm.encode(obs)
        pred_next = self.wm.predict(self.wm.step_dynamics(z, act) + self.corrector(z, act))
        recon = self.wm.predict(z)
        kl = self.wm.encoder.kl_divergence(mean, log_var)
        loss = F.mse_loss(pred_next, nxt) + 0.1 * F.mse_loss(recon, obs) + 1e-3 * kl

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.wm.parameters()) + list(self.corrector.parameters()), 1.0)
        self.opt.step()
        return F.mse_loss(pred_next, nxt).item()

    # ── held-out generalization probe (understand vs memorize) ───────────────
    def _make_probes(self, k: int = 15):
        pool = self.proposer.propose(k * 3)
        self.probes = pool[:k]
        self.log(f"[probe] held-out set ({len(self.probes)} cmds, never trained on)")

    @torch.no_grad()
    def evaluate_generalization(self) -> float:
        if not self.probes:
            return float("nan")
        errs = []
        for cmd in self.probes:
            next_obs = self.env.run_probe(cmd)   # run without training on it
            errs.append(self._prediction_error(self.env._last, self.embed_fn(cmd), next_obs))
        return float(np.mean(errs)) if errs else float("nan")

    # ── main loop ────────────────────────────────────────────────────────────
    def run(self, max_steps: int = 5000, budget_seconds: float | None = None,
            report_every: int = 200):
        self.log(f"[atlas] self-training | device={self.device} | "
                 f"obs={self.obs_dim} act={self.env.get_action_dim()} "
                 f"latent={self.wm.latent_dim}")
        self._make_probes()
        obs = self.env.reset()
        start = time.time()
        recent_err = deque(maxlen=report_every)

        for step in range(1, max_steps + 1):
            if budget_seconds and time.time() - start > budget_seconds:
                self.log(f"[atlas] time budget reached at step {step}")
                break

            candidates = self.proposer.propose(self.n_candidates)
            cmd = self._select(obs, candidates)
            action = self.embed_fn(cmd)

            next_obs, reward, done, info = self.env.step(cmd)
            error = self._prediction_error(obs, action, next_obs)  # REAL surprise

            fam = info["family"]
            self.tracker.update(fam, error)
            self.proposer.observe(cmd, info["result"])
            self.replay.add(obs, action, next_obs, error)
            self.pred_state_counts[_bucket(next_obs)] += 1
            self.action_region[_bucket(action)] += 1
            recent_err.append(error)

            train_err = self._train_batch()
            self.growth.maybe_grow(step)

            obs = next_obs
            if info["recovered"]:
                obs = self.env.reset()

            if step % report_every == 0:
                self._report(step, np.mean(recent_err), start)

        return self._finish(start)

    def _report(self, step, train_err, start):
        heldout = self.evaluate_generalization()
        cov = len(self.env.seen)
        params = sum(p.numel() for p in self.wm.parameters()) + self.corrector.num_params()
        # aggregate learning progress across active families
        lps = [self.tracker.learning_progress(f) for f in self.tracker.err
               if len(self.tracker.err[f]) >= self.tracker.window]
        lp = float(np.mean(lps)) if lps else 0.0

        self.history["step"].append(step)
        self.history["train_err"].append(round(float(train_err), 5))
        self.history["heldout_err"].append(round(float(heldout), 5))
        self.history["coverage"].append(cov)
        self.history["params"].append(params)
        self.history["learning_progress"].append(round(lp, 4))

        gen = "generalizing" if heldout <= train_err * 1.5 else "MEMORIZING?"
        self.log(
            f"[{step:5d}] train_err={train_err:.4f} heldout_err={heldout:.4f} ({gen}) | "
            f"coverage={cov} | families={len(self.tracker.err)} | "
            f"blocks={len(self.corrector.blocks)} params={params:,} | "
            f"lp={lp:+.3f} | {time.time()-start:.0f}s"
        )
        # per-family competence snapshot
        for f in sorted(self.tracker.err):
            m = self.tracker.mean_error(f)
            tag = ("mastered" if self.tracker.is_mastered(f) else
                   "noisy-TV" if self.tracker.is_noisy(f) else
                   "underfit→grow" if self.tracker.needs_capacity(f) else
                   "learning" if self.tracker.is_improving(f) else "exploring")
            self.log(f"        {f:8s} err={m:.3f} [{tag}]")
        stall = self.growth.stall_report()
        if stall:
            self.log(f"        !! {stall}")
        self._save()

    def _save(self):
        torch.save({"wm": self.wm.state_dict(), "corrector": self.corrector.state_dict(),
                    "history": self.history, "growth_events": self.growth.events},
                   self.ckpt_dir / "model.pt")
        (self.ckpt_dir / "history.json").write_text(json.dumps(self.history, indent=2))

    def _finish(self, start):
        self._save()
        mastered = [f for f in self.tracker.err if self.tracker.is_mastered(f)]
        noisy = [f for f in self.tracker.err if self.tracker.is_noisy(f)]
        self.log("")
        self.log("── FINAL ──")
        self.log(f"  families seen : {len(self.tracker.err)}")
        self.log(f"  mastered      : {mastered}")
        self.log(f"  noisy (skipped): {noisy}")
        self.log(f"  coverage      : {len(self.env.seen)} distinct behaviors")
        self.log(f"  growth events : {len(self.growth.events)}")
        self.log(f"  recoveries    : {self.env.recoveries}")
        self.log(f"  time          : {time.time()-start:.0f}s")
        return self.history

"""
VMWorldEnv — the real environment: a whole Linux box the agent acts on.

Unlike the toy grid/physics envs, the "action" here is a **shell command string**
(chosen by the agent from the CommandProposer's candidates). ``step`` runs it in
the isolated VM and featurizes the *real* outcome into the next observation.

State/observation = features of the last command's real result (exit code, output
size/entropy, error signatures, timing, …). The world model's job is to predict
this vector for an action before it runs — prediction error is the surprise signal.

Reward is count-based novelty (intrinsic curiosity): behaviors the agent has rarely
produced are worth more, so it seeks out unseen corners of the machine. Distinct
behavior signatures = coverage.

Safety: after a plausibly-destructive or failed command we health-check the VM and,
if it's bricked, roll back to the base snapshot and continue — recovery, not a
rewarded action.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import numpy as np

from .base import BaseEnvironment
from ..agents.command_space import embed_command, command_family, ACTION_DIM

STATE_DIM = 24


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


class VMWorldEnv(BaseEnvironment):
    """A real Linux VM presented as an RL-style environment."""

    def __init__(self, vm, log=print, auto_recover: bool = True,
                 snapshot_tag: str = "base"):
        self.vm = vm
        self.log = log
        self.auto_recover = auto_recover
        self.snapshot_tag = snapshot_tag

        self.seen = Counter()          # behavior signature -> count (novelty/coverage)
        self.family_seen = Counter()   # family -> count
        self._last = np.zeros(STATE_DIM, dtype=np.float32)
        self.steps = 0
        self.recoveries = 0

    # ── BaseEnvironment API ──────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        self._last = np.zeros(STATE_DIM, dtype=np.float32)
        return self._last.copy()

    def get_observation_dim(self) -> int:
        return STATE_DIM

    def get_action_dim(self) -> int:
        return ACTION_DIM

    def render(self):
        return None

    def run_probe(self, command: str, timeout: int = 8) -> np.ndarray:
        """Run a command purely to measure prediction error (held-out probe);
        does not mutate learning state."""
        return self.featurize(command, self.vm.run(command, timeout=timeout))

    # ── the real step ────────────────────────────────────────────────────────
    def step(self, command: str, timeout: int = 8):
        """Run ``command`` in the VM; return (obs, reward, done, info) with the
        real outcome. ``command`` is a string (the agent's chosen action)."""
        self.steps += 1
        result = self.vm.run(command, timeout=timeout)

        obs = self.featurize(command, result)
        sig = self.behavior_signature(command, result)

        # count-based novelty: rarer behavior -> higher intrinsic reward
        self.seen[sig] += 1
        self.family_seen[command_family(command)] += 1
        reward = 1.0 / math.sqrt(self.seen[sig])
        is_new = self.seen[sig] == 1

        recovered = self._maybe_recover(command, result)

        self._last = obs
        info = {
            "command": command,
            "result": result,
            "signature": sig,
            "is_new_behavior": is_new,
            "coverage": len(self.seen),
            "recovered": recovered,
            "family": command_family(command),
        }
        return obs, reward, False, info

    # ── featurization: real result -> observation vector ─────────────────────
    def featurize(self, command: str, result) -> np.ndarray:
        out, err = result.stdout, result.stderr
        low = (out + " " + err).lower()
        v = np.zeros(STATE_DIM, dtype=np.float32)
        v[0] = max(-1.0, min(1.0, result.exit_code / 128.0))
        v[1] = 1.0 if result.exit_code == 0 else 0.0
        v[2] = 1.0 if result.crashed else 0.0
        v[3] = 1.0 if result.timed_out else 0.0
        v[4] = min(len(out) / 500.0, 1.0)
        v[5] = min(len(err) / 500.0, 1.0)
        v[6] = min(out.count("\n") / 50.0, 1.0)
        v[7] = _entropy(out[:1000]) / 8.0
        v[8] = 1.0 if "error" in low else 0.0
        v[9] = 1.0 if ("not found" in low or "no such" in low) else 0.0
        v[10] = 1.0 if "permission denied" in low else 0.0
        v[11] = 1.0 if "usage" in low else 0.0
        v[12] = 1.0 if any(ch.isdigit() for ch in out[:200]) else 0.0
        v[13] = 1.0 if out.strip() else 0.0
        v[14] = 1.0 if err.strip() else 0.0
        v[15] = min(result.duration / 8.0, 1.0)            # normalized wall time
        v[16] = embed_command(command)[29]                 # was destructive
        v[17] = 1.0 if result.exit_code == -1 else 0.0     # launch failure
        v[18] = 1.0 if result.exit_code == 127 else 0.0    # command not found
        v[19] = 1.0 if result.exit_code == 126 else 0.0    # not executable
        v[20] = 1.0 if "\n" in out.strip() else 0.0        # multiline
        v[21] = 1.0 if len(out) > 2000 else 0.0            # large output
        v[22] = 1.0 if (result.exit_code == 0 and not out.strip()) else 0.0  # silent success
        v[23] = 1.0                                        # bias
        return v

    def behavior_signature(self, command: str, result):
        """Coarse hash of *what kind of thing happened* — for novelty/coverage."""
        size_bucket = 0 if not result.stdout else min(len(result.stdout).bit_length(), 12)
        return (
            command_family(command),
            "ok" if result.exit_code == 0 else ("crash" if result.crashed else
                   ("timeout" if result.timed_out else f"rc{result.exit_code}")),
            size_bucket,
            bool(result.stderr.strip()),
        )

    # ── auto-recovery (the rollback safety net) ──────────────────────────────
    def _maybe_recover(self, command: str, result) -> bool:
        if not self.auto_recover:
            return False
        # only pay for a health check when something might have broken the box
        risky = embed_command(command)[29] > 0 or result.exit_code in (-1, 124) \
            or result.crashed
        if not risky:
            return False
        if self.vm.health_ok():
            return False
        self.log(f"[recover] VM unhealthy after: {command!r} → rolling back")
        try:
            self.vm.rollback(self.snapshot_tag)
            self.recoveries += 1
            return True
        except Exception as e:
            self.log(f"[recover] rollback FAILED: {e}")
            return False

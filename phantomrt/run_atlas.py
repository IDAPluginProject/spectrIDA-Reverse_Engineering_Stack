"""
Atlas — turn the self-directed learner loose on the isolated VM.

    python run_atlas.py                 # 5000 steps
    python run_atlas.py --budget 300    # run for ~5 minutes
    python run_atlas.py --steps 20000
    python run_atlas.py --fresh         # rebuild the VM from scratch first

What it does (honestly):
  * ensures the isolated `atlas-vm` exists (offline, no host mount, non-root),
  * takes a `base` snapshot so the agent can be rolled back if it bricks the box,
  * runs the closed-loop curiosity learner: it PICKS commands to run, PREDICTS
    their outcome, learns from the real prediction error, and grows only when a
    behavior family is provably learnable-but-underfit,
  * logs training error, held-out generalization error (understand vs memorize),
    coverage, growth events, and stall warnings to train_log.txt.
"""
import argparse
import sys
import time
from pathlib import Path

# Windows consoles default to cp1252 and choke on unicode (→, ──). Force UTF-8.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch

from atlas.vm import WslVM
from atlas.environments.vm_world import VMWorldEnv
from atlas.agents.command_space import CommandProposer
from atlas.training.self_train import SelfTrainer

LOG_PATH = Path("train_log.txt")


def make_logger():
    LOG_PATH.write_text("")

    def log(msg=""):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else ""
        print(line)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--budget", type=float, default=None, help="wall-clock seconds")
    ap.add_argument("--fresh", action="store_true", help="rebuild the VM first")
    ap.add_argument("--report-every", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    log = make_logger()
    log("ATLAS — self-directed learner on isolated VM")
    log(f"device={args.device}")

    # 1. isolated VM + rollback snapshot
    vm = WslVM(log=log)
    vm.provision(force=args.fresh)
    if not vm.health_ok():
        log("FATAL: VM unhealthy after provisioning"); return 1
    if args.fresh or not vm.has_snapshot("base"):
        vm.snapshot("base")   # the safety net for auto-recovery
    log("VM ready (isolated, offline, non-root); base snapshot in place")

    # 2. wire environment + agent + trainer
    proposer = CommandProposer(vm)
    env = VMWorldEnv(vm, log=log)
    trainer = SelfTrainer(env, proposer, device=args.device, log=log)

    # 3. loose
    trainer.run(max_steps=args.steps, budget_seconds=args.budget,
                report_every=args.report_every)
    log("done — checkpoint at experiments/atlas_self/model.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Point Atlas at ONE binary and let it hunt crashes (coverage-guided).

    python run_fuzz.py                       # built-in vulnerable demo target
    python run_fuzz.py --source path/to.c    # your own C source (gets instrumented)
    python run_fuzz.py --steps 4000 --budget 180

It compiles the target with function-level coverage instrumentation inside the
isolated VM, then the world model learns input->behavior and its curiosity chases
new code paths and crashes. Reports coverage, unique crashes, and the crashing
inputs it found.
"""
import argparse
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch

from atlas.vm import WslVM
from atlas.analysis.binary_fuzz import (
    BinaryFuzzEnv, InputProposer, embed_input, input_family,
)
from atlas.training.self_train import SelfTrainer

LOG = Path("fuzz_log.txt")


def make_logger():
    LOG.write_text("")

    def log(msg=""):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else ""
        print(line)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="C source file to fuzz (default: demo target)")
    ap.add_argument("--mode", choices=["stdin", "argv"], default="stdin")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--report-every", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    log = make_logger()
    log("ATLAS FUZZ — coverage-guided crash hunting")

    vm = WslVM(log=log)
    vm.provision()
    if not vm.health_ok():
        log("FATAL: VM unhealthy"); return 1

    source = Path(args.source).read_text() if args.source else None
    env = BinaryFuzzEnv(vm, source=source, mode=args.mode, log=log)
    proposer = InputProposer(mode=args.mode)
    trainer = SelfTrainer(env, proposer, device=args.device, log=log,
                          embed_fn=embed_input, family_fn=input_family,
                          checkpoint_dir="experiments/atlas_fuzz_self")

    trainer.run(max_steps=args.steps, budget_seconds=args.budget,
                report_every=args.report_every)

    s = env.summary()
    log("")
    log(f"functions covered : {s['functions_covered']}")
    log(f"unique crashes    : {s['unique_crashes']}")
    for kind, hx in s["crash_inputs"].items():
        log(f"  [{kind}] input(hex)={hx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

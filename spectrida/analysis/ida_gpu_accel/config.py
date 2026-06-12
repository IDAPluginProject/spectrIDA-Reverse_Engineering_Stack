"""
ida_gpu_accel/config.py
Central toggles — edit here or override via environment variables.

  IDA_GPU=0          disable GPU (CPU-only mode)
  IDA_CPU_THREADS=N  override thread count
  IDA_PRESEED=0      disable pre-seeding IDA with scan results
"""
import os

import torch

# ── GPU ─────────���─────────────────────────────────────────────────────────────
# Scanning phases (prologues, BL targets, basic block boundaries, strings)
GPU_ENABLED: bool = (
    os.environ.get("IDA_GPU", "1") != "0"
    and torch.cuda.is_available()
)

# ─�� CPU ────────────────��────────────────────────────���─────────────────────────
# Always active alongside GPU.
# Controls: shard worker count, CPU fallback scanner threads, numpy parallel ops.
CPU_THREADS: int = int(os.environ.get("IDA_CPU_THREADS", "8"))

# ── Pre-seeder ──────────────��─────────────────────────────────────────────────
# Feed GPU/CPU scan results into IDA before auto_wait().
# Disable only for benchmarking — analysis quality unchanged either way.
PRESEED_ENABLED: bool = os.environ.get("IDA_PRESEED", "1") != "0"

# ── Derived ───────────────────────────────────────────────────────────────────
DEVICE: str = "cuda" if GPU_ENABLED else "cpu"

def status() -> str:
    lines = [
        f"  GPU_ENABLED   = {GPU_ENABLED}  (device: {DEVICE})",
        f"  CPU_THREADS   = {CPU_THREADS}",
        f"  PRESEED       = {PRESEED_ENABLED}",
    ]
    if GPU_ENABLED:
        props = torch.cuda.get_device_properties(0)
        lines.append(f"  GPU           = {props.name}  {props.total_memory // 1024**2} MB")
    return "\n".join(lines)

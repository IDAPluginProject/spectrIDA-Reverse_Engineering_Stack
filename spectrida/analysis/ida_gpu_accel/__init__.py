"""
ida_gpu_accel
=============
Toggleable GPU+CPU acceleration for IDA Pro auto-analysis.

Usage in shard_worker.py:
    from ida_gpu_accel import preseeder, config
    print(config.status())
    preseeder.seed_from_binary(raw_bytes, base_ea,
                               text_start=shard_start, text_end=shard_end)
    ida_auto.plan_range(shard_start, shard_end)
    ida_auto.auto_wait()

Env vars:
    IDA_GPU=0           disable CUDA, fall back to CPU-only
    IDA_CPU_THREADS=N   override thread count (default 8)
    IDA_PRESEED=0       disable pre-seeding (scan still runs for benchmarking)
"""

from .arm64_scanner import scan
from .config import CPU_THREADS, DEVICE, GPU_ENABLED, PRESEED_ENABLED, status
from .preseeder import seed, seed_from_binary

__all__ = [
    "GPU_ENABLED", "CPU_THREADS", "PRESEED_ENABLED", "DEVICE",
    "status", "scan", "seed", "seed_from_binary",
]

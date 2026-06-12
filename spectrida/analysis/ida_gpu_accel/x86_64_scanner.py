"""
ida_gpu_accel/x86_64_scanner.py

GPU+CPU scanner for x86_64 binaries.

Detects function prologues by scanning for common x86_64 entry patterns:
  - PUSH RBP (55) preceded by C3/CC/90/00 boundary byte
  - SUB RSP, N  (48 83 EC xx)
  - MOV [RSP+N], RBX  (48 89 5C 24)
  - PUSH RBX + REX prefix sequences

GPU path: PyTorch byte-level tensor operations.
CPU path: numpy vectorised byte scan.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import CPU_THREADS, DEVICE, GPU_ENABLED

# Bytes that can immediately precede a function start (padding/terminator)
_BOUNDARY_BYTES = frozenset([0xC3, 0xC2, 0xCC, 0x90, 0x00, 0xCB, 0xCF])

# 3-byte sequences that are strong function-start indicators
# Each entry: (offset_of_first_byte, byte0, byte1, byte2)
_SEQ3 = [
    (0x48, 0x83, 0xEC),  # SUB RSP, imm8
    (0x48, 0x89, 0x5C),  # MOV [RSP+N], RBX
    (0x48, 0x89, 0x4C),  # MOV [RSP+N], RCX
    (0x48, 0x81, 0xEC),  # SUB RSP, imm32
    (0x55, 0x48, 0x89),  # PUSH RBP; MOV RBP, RSP (classic)
]


def _x86_prologues_numpy(data: bytes, base_ea: int) -> list[int]:
    """Find x86_64 function starts using numpy byte scan."""
    arr = np.frombuffer(data, dtype=np.uint8)
    n   = len(arr)
    hits: set[int] = set()

    # Pattern 1: PUSH RBP (0x55) at byte following a boundary byte
    push_rbp = np.where(arr == 0x55)[0]
    for idx in push_rbp:
        if idx == 0:
            hits.add(base_ea)
            continue
        prev = int(arr[idx - 1])
        if prev in _BOUNDARY_BYTES:
            hits.add(base_ea + int(idx))
        # Also catch aligned starts (16-byte aligned) even without boundary
        elif (base_ea + int(idx)) % 16 == 0:
            hits.add(base_ea + int(idx))

    # Pattern 2: 48 83 EC (SUB RSP) — very reliable function start
    if n >= 3:
        m = (arr[:-2] == 0x48) & (arr[1:-1] == 0x83) & (arr[2:] == 0xEC)
        for idx in np.where(m)[0]:
            hits.add(base_ea + int(idx))

    # Pattern 3: 55 48 89 E5 (PUSH RBP; MOV RBP, RSP)
    if n >= 4:
        m = (arr[:-3] == 0x55) & (arr[1:-2] == 0x48) & (arr[2:-1] == 0x89) & (arr[3:] == 0xE5)
        for idx in np.where(m)[0]:
            hits.add(base_ea + int(idx))

    # Pattern 4: 48 81 EC (SUB RSP, imm32)
    if n >= 3:
        m = (arr[:-2] == 0x48) & (arr[1:-1] == 0x81) & (arr[2:] == 0xEC)
        for idx in np.where(m)[0]:
            hits.add(base_ea + int(idx))

    return sorted(hits)


def _gpu_scan_x86(data: bytes, base_ea: int) -> list[int]:
    """GPU-accelerated x86_64 prologue scan using PyTorch."""
    import torch

    t0 = time.perf_counter()
    arr_np = np.frombuffer(data, dtype=np.uint8).copy()
    t = torch.from_numpy(arr_np.astype(np.int32)).to(DEVICE)
    n = len(t)
    hits: set[int] = set()

    # Pattern: PUSH RBP (0x55) with boundary predecessor
    push_mask = (t == 0x55)
    if n > 1:
        prev  = torch.cat([torch.tensor([-1], device=DEVICE), t[:-1]])
        boundary = (
            (prev == 0xC3) | (prev == 0xCC) | (prev == 0x90) |
            (prev == 0x00) | (prev == 0xC2) | (prev == 0xCB)
        )
        push_hits = (push_mask & boundary).nonzero(as_tuple=True)[0].cpu().tolist()
        for idx in push_hits:
            hits.add(base_ea + idx)
        # 16-byte aligned PUSH RBP even without boundary
        aligned = torch.arange(n, device=DEVICE)
        aligned_push = (push_mask & (((aligned + base_ea) % 16) == 0)).nonzero(as_tuple=True)[0].cpu().tolist()
        for idx in aligned_push:
            hits.add(base_ea + idx)

    # Pattern: 48 83 EC (SUB RSP, imm8)
    if n >= 3:
        m = (t[:-2] == 0x48) & (t[1:-1] == 0x83) & (t[2:] == 0xEC)
        for idx in m.nonzero(as_tuple=True)[0].cpu().tolist():
            hits.add(base_ea + idx)

    # Pattern: 55 48 89 E5 (PUSH RBP; MOV RBP,RSP)
    if n >= 4:
        m = (t[:-3] == 0x55) & (t[1:-2] == 0x48) & (t[2:-1] == 0x89) & (t[3:] == 0xE5)
        for idx in m.nonzero(as_tuple=True)[0].cpu().tolist():
            hits.add(base_ea + idx)

    # Pattern: 48 81 EC (SUB RSP, imm32)
    if n >= 3:
        m = (t[:-2] == 0x48) & (t[1:-1] == 0x81) & (t[2:] == 0xEC)
        for idx in m.nonzero(as_tuple=True)[0].cpu().tolist():
            hits.add(base_ea + idx)

    dt = time.perf_counter() - t0
    result = sorted(hits)
    print(f"[ida_gpu_accel] x86_64 GPU scan: {len(result)} prologues ({dt:.2f}s)", flush=True)
    return result


def _cpu_chunk(args):
    chunk, offset, base_ea = args
    return _x86_prologues_numpy(chunk, base_ea + offset)


def scan_x86_64(data: bytes, base_ea: int) -> tuple[list[int], list[int], list[int], list[tuple[int, str]]]:
    """
    Scan x86_64 code. Returns (prologues, call_targets, bb_heads, strings).
    call_targets and bb_heads are empty (too expensive to resolve without disasm).
    """
    from .arm64_scanner import _cpu_string_scan

    if GPU_ENABLED:
        try:
            prologues = _gpu_scan_x86(data, base_ea)
            strings   = _cpu_string_scan(data, base_ea)
            return prologues, [], prologues, strings
        except Exception as e:
            print(f"[ida_gpu_accel] x86_64 GPU scan failed ({e}), falling back", flush=True)

    # CPU fallback — chunked threadpool
    t0 = time.perf_counter()
    n = len(data)
    chunk_size = max(1024, n // CPU_THREADS)
    chunks = [(data[i:i+chunk_size], i, base_ea) for i in range(0, n, chunk_size)]
    all_hits: set[int] = set()
    with ThreadPoolExecutor(max_workers=CPU_THREADS) as pool:
        for hits in pool.map(_cpu_chunk, chunks):
            all_hits.update(hits)
    from .arm64_scanner import _cpu_string_scan
    strings = _cpu_string_scan(data, base_ea)
    dt = time.perf_counter() - t0
    result = sorted(all_hits)
    print(f"[ida_gpu_accel] x86_64 CPU scan ({CPU_THREADS}T): {len(result)} prologues ({dt:.2f}s)", flush=True)
    return result, [], result, strings

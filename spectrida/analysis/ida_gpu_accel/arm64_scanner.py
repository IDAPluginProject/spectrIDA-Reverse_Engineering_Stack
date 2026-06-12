"""
ida_gpu_accel/arm64_scanner.py

Scans ARM64 binary data for:
  1. Function prologues  (STP X29,X30,[SP,#-N]!)
  2. BL call targets     (BL <imm26>)
  3. Basic-block heads   (branch targets of B/BL/Bcc/CBZ/CBNZ/TBZ/TBNZ)
  4. ASCII strings       (len >= 5, printable)

GPU path  : PyTorch CUDA — one vectorised kernel per scan type.
CPU path  : numpy + concurrent.futures — CPU_THREADS workers.

Both paths return the same result types so callers are path-agnostic.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import CPU_THREADS, DEVICE, GPU_ENABLED

# ── ARM64 constants ────────────────────────────────────────────────────────────
# STP X29, X30, [SP, #-N]!   (pre-indexed, any frame size 16..512)
_PROLOGUE_MASK  = np.uint32(0xFFC07FFF)
_PROLOGUE_VALUE = np.uint32(0xA9807BFD)

# BL   bits[31:26] = 100101
_BL_MASK        = np.uint32(0xFC000000)
_BL_VALUE       = np.uint32(0x94000000)

# B    bits[31:26] = 000101
_B_MASK         = np.uint32(0xFC000000)
_B_VALUE        = np.uint32(0x14000000)

# Bcc  bits[31:24] = 01010100, bit[4] = 0
_BCC_MASK       = np.uint32(0xFF00001F)
_BCC_VALUE      = np.uint32(0x54000000)

# CBZ / CBNZ   bits[31:25] = x011010x
_CBZ_MASK       = np.uint32(0x7E000000)
_CBZ_VALUE      = np.uint32(0x34000000)

# TBZ / TBNZ   bits[31:25] = x011011x
_TBZ_MASK       = np.uint32(0x7E000000)
_TBZ_VALUE      = np.uint32(0x36000000)


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sign_extend_26(v: int) -> int:
    """Sign-extend a 26-bit immediate to Python int."""
    if v & (1 << 25):
        v -= 1 << 26
    return v

def _sign_extend_19(v: int) -> int:
    if v & (1 << 18):
        v -= 1 << 19
    return v

def _sign_extend_14(v: int) -> int:
    if v & (1 << 13):
        v -= 1 << 14
    return v


# ─────────────────────────────────────────────────────────────────────────────
#  GPU path (PyTorch CUDA)
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_scan(data: bytes, base_ea: int) -> tuple[list[int], list[int], list[int], list[tuple[int,str]]]:
    """Run all four scans on CUDA. Returns (prologues, bl_targets, bb_heads, strings)."""
    import torch

    t0 = time.perf_counter()

    # Align to 4-byte boundary
    trim = len(data) - (len(data) % 4)

    # Use writable numpy array → torch int64 (avoids uint32/int32 overflow)
    raw_words = np.frombuffer(data[:trim], dtype=np.uint32).copy()
    # int64 on GPU — all values fit, no sign-overflow
    u64 = torch.from_numpy(raw_words.astype(np.int64)).to(DEVICE)

    def match(mask_np: np.uint32, val_np: np.uint32):
        """Return list of indices where (word & mask) == val."""
        m = int(mask_np)
        v = int(val_np)
        hits = ((u64 & m) == v).nonzero(as_tuple=True)[0]
        return hits.cpu().tolist()

    # ── Prologues ──────────────────────────────────────────────────────────
    prologue_idx = match(_PROLOGUE_MASK, _PROLOGUE_VALUE)
    prologues = [base_ea + idx * 4 for idx in prologue_idx]

    # ── BL targets ─────────────────────────────────────────────────────────
    bl_idx = match(_BL_MASK, _BL_VALUE)
    bl_targets_set: set[int] = set()
    for idx in bl_idx:
        insn = int(raw_words[idx])
        imm26 = insn & 0x03FFFFFF
        target = base_ea + idx * 4 + _sign_extend_26(imm26) * 4
        bl_targets_set.add(target)
    bl_targets = sorted(bl_targets_set)

    # ── Basic-block heads (all branch targets) ──────────────────────────────
    bb_heads_set: set[int] = set(prologues)

    # B unconditional
    for idx in match(_B_MASK, _B_VALUE):
        imm26 = int(raw_words[idx]) & 0x03FFFFFF
        bb_heads_set.add(base_ea + idx * 4 + _sign_extend_26(imm26) * 4)
        bb_heads_set.add(base_ea + (idx + 1) * 4)

    # Bcc
    for idx in match(_BCC_MASK, _BCC_VALUE):
        imm19 = (int(raw_words[idx]) >> 5) & 0x7FFFF
        bb_heads_set.add(base_ea + idx * 4 + _sign_extend_19(imm19) * 4)
        bb_heads_set.add(base_ea + (idx + 1) * 4)

    # CBZ/CBNZ
    for idx in match(_CBZ_MASK, _CBZ_VALUE):
        imm19 = (int(raw_words[idx]) >> 5) & 0x7FFFF
        bb_heads_set.add(base_ea + idx * 4 + _sign_extend_19(imm19) * 4)
        bb_heads_set.add(base_ea + (idx + 1) * 4)

    # TBZ/TBNZ
    for idx in match(_TBZ_MASK, _TBZ_VALUE):
        imm14 = (int(raw_words[idx]) >> 5) & 0x3FFF
        bb_heads_set.add(base_ea + idx * 4 + _sign_extend_14(imm14) * 4)
        bb_heads_set.add(base_ea + (idx + 1) * 4)

    bb_heads = sorted(bb_heads_set)

    # ── ASCII strings (CPU-side — no point doing this on GPU) ──────────────
    strings = _cpu_string_scan(data, base_ea)

    dt = time.perf_counter() - t0
    print(f"[ida_gpu_accel] GPU scan: {len(prologues)} prologues, {len(bl_targets)} BL targets, "
          f"{len(bb_heads)} BB heads, {len(strings)} strings  ({dt:.2f}s)", flush=True)
    return prologues, bl_targets, bb_heads, strings


# ─────────────────────────────────────────────────────────────────────────────
#  CPU path (numpy + threadpool)
# ─────────────────────────────────────────────────────────────────────────────

def _cpu_scan_chunk(args):
    """Scan a byte chunk; returns (prologues, bl_targets, bb_heads) as lists."""
    chunk_data, chunk_offset, base_ea = args
    trim = len(chunk_data) - (len(chunk_data) % 4)
    if trim == 0:
        return [], [], []

    words = np.frombuffer(chunk_data[:trim], dtype=np.uint32)
    offsets = np.arange(len(words), dtype=np.int64) * 4 + chunk_offset

    # Prologues
    pro_mask = (words & _PROLOGUE_MASK) == _PROLOGUE_VALUE
    prologues = (base_ea + offsets[pro_mask]).tolist()

    # BL hits — need to resolve targets individually
    bl_mask  = (words & _BL_MASK) == _BL_VALUE
    bl_idx   = np.where(bl_mask)[0]
    bl_targets_set: set[int] = set()
    for i in bl_idx:
        insn = int(words[i])
        imm26 = insn & 0x03FFFFFF
        tgt = base_ea + int(offsets[i]) + _sign_extend_26(imm26) * 4
        bl_targets_set.add(tgt)

    # BB heads
    bb_set: set[int] = set(prologues)

    # B
    b_mask = (words & _B_MASK) == _B_VALUE
    for i in np.where(b_mask)[0]:
        imm26 = int(words[i]) & 0x03FFFFFF
        bb_set.add(base_ea + int(offsets[i]) + _sign_extend_26(imm26) * 4)
        bb_set.add(base_ea + int(offsets[i]) + 4)

    # Bcc
    bcc_mask = (words & _BCC_MASK) == _BCC_VALUE
    for i in np.where(bcc_mask)[0]:
        imm19 = (int(words[i]) >> 5) & 0x7FFFF
        bb_set.add(base_ea + int(offsets[i]) + _sign_extend_19(imm19) * 4)
        bb_set.add(base_ea + int(offsets[i]) + 4)

    # CBZ/CBNZ
    cbz_mask = (words & _CBZ_MASK) == _CBZ_VALUE
    for i in np.where(cbz_mask)[0]:
        imm19 = (int(words[i]) >> 5) & 0x7FFFF
        bb_set.add(base_ea + int(offsets[i]) + _sign_extend_19(imm19) * 4)
        bb_set.add(base_ea + int(offsets[i]) + 4)

    # TBZ/TBNZ
    tbz_mask = (words & _TBZ_MASK) == _TBZ_VALUE
    for i in np.where(tbz_mask)[0]:
        imm14 = (int(words[i]) >> 5) & 0x3FFF
        bb_set.add(base_ea + int(offsets[i]) + _sign_extend_14(imm14) * 4)
        bb_set.add(base_ea + int(offsets[i]) + 4)

    return prologues, list(bl_targets_set), list(bb_set)


def _cpu_string_scan(data: bytes, base_ea: int, min_len: int = 5) -> list[tuple[int, str]]:
    """Find ASCII strings of length >= min_len."""
    results = []
    i, n = 0, len(data)
    while i < n:
        j = i
        while j < n and 0x20 <= data[j] < 0x7F:
            j += 1
        if j - i >= min_len:
            results.append((base_ea + i, data[i:j].decode("ascii", errors="replace")))
        i = j + 1
    return results


def _cpu_scan(data: bytes, base_ea: int) -> tuple[list[int], list[int], list[int], list[tuple[int,str]]]:
    """CPU-only scan using CPU_THREADS workers."""
    t0 = time.perf_counter()

    n = len(data)
    # Align chunks to 4-byte boundaries
    chunk_size = max(4, (n // CPU_THREADS) & ~3)
    chunks = []
    offset = 0
    while offset < n:
        end = min(offset + chunk_size, n)
        chunks.append((data[offset:end], offset, base_ea))
        offset = end

    all_prologues: list[int] = []
    all_bl: set[int] = set()
    all_bb: set[int] = set()

    with ThreadPoolExecutor(max_workers=CPU_THREADS) as pool:
        futures = [pool.submit(_cpu_scan_chunk, c) for c in chunks]
        for f in futures:
            pros, bls, bbs = f.result()
            all_prologues.extend(pros)
            all_bl.update(bls)
            all_bb.update(bbs)

    strings = _cpu_string_scan(data, base_ea)

    dt = time.perf_counter() - t0
    print(f"[ida_gpu_accel] CPU scan ({CPU_THREADS}T): {len(all_prologues)} prologues, "
          f"{len(all_bl)} BL targets, {len(all_bb)} BB heads, {len(strings)} strings  ({dt:.2f}s)", flush=True)
    return sorted(all_prologues), sorted(all_bl), sorted(all_bb), strings


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def scan(data: bytes, base_ea: int) -> tuple[list[int], list[int], list[int], list[tuple[int,str]]]:
    """
    Scan ARM64 code bytes starting at base_ea.

    Returns:
        prologues  : list of function-start EAs
        bl_targets : list of call-target EAs (from BL instructions)
        bb_heads   : list of basic-block head EAs
        strings    : list of (ea, text) for ASCII strings
    """
    if GPU_ENABLED:
        try:
            return _gpu_scan(data, base_ea)
        except Exception as e:
            print(f"[ida_gpu_accel] GPU scan failed ({e}), falling back to CPU", flush=True)
    return _cpu_scan(data, base_ea)

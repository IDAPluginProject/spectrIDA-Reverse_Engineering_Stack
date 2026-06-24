"""
ida_gpu_accel/arm32_scanner.py

Scans 32-bit ARM binaries (ARM + Thumb/Thumb-2 interworking) for:
  1. Function prologues  (PUSH {.., LR} / PUSH.W {.., LR})
  2. BL/BLX call targets
  3. ASCII strings

Unlike x86_64_scanner.py and arm64_scanner.py, this is Capstone-based, not a
hand-rolled bitmask scanner, and has no GPU path. Reason: ARM32 code switches
between two completely different instruction encodings (4-byte ARM, 2-/4-byte
Thumb-2) with no per-byte tag marking which is active at a given address --
guessing wrong silently produces garbage from the wrong decoder. That's
exactly the bug class that crashed 32-bit ARM analysis before this scanner
existed (see CHANGELOG 0.2.5: a binary scanned with the AArch64 decoder in
one place and the x86_64 decoder in another). Hand-deriving Thumb-2's
bit-level encodings would carry the same risk; Capstone's decoder is
authoritative, so a threaded linear sweep using it is the safer foundation,
even though it's slower per-byte than a vectorised bitmask scan. Typical
Android .so .text sizes (low single-digit MB) make this fast enough without
needing GPU acceleration.

Mode is encoded with the standard ARM EABI convention -- odd address ==
Thumb (real instruction address is ea & ~1), even == ARM -- so the
list[int] contract the rest of the pipeline expects doesn't change shape.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from .config import CPU_THREADS

try:
    import capstone
    from capstone import arm_const
    HAS_CAPSTONE = True
except ImportError:
    HAS_CAPSTONE = False

_CHUNK_OVERLAP = 8  # bytes of lookahead past a chunk's nominal end, so an
                    # instruction straddling the boundary still decodes


def thumb_addr(ea: int) -> int:
    """Tag an address as Thumb mode using the ARM EABI odd-address convention."""
    return ea | 1


def is_thumb(ea: int) -> bool:
    return bool(ea & 1)


def strip_mode(ea: int) -> int:
    """The real, even instruction address, regardless of mode tag."""
    return ea & ~1


def _sweep_chunk(args) -> tuple[set[int], set[int]]:
    """One thread's linear sweep of a byte range in ONE mode.

    ``chunk_data`` may extend _CHUNK_OVERLAP bytes past ``nominal_len`` so an
    instruction starting near the end of this chunk has enough trailing
    bytes to decode -- but a hit only counts if its *start* EA falls within
    [0, nominal_len), so the next chunk (which starts its own sweep right at
    that boundary) doesn't also claim it.
    """
    chunk_data, nominal_len, chunk_offset, base_ea, thumb = args
    if not HAS_CAPSTONE:
        return set(), set()

    md = capstone.Cs(capstone.CS_ARCH_ARM, capstone.CS_MODE_THUMB if thumb else capstone.CS_MODE_ARM)
    md.detail = True
    step = 2 if thumb else 4

    prologues: set[int] = set()
    bl_targets: set[int] = set()

    offset = 0
    while offset < nominal_len:
        ea = base_ea + chunk_offset + offset
        insns = list(md.disasm(chunk_data[offset:offset + 8], ea, count=1))
        if not insns:
            offset += step
            continue
        insn = insns[0]

        if insn.id == arm_const.ARM_INS_PUSH:
            # PUSH {..., LR} is the reliable function-start signal -- a bare
            # PUSH (no LR) also shows up mid-function spilling extra
            # callee-saved regs, so only count it as a prologue when LR is
            # actually in the register list.
            if any(op.type == arm_const.ARM_OP_REG and op.reg == arm_const.ARM_REG_LR
                   for op in insn.operands):
                prologues.add(thumb_addr(ea) if thumb else ea)

        elif insn.id in (arm_const.ARM_INS_BL, arm_const.ARM_INS_BLX):
            for op in insn.operands:
                if op.type == arm_const.ARM_OP_IMM:
                    target = op.imm
                    # BL keeps the caller's mode; BLX (immediate form)
                    # switches it -- that's the whole point of BLX.
                    target_is_thumb = thumb if insn.id == arm_const.ARM_INS_BL else not thumb
                    bl_targets.add(thumb_addr(target) if target_is_thumb else target)

        offset += insn.size if insn.size >= step else step

    return prologues, bl_targets


def scan(data: bytes, base_ea: int) -> tuple[list[int], list[int], list[int], list[tuple[int, str]]]:
    """
    Scan ARM32 code bytes for prologues and BL/BLX targets, sweeping in both
    ARM and Thumb-2 mode (no ELF mapping-symbol lookup yet -- pure
    brute-force, unioned). Each returned EA is mode-tagged via the
    odd-address convention (see module docstring) -- callers must
    strip_mode()/is_thumb() it, not use it as a raw address.

    Returns:
        prologues  : list of mode-tagged function-start EAs
        bl_targets : list of mode-tagged call-target EAs
        bb_heads   : same as prologues -- no separate basic-block tracking
                     yet, kept for API-shape parity with the other scanners
        strings    : list of (ea, text) for ASCII strings
    """
    if not HAS_CAPSTONE:
        raise ImportError("capstone not installed -- pip install capstone")

    t0 = time.perf_counter()
    n = len(data)
    chunk_size = max(64, n // CPU_THREADS)

    jobs = []
    offset = 0
    while offset < n:
        end = min(offset + chunk_size, n)
        nominal_len = end - offset
        padded = data[offset:min(end + _CHUNK_OVERLAP, n)]
        jobs.append((padded, nominal_len, offset, base_ea, True))   # Thumb sweep
        jobs.append((padded, nominal_len, offset, base_ea, False))  # ARM sweep
        offset = end

    all_prologues: set[int] = set()
    all_bl: set[int] = set()
    with ThreadPoolExecutor(max_workers=CPU_THREADS) as pool:
        for pros, bls in pool.map(_sweep_chunk, jobs):
            all_prologues.update(pros)
            all_bl.update(bls)

    from .arm64_scanner import _cpu_string_scan
    strings = _cpu_string_scan(data, base_ea)

    prologues = sorted(all_prologues)
    bl_targets = sorted(all_bl)
    dt = time.perf_counter() - t0
    print(f"[ida_gpu_accel] arm32 scan ({CPU_THREADS}T x2 modes): {len(prologues)} prologues, "
          f"{len(bl_targets)} BL/BLX targets, {len(strings)} strings  ({dt:.2f}s)", flush=True)
    return prologues, bl_targets, prologues, strings

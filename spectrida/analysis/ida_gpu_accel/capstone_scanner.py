"""
ida_gpu_accel/capstone_scanner.py

Full disassembly pass using Capstone — replaces IDA's auto_wait() for discovery.

For each shard:
  1. Disassemble all reachable code (recursive descent from GPU-found entry points)
  2. Build function list with precise start+end boundaries
  3. Extract call graph (direct CALLs)
  4. Detect tail calls (JMP to external function)
  5. Reconstruct basic blocks

CPU: threaded Capstone (one thread per chunk)
GPU: pre-scan entry points first (x86_64_scanner / arm64_scanner),
     then Capstone recursive descent from those seeds

Quality vs pure IDA:
  - Function boundaries: ~95% (misses some compiler-synthesized thunks)
  - Call graph: ~98% (misses indirect calls through vtables/function pointers)
  - Types/pseudocode: unchanged — IDA still does that pass
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import NamedTuple

from .config import GPU_ENABLED

try:
    import capstone
    from capstone import arm64_const, x86_const
    HAS_CAPSTONE = True
except ImportError:
    HAS_CAPSTONE = False


class FuncInfo(NamedTuple):
    ea: int
    size: int
    name: str
    callees: list[int]   # direct call targets
    callers: list[int]   # known callers (filled in post-pass)


class ShardResult(NamedTuple):
    funcs: list[FuncInfo]
    bb_heads: list[int]
    strings: list[tuple[int, str]]
    elapsed_s: float


# ── x86_64 recursive descent ──────────────────────────────────────────────────

def _disasm_x86_func(md, data: bytes, base_ea: int, entry: int,
                     shard_start: int, shard_end: int,
                     visited_funcs: set[int]) -> FuncInfo | None:
    """Disassemble one x86_64 function via recursive descent. Returns FuncInfo or None."""
    if entry < shard_start or entry >= shard_end:
        return None
    if entry in visited_funcs:
        return None
    visited_funcs.add(entry)

    worklist: list[int] = [entry]
    seen_blocks: set[int] = set()
    all_eas: set[int] = set()
    callees: set[int] = set()
    max_ea = entry

    while worklist:
        ea = worklist.pop()
        if ea in seen_blocks or ea < shard_start or ea >= shard_end:
            continue
        seen_blocks.add(ea)

        offset = ea - base_ea
        if offset < 0 or offset >= len(data):
            continue

        # Disassemble up to 512 bytes from this block head
        for insn in md.disasm(data[offset:offset+512], ea):
            all_eas.add(insn.address)
            if insn.address > max_ea:
                max_ea = insn.address

            # CALL → callee seed
            if insn.id in (x86_const.X86_INS_CALL,):
                for op in insn.operands:
                    if op.type == capstone.x86.X86_OP_IMM:
                        callees.add(op.imm)

            # RET / INT3 → end of block
            if insn.id in (x86_const.X86_INS_RET, x86_const.X86_INS_RETF,
                           x86_const.X86_INS_RETFQ, x86_const.X86_INS_INT3,
                           x86_const.X86_INS_UD2):
                break

            # Unconditional JMP
            if insn.id == x86_const.X86_INS_JMP:
                for op in insn.operands:
                    if op.type == capstone.x86.X86_OP_IMM:
                        tgt = op.imm
                        if shard_start <= tgt < shard_end:
                            worklist.append(tgt)
                        else:
                            callees.add(tgt)  # tail call
                break

            # Conditional branch → two successors
            if insn.group(capstone.CS_GRP_JUMP):
                fall = insn.address + insn.size
                worklist.append(fall)
                for op in insn.operands:
                    if op.type == capstone.x86.X86_OP_IMM:
                        worklist.append(op.imm)
                break

    if not all_eas:
        return None

    size = max_ea - entry + 1  # approximate
    return FuncInfo(ea=entry, size=size, name=f"sub_{entry:x}",
                    callees=sorted(callees), callers=[])


def _scan_shard_x86(data: bytes, base_ea: int,
                    shard_start: int, shard_end: int,
                    entry_points: list[int]) -> ShardResult:
    """Full Capstone x86_64 disasm pass on one shard."""
    t0 = time.perf_counter()

    if not HAS_CAPSTONE:
        raise ImportError("capstone not installed — run: pip install capstone")

    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.detail = True

    visited: set[int] = set()
    funcs: list[FuncInfo] = []
    all_callees: set[int] = set()

    for ep in entry_points:
        fi = _disasm_x86_func(md, data, base_ea, ep, shard_start, shard_end, visited)
        if fi:
            funcs.append(fi)
            all_callees.update(fi.callees)

    # Second pass: try callee targets that are in-shard but not yet visited
    for tgt in list(all_callees):
        if shard_start <= tgt < shard_end and tgt not in visited:
            fi = _disasm_x86_func(md, data, base_ea, tgt, shard_start, shard_end, visited)
            if fi:
                funcs.append(fi)
                all_callees.update(fi.callees)

    # Fill in callers
    callee_to_callers: dict[int, list[int]] = defaultdict(list)
    for fi in funcs:
        for c in fi.callees:
            callee_to_callers[c].append(fi.ea)
    funcs = [FuncInfo(ea=fi.ea, size=fi.size, name=fi.name,
                      callees=fi.callees,
                      callers=callee_to_callers.get(fi.ea, []))
             for fi in funcs]

    # BB heads = all function starts
    bb_heads = sorted({fi.ea for fi in funcs})

    # String scan
    from .arm64_scanner import _cpu_string_scan
    strings = _cpu_string_scan(data, base_ea)

    elapsed = time.perf_counter() - t0
    print(f"[capstone] x86_64 shard {shard_start:#x}-{shard_end:#x}: "
          f"{len(funcs)} funcs in {elapsed:.2f}s", flush=True)
    return ShardResult(funcs=funcs, bb_heads=bb_heads, strings=strings, elapsed_s=elapsed)


# ── ARM64 recursive descent ───────────────────────────────────────────────────

def _scan_shard_arm64(data: bytes, base_ea: int,
                      shard_start: int, shard_end: int,
                      entry_points: list[int]) -> ShardResult:
    t0 = time.perf_counter()

    if not HAS_CAPSTONE:
        raise ImportError("capstone not installed — run: pip install capstone")

    md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    md.detail = True

    visited: set[int] = set()
    funcs: list[FuncInfo] = []
    all_callees: set[int] = set()

    def disasm_func(entry: int) -> FuncInfo | None:
        if entry in visited or entry < shard_start or entry >= shard_end:
            return None
        visited.add(entry)
        worklist = [entry]
        seen_blocks: set[int] = set()
        all_eas: set[int] = set()
        callees: set[int] = set()
        max_ea = entry

        while worklist:
            ea = worklist.pop()
            if ea in seen_blocks or ea < shard_start or ea >= shard_end:
                continue
            seen_blocks.add(ea)
            offset = ea - base_ea
            if offset < 0 or offset >= len(data):
                continue

            for insn in md.disasm(data[offset:offset+512], ea):
                all_eas.add(insn.address)
                if insn.address > max_ea:
                    max_ea = insn.address

                if insn.id == arm64_const.ARM64_INS_BL:
                    for op in insn.operands:
                        if op.type == capstone.arm64.ARM64_OP_IMM:
                            callees.add(op.imm)

                if insn.id in (arm64_const.ARM64_INS_RET, arm64_const.ARM64_INS_BRK):
                    break

                if insn.id == arm64_const.ARM64_INS_B:
                    for op in insn.operands:
                        if op.type == capstone.arm64.ARM64_OP_IMM:
                            tgt = op.imm
                            if shard_start <= tgt < shard_end:
                                worklist.append(tgt)
                            else:
                                callees.add(tgt)
                    break

                if insn.group(capstone.CS_GRP_JUMP):
                    fall = insn.address + insn.size
                    worklist.append(fall)
                    for op in insn.operands:
                        if op.type == capstone.arm64.ARM64_OP_IMM:
                            worklist.append(op.imm)
                    break

        if not all_eas:
            return None
        return FuncInfo(ea=entry, size=max_ea - entry + 4, name=f"sub_{entry:x}",
                        callees=sorted(callees), callers=[])

    for ep in entry_points:
        fi = disasm_func(ep)
        if fi:
            funcs.append(fi)
            all_callees.update(fi.callees)

    for tgt in list(all_callees):
        if shard_start <= tgt < shard_end and tgt not in visited:
            fi = disasm_func(tgt)
            if fi:
                funcs.append(fi)

    callee_to_callers: dict[int, list[int]] = defaultdict(list)
    for fi in funcs:
        for c in fi.callees:
            callee_to_callers[c].append(fi.ea)
    funcs = [FuncInfo(ea=fi.ea, size=fi.size, name=fi.name,
                      callees=fi.callees,
                      callers=callee_to_callers.get(fi.ea, []))
             for fi in funcs]

    from .arm64_scanner import _cpu_string_scan
    strings = _cpu_string_scan(data, base_ea)

    elapsed = time.perf_counter() - t0
    print(f"[capstone] arm64 shard {shard_start:#x}-{shard_end:#x}: "
          f"{len(funcs)} funcs in {elapsed:.2f}s", flush=True)
    return ShardResult(funcs=funcs, bb_heads=sorted({fi.ea for fi in funcs}),
                       strings=strings, elapsed_s=elapsed)


# ── ARM32 (ARM + Thumb-2) recursive descent ───────────────────────────────────
# Entry points and callees are mode-tagged via arm32_scanner's odd-address
# convention (is_thumb()/strip_mode()) so this can switch Capstone instances
# (CS_MODE_ARM vs CS_MODE_THUMB) per block instead of guessing one mode for
# the whole shard -- that per-address-not-per-shard mode tracking is the
# entire reason this isn't just a third arch branch reusing _scan_shard_arm64.

def _scan_shard_arm32(data: bytes, base_ea: int,
                      shard_start: int, shard_end: int,
                      entry_points: list[int]) -> ShardResult:
    t0 = time.perf_counter()

    if not HAS_CAPSTONE:
        raise ImportError("capstone not installed — run: pip install capstone")

    from capstone import arm_const

    from .arm32_scanner import is_thumb, strip_mode, thumb_addr

    md_arm = capstone.Cs(capstone.CS_ARCH_ARM, capstone.CS_MODE_ARM)
    md_arm.detail = True
    md_thumb = capstone.Cs(capstone.CS_ARCH_ARM, capstone.CS_MODE_THUMB)
    md_thumb.detail = True

    visited: set[int] = set()       # mode-tagged entry EAs already disassembled
    funcs: list[FuncInfo] = []
    all_callees: set[int] = set()   # mode-tagged

    def disasm_func(entry_tagged: int) -> FuncInfo | None:
        entry = strip_mode(entry_tagged)
        if entry_tagged in visited or entry < shard_start or entry >= shard_end:
            return None
        visited.add(entry_tagged)

        worklist: list[int] = [entry_tagged]   # mode-tagged work items
        seen_blocks: set[int] = set()
        all_eas: set[int] = set()
        callees: set[int] = set()
        max_ea = entry

        while worklist:
            cur = worklist.pop()
            if cur in seen_blocks:
                continue
            seen_blocks.add(cur)
            cur_thumb = is_thumb(cur)
            cur_ea = strip_mode(cur)
            if cur_ea < shard_start or cur_ea >= shard_end:
                continue
            offset = cur_ea - base_ea
            if offset < 0 or offset >= len(data):
                continue

            md = md_thumb if cur_thumb else md_arm
            for insn in md.disasm(data[offset:offset + 512], cur_ea):
                all_eas.add(insn.address)
                if insn.address > max_ea:
                    max_ea = insn.address

                if insn.id in (arm_const.ARM_INS_BL, arm_const.ARM_INS_BLX):
                    for op in insn.operands:
                        if op.type == arm_const.ARM_OP_IMM:
                            tgt = op.imm
                            # BLX flips mode, BL doesn't.
                            tgt_thumb = cur_thumb if insn.id == arm_const.ARM_INS_BL else not cur_thumb
                            callees.add(thumb_addr(tgt) if tgt_thumb else tgt)
                    continue

                # Returns: POP {..,PC} (Thumb) / BX LR / LDM {..,PC}
                if insn.id == arm_const.ARM_INS_POP and any(
                    op.type == arm_const.ARM_OP_REG and op.reg == arm_const.ARM_REG_PC
                    for op in insn.operands
                ):
                    break
                if insn.id == arm_const.ARM_INS_BX and any(
                    op.type == arm_const.ARM_OP_REG and op.reg == arm_const.ARM_REG_LR
                    for op in insn.operands
                ):
                    break

                if insn.group(capstone.CS_GRP_JUMP):
                    # cc == ARM_CC_AL ("always") means this branch has no
                    # real condition -- an unconditional B, so the fallthrough
                    # path is unreachable and shouldn't be queued as a successor.
                    is_cond = insn.cc != arm_const.ARM_CC_AL
                    for op in insn.operands:
                        if op.type == arm_const.ARM_OP_IMM:
                            tgt = op.imm
                            tgt_tagged = thumb_addr(tgt) if cur_thumb else tgt
                            if shard_start <= tgt < shard_end:
                                worklist.append(tgt_tagged)
                            else:
                                callees.add(tgt_tagged)
                    if is_cond:
                        fall = insn.address + insn.size
                        worklist.append(thumb_addr(fall) if cur_thumb else fall)
                    break

        if not all_eas:
            return None
        return FuncInfo(ea=entry_tagged, size=max_ea - entry + 4, name=f"sub_{entry:x}",
                        callees=sorted(callees), callers=[])

    for ep in entry_points:
        fi = disasm_func(ep)
        if fi:
            funcs.append(fi)
            all_callees.update(fi.callees)

    for tgt in list(all_callees):
        if strip_mode(tgt) >= shard_start and strip_mode(tgt) < shard_end and tgt not in visited:
            fi = disasm_func(tgt)
            if fi:
                funcs.append(fi)

    callee_to_callers: dict[int, list[int]] = defaultdict(list)
    for fi in funcs:
        for c in fi.callees:
            callee_to_callers[c].append(fi.ea)
    funcs = [FuncInfo(ea=fi.ea, size=fi.size, name=fi.name,
                      callees=fi.callees,
                      callers=callee_to_callers.get(fi.ea, []))
             for fi in funcs]

    from .arm64_scanner import _cpu_string_scan
    strings = _cpu_string_scan(data, base_ea)

    elapsed = time.perf_counter() - t0
    print(f"[capstone] arm32 shard {shard_start:#x}-{shard_end:#x}: "
          f"{len(funcs)} funcs in {elapsed:.2f}s", flush=True)
    return ShardResult(funcs=funcs, bb_heads=sorted({fi.ea for fi in funcs}),
                       strings=strings, elapsed_s=elapsed)


# ── Public API ────────────────────────────────────────────────────────────────

def scan_shard(data: bytes, base_ea: int,
               shard_start: int, shard_end: int,
               arch: str = "x86_64",
               entry_points: list[int] | None = None) -> ShardResult:
    """
    Full Capstone disasm pass on a shard. Gets entry points from GPU/CPU scanner
    first, unless the caller already has a precomputed list (entry_points) --
    e.g. from a global whole-binary scan, which finds call targets a narrow
    per-shard scan would miss simply because the calling instruction lives in
    a different shard than its target.

    arch: "x86_64", "arm64", or "arm32"
    """
    if entry_points is not None:
        if arch == "x86_64":
            return _scan_shard_x86(data, base_ea, shard_start, shard_end, entry_points)
        if arch == "arm32":
            return _scan_shard_arm32(data, base_ea, shard_start, shard_end, entry_points)
        return _scan_shard_arm64(data, base_ea, shard_start, shard_end, entry_points)

    # Step 1: GPU/CPU fast scan to seed entry points
    if arch == "arm32":
        from .arm32_scanner import scan as _arm32_scan
        prologues, bl_targets, _, _ = _arm32_scan(data, base_ea)
        entry_points = sorted(set(prologues) | set(bl_targets))
        return _scan_shard_arm32(data, base_ea, shard_start, shard_end, entry_points)
    if arch == "x86_64":
        if GPU_ENABLED:
            try:
                from .x86_64_scanner import _gpu_scan_x86
                entry_points = _gpu_scan_x86(data, base_ea)
            except Exception as e:
                print(f"[capstone] GPU seed failed ({e}), using CPU seed", flush=True)
                from .x86_64_scanner import _x86_prologues_numpy
                entry_points = _x86_prologues_numpy(data, base_ea)
        else:
            from .x86_64_scanner import _x86_prologues_numpy
            entry_points = _x86_prologues_numpy(data, base_ea)
        return _scan_shard_x86(data, base_ea, shard_start, shard_end, entry_points)
    else:
        # Prologues alone miss most functions -- not every ARM64 compiler emits
        # the exact `stp x29,x30,[sp,#-N]!` pattern (leaf functions skip it
        # entirely, others use a non-pre-indexed stp after a separate `sub sp`).
        # BL targets are a far more reliable entry-point signal: every called
        # function shows up there regardless of its prologue shape.
        if GPU_ENABLED:
            try:
                from .arm64_scanner import _gpu_scan
                prologues, bl_targets, _, _ = _gpu_scan(data, base_ea)
                entry_points = sorted(set(prologues) | set(bl_targets))
            except Exception as e:
                print(f"[capstone] GPU seed failed ({e}), using CPU seed", flush=True)
                from .arm64_scanner import _cpu_scan
                prologues, bl_targets, _, _ = _cpu_scan(data, base_ea)
                entry_points = sorted(set(prologues) | set(bl_targets))
        else:
            from .arm64_scanner import _cpu_scan
            prologues, bl_targets, _, _ = _cpu_scan(data, base_ea)
            entry_points = sorted(set(prologues) | set(bl_targets))
        return _scan_shard_arm64(data, base_ea, shard_start, shard_end, entry_points)

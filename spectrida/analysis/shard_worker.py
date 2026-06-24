"""
shard_worker.py
Run inside ONE idalib subprocess. Analyzes a specific address range of a binary.

Args: <binary_path> <shard_start_hex> <shard_end_hex> <result_json_path>

Strategy:
  1. open_database(binary, run_auto_analysis=False)
  2. Mark all code segments outside [shard_start, shard_end) as SEG_DATA
  3. GPU fast-scan: find prologues/entry points, seed into IDA
  4. Capstone recursive descent: build full function list
  5. Apply Capstone results into IDA
  6. auto_wait() for type propagation only (discovery already done)
  7. Export functions + names to JSON
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Arch values a FormatHandler can report for which no ida_gpu_accel scanner
# module / capstone_scanner.scan_shard() path exists. Empty now that arm32
# has a real scanner (arm32_scanner.py + capstone_scanner._scan_shard_arm32)
# -- kept as a named set (not removed outright) so the next genuinely
# unsupported arch has an obvious place to land instead of silently
# misrouting its bytes through the wrong decoder.
_UNSUPPORTED_SCAN_ARCHES: set[str] = set()

IDA_DIR   = os.environ.get("SPECTRIDA_IDALIB") or r"C:\Program Files\IDA Professional 9.1"
ACCEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, IDA_DIR)
sys.path.insert(0, ACCEL_DIR)

binary       = sys.argv[1]
shard_start  = int(sys.argv[2], 16)
shard_end    = int(sys.argv[3], 16)
result_path  = sys.argv[4]
arch_hint    = sys.argv[5] if len(sys.argv) > 5 else None
entries_path = sys.argv[6] if len(sys.argv) > 6 else None

def log(msg: str):
    print(f"[shard {shard_start:#x}] {msg}", flush=True)

from spectrida.analysis.formats import detect as detect_format

handler = detect_format(binary)
image = handler.prepare(binary, workdir=os.path.dirname(binary))

import idapro

idapro.enable_console_messages(False)

rc = idapro.open_database(image.binary_path, run_auto_analysis=False)
if rc != 0:
    sys.exit(f"open_database failed rc={rc}")

# IDA has no native NSO loader -- open_database() above just dumps the raw
# (possibly still LZ4-compressed) file bytes into whatever segment its
# generic Binary File fallback creates, at the wrong base, on the wrong
# processor. post_open() (no-op for formats IDA loads natively) is where a
# handler like NSO decompresses + remaps it properly before any scanning
# happens.
handler.post_open()

import ida_bytes
import ida_segment
import idaapi
import idautils
import idc

# ── Detect arch ───────────────────────────────────────────────────────────────
# Priority: explicit CLI override > the handler's own hint > IDA's own
# detection. A caller that already knows the binary format (e.g. NSO ==
# always AArch64 on Switch) can hand it down directly -- IDA's headless
# binary loader has no native NSO support and silently defaults to metapc,
# so procname can't be trusted to tell us this on its own.
if arch_hint:
    arch = arch_hint
elif image.arch:
    arch = image.arch
else:
    try:
        info = idaapi.get_inf_structure()
        proc = info.procname.lower() if hasattr(info, "procname") else ""
        # "arm" alone matches both 32-bit ARM and AArch64 procnames -- check
        # the 64-bit markers first so 32-bit ARM doesn't get misclassified
        # as arm64 (there's no scanner support for either right now if it's
        # genuinely 32-bit, but at least it's labeled correctly).
        if "aarch64" in proc or "arm64" in proc:
            arch = "arm64"
        elif "arm" in proc:
            arch = "arm32"
        else:
            arch = "x86_64"
    except Exception:
        arch = "x86_64"

log(f"arch={arch}")

# ── Mark segments outside shard as DATA ──────────────────────────────────────
for seg_ea in list(idautils.Segments()):
    seg   = ida_segment.getseg(seg_ea)
    stype = idc.get_segm_attr(seg_ea, idc.SEGATTR_TYPE)
    if stype == idc.SEG_CODE:
        if seg.end_ea <= shard_start or seg.start_ea >= shard_end:
            idc.set_segm_type(seg_ea, idc.SEG_DATA)

t_start = time.time()

raw = ida_bytes.get_bytes(shard_start, shard_end - shard_start)

# ── Entry points ───────────────────────────────────────────────────────────────
# A precomputed global entry-points file (entries_path) takes priority -- a
# per-shard scan only ever sees BL targets whose CALLING instruction happens
# to live in this same narrow shard, missing every cross-shard call. A single
# whole-binary scan done once up front doesn't have that blind spot.
entry_points = []
if entries_path:
    try:
        all_entries = json.loads(Path(entries_path).read_text())
        entry_points = [ea for ea in all_entries if shard_start <= ea < shard_end]
        log(f"global entry points: {len(entry_points)} in range")
    except Exception as _e:
        log(f"failed to load global entries ({_e}), falling back to local scan")
        entries_path = None

if not entries_path and arch in _UNSUPPORTED_SCAN_ARCHES:
    log(f"no GPU scanner for arch={arch!r}, skipping (will rely on whatever IDA's own analysis finds)")
elif not entries_path:
    try:
        if arch == "x86_64":
            from ida_gpu_accel.config import GPU_ENABLED
            from ida_gpu_accel.x86_64_scanner import _gpu_scan_x86, _x86_prologues_numpy
            if raw:
                if GPU_ENABLED:
                    entry_points = _gpu_scan_x86(raw, shard_start)
                else:
                    entry_points = _x86_prologues_numpy(raw, shard_start)
                log(f"GPU scan: {len(entry_points)} entry points")
        elif arch == "arm32":
            # No GPU path -- arm32_scanner.scan() is a threaded Capstone
            # sweep (see its module docstring for why a bitmask scanner
            # isn't safe here: ARM/Thumb mode ambiguity).
            from ida_gpu_accel.arm32_scanner import scan as scan_arm32
            if raw:
                prologues, bl_targets, _, _ = scan_arm32(raw, shard_start)
                entry_points = sorted(set(prologues) | set(bl_targets))
                log(f"CPU scan: {len(entry_points)} entry points")
        else:
            from ida_gpu_accel.arm64_scanner import scan
            if raw:
                # Prologue pattern alone misses leaf functions / non-standard
                # frame setups -- BL targets (real call destinations) catch those.
                prologues, bl_targets, _, _ = scan(raw, shard_start)
                entry_points = sorted(set(prologues) | set(bl_targets))
                log(f"GPU scan: {len(entry_points)} entry points")
    except Exception as _e:
        log(f"GPU scan error (non-fatal): {_e}")

# ── Capstone recursive descent ────────────────────────────────────────────────
# We write JSON directly from Capstone results — no add_func() in workers.
# add_func() causes idalib C++ crashes under parallel load.
# The merge pass applies all functions to a single IDA instance safely.
capstone_funcs: list[dict] = []

try:
    if arch in _UNSUPPORTED_SCAN_ARCHES:
        raise RuntimeError(f"no capstone scanner for arch={arch!r}")

    from ida_gpu_accel.capstone_scanner import HAS_CAPSTONE, scan_shard
    if not HAS_CAPSTONE:
        raise ImportError("capstone not installed")
    if not raw:
        raise RuntimeError("no raw bytes")

    log(f"Capstone pass starting ({len(entry_points)} seeds)...")
    result = scan_shard(raw, shard_start, shard_start, shard_end, arch=arch,
                       entry_points=entry_points if entries_path else None)

    # arm32 EAs are mode-tagged (odd == Thumb, see arm32_scanner.py) for the
    # scanner/disassembler's own benefit -- IDA's add_func() needs the real,
    # even address, but the merge step (parallel_analyze.py) still needs to
    # know which functions were Thumb so it can mark the T register before
    # add_func(), or IDA decodes the bytes as ARM and gets garbage. Stripping
    # the address while keeping a separate "thumb" flag is a no-op for every
    # other arch (x86_64/arm64 EAs are already even, thumb is always False).
    for fi in result.funcs:
        ea = fi.ea & ~1
        if shard_start <= ea < shard_end:
            capstone_funcs.append({"ea": ea, "name": f"sub_{ea:x}", "size": getattr(fi, "size", 0),
                                    "callers": [], "thumb": bool(fi.ea & 1)})

    log(f"Capstone: {len(capstone_funcs)} funcs found")

except Exception as _exc:
    log(f"Capstone failed ({_exc}), falling back to GPU preseed only")
    for raw_ea in entry_points:
        ea = raw_ea & ~1
        if shard_start <= ea < shard_end:
            capstone_funcs.append({"ea": ea, "name": f"sub_{ea:x}", "size": 0,
                                    "callers": [], "thumb": bool(raw_ea & 1)})

elapsed = time.time() - t_start
log(f"done in {elapsed:.1f}s")

# ── Export ────────────────────────────────────────────────────────────────────
funcs = capstone_funcs

Path(result_path).write_text(json.dumps({
    "shard_start": shard_start,
    "shard_end":   shard_end,
    "elapsed_s":   elapsed,
    "funcs":       funcs,
}))

log(f"{len(funcs)} funcs exported")
try:
    idapro.close_database()
except Exception:
    pass

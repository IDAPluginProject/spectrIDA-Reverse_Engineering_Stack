"""
parallel_analyze.py
Parallel IDA auto-analysis — splits a binary into N shards, runs N idalib
instances simultaneously, merges results into a master .i64.

Usage:
  python parallel_analyze.py <binary> [--workers N] [--out output.i64]

Default workers = os.cpu_count() // 2 (leave room for idalib memory)
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

IDA_DIR   = os.environ.get("SPECTRIDA_IDALIB") or r"C:\Program Files\IDA Professional 9.1"
IDAT_EXE  = str(Path(IDA_DIR) / "idat.exe")
PYTHON    = sys.executable
WORKER    = str(Path(__file__).parent / "shard_worker.py")
MERGE_IDC = str(Path(__file__).parent / "merge_shards.idc")

sys.path.insert(0, IDA_DIR)


# ── Step 1: Discovery pass ─────────────────────────────────────────────────────
# Open binary quickly to find code segment boundaries WITHOUT full analysis.
# We only need to know the .text range to shard it.

DISCOVER_SCRIPT = """
import sys, json, struct

with open(sys.argv[1], 'rb') as f:
    h = f.read(0x1000)

pe_off       = struct.unpack_from('<I', h, 0x3C)[0]
# COFF header
num_sections = struct.unpack_from('<H', h, pe_off + 6)[0]
opt_hdr_size = struct.unpack_from('<H', h, pe_off + 20)[0]   # correct offset
# Optional header magic: 0x010B=PE32, 0x020B=PE32+
opt_magic    = struct.unpack_from('<H', h, pe_off + 24)[0]
is64         = opt_magic == 0x020B
# ImageBase: offset 28 in PE32 opt hdr, offset 24 in PE32+ opt hdr
ibase_off    = pe_off + 24 + (24 if is64 else 28)
image_base   = struct.unpack_from('<Q' if is64 else '<I', h, ibase_off)[0]

sect_off = pe_off + 24 + opt_hdr_size
min_start = None
max_end   = None
for i in range(num_sections):
    o    = sect_off + i * 40
    name  = h[o:o+8].rstrip(b'\\x00').decode('ascii', errors='replace')
    vsize = struct.unpack_from('<I', h, o+8)[0]
    vaddr = struct.unpack_from('<I', h, o+12)[0]
    flags = struct.unpack_from('<I', h, o+36)[0]
    if flags & 0x20:   # IMAGE_SCN_CNT_CODE
        start = image_base + vaddr
        end   = start + vsize
        if min_start is None or start < min_start:
            min_start = start
        if max_end is None or end > max_end:
            max_end = end
        print(f"# code section '{name}': {start:#x} - {end:#x}", file=sys.stderr)
if min_start is not None:
    print(json.dumps({"start": min_start, "end": max_end, "size": max_end - min_start}))
else:
    print(json.dumps({"error": "no code segment found"}))
"""


def _pe_sections(binary: str):
    """Return list of (name, va, raw_off, raw_size) for all PE sections."""
    with open(binary, "rb") as f:
        h = f.read(0x1000)
    pe_off = struct.unpack_from("<I", h, 0x3C)[0]
    num_sects   = struct.unpack_from("<H", h, pe_off + 6)[0]
    opt_sz      = struct.unpack_from("<H", h, pe_off + 20)[0]
    sect_off    = pe_off + 24 + opt_sz
    sects = []
    for i in range(num_sects):
        o = sect_off + i * 40
        name     = h[o:o+8].rstrip(b"\x00").decode("ascii", errors="replace")
        vsize    = struct.unpack_from("<I", h, o+8)[0]
        vaddr    = struct.unpack_from("<I", h, o+12)[0]
        raw_size = struct.unpack_from("<I", h, o+16)[0]
        raw_off  = struct.unpack_from("<I", h, o+20)[0]
        sects.append((name, vaddr, raw_off, raw_size, vsize))
    return sects


def _image_base(binary: str) -> int:
    with open(binary, "rb") as f:
        h = f.read(0x1000)
    pe_off  = struct.unpack_from("<I", h, 0x3C)[0]
    machine = struct.unpack_from("<H", h, pe_off + 4)[0]
    is64    = machine == 0x8664 or machine == 0xAA64
    ibase_off = pe_off + 24 + (28 if not is64 else 24)
    fmt = "<Q" if is64 else "<I"
    return struct.unpack_from(fmt, h, ibase_off)[0]


def make_shard_binary(src: str, dst: str, shard_start_va: int, shard_end_va: int) -> None:
    """Copy src → dst, zeroing raw bytes of PE sections that fall entirely outside the shard VA range."""
    import shutil
    shutil.copy2(src, dst)
    base   = _image_base(src)
    sects  = _pe_sections(src)
    rel_s  = shard_start_va - base
    rel_e  = shard_end_va   - base
    with open(dst, "r+b") as f:
        for name, vaddr, raw_off, raw_size, vsize in sects:
            sect_end = vaddr + vsize
            # Zero out sections that don't overlap with our shard at all
            if sect_end <= rel_s or vaddr >= rel_e:
                if raw_off and raw_size:
                    f.seek(raw_off)
                    f.write(b"\x00" * raw_size)
            # Partial overlap: zero the part before the shard
            elif vaddr < rel_s:
                zero_bytes = min(rel_s - vaddr, raw_size)
                if raw_off and zero_bytes > 0:
                    f.seek(raw_off)
                    f.write(b"\x00" * zero_bytes)
            # Partial overlap: zero the part after the shard
            if sect_end > rel_e and vaddr < rel_e:
                start_in_sect = max(rel_e - vaddr, 0)
                zero_off = raw_off + start_in_sect
                zero_bytes = raw_size - start_in_sect
                if zero_off < raw_off + raw_size and zero_bytes > 0:
                    f.seek(zero_off)
                    f.write(b"\x00" * zero_bytes)


def discover_text_range(binary: str) -> tuple[int, int]:
    """Return (start, end) of the primary code segment."""
    script_path = Path(tempfile.mktemp(suffix=".py"))
    script_path.write_text(DISCOVER_SCRIPT)
    result = subprocess.run(
        [PYTHON, str(script_path), binary],
        capture_output=True, text=True,
        cwd=IDA_DIR,
    )
    script_path.unlink(missing_ok=True)
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                d = json.loads(line)
                if "start" in d:
                    return d["start"], d["end"]
            except Exception:
                pass
    raise RuntimeError(f"Could not discover code segment.\nstdout: {result.stdout}\nstderr: {result.stderr}")


# ── Step 2: Per-shard worker ───────────────────────────────────────────────────

def run_shard(binary: str, shard_start: int, shard_end: int, result_path: str) -> dict:
    """Spawn a subprocess running shard_worker.py."""
    t0 = time.time()
    proc = subprocess.run(
        [PYTHON, WORKER, binary, hex(shard_start), hex(shard_end), result_path],
        capture_output=True, text=True,
        cwd=IDA_DIR,
    )
    wall = time.time() - t0
    # Always print worker log lines (they start with [shard ...])
    for line in (proc.stdout or "").splitlines():
        if line.strip():
            print(f"  {line}", flush=True)
    # Try to read the JSON even if returncode != 0 — close_database() sometimes
    # crashes idalib after successfully writing the result file.
    try:
        data = json.loads(Path(result_path).read_text())
        data["wall_s"] = wall
        return data
    except Exception:
        pass
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "no output")[-400:]
        return {"error": err, "shard_start": shard_start, "wall_s": wall}
    return {"error": "no json + rc=0", "shard_start": shard_start, "wall_s": wall}


# ── Step 3: Merge ──────────────────────────────────────────────────────────────
# Open binary fresh, apply all function definitions + names from shard JSONs,
# then run a final auto_wait() to stitch xrefs across shard boundaries.

MERGE_LOADER = """
import sys, json
sys.path.insert(0, __import__("os").environ.get("SPECTRIDA_IDALIB") or r"C:\\Program Files\\IDA Professional 9.1")
import idapro
idapro.enable_console_messages(False)
binary      = sys.argv[1]
shard_jsons = sys.argv[2:]   # list of shard result JSON paths

idapro.open_database(binary, run_auto_analysis=False)

import idc, ida_funcs, idaapi

applied = 0
for path in shard_jsons:
    try:
        data = json.load(open(path))
    except Exception as e:
        print(f"[merge] skip {path}: {e}", flush=True)
        continue
    for fn in data.get("funcs", []):
        ea   = fn["ea"]
        name = fn.get("name", "")
        size = fn.get("size", 0)
        # Create function boundary if IDA doesn't know it yet
        if ida_funcs.get_func(ea) is None:
            if size > 0:
                ida_funcs.add_func(ea, ea + size)
            else:
                ida_funcs.add_func(ea)
        # Apply name if it's a real name (not sub_XXXX)
        if name and not name.startswith("sub_") and not name.startswith("j_"):
            idc.set_name(ea, name, 0x800)  # 0x800 = SN_FORCE (renamed in IDA 9.x)
        applied += 1

print(f"[merge] applied {applied} functions, saving...", flush=True)

import os, pathlib
out_dir = pathlib.Path(os.environ.get("SPECTRIDA_OUTPUT_DIR") or r"C:\\Projects\\parallel_ida\\output")
try:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[merge] output dir: {out_dir}", flush=True)
except BaseException as e:
    print(f"[merge] mkdir failed: {e}", flush=True)

stem = pathlib.Path(sys.argv[1]).stem
out = str(out_dir / (stem + "_parallel.i64"))
print(f"[merge] save path: {out}", flush=True)

saved = False
try:
    idc.save_database(out, 1)  # 1 = DBFL_TEMP: save a copy, don't change internal db path
    print(f"[merge] saved (idc.save_database) -> {out}", flush=True)
    saved = True
except BaseException as e:
    print(f"[merge] idc.save_database raised {type(e).__name__}: {e}", flush=True)

if not saved:
    try:
        idaapi.save_database(out, 1)
        print(f"[merge] saved (idaapi.save_database) -> {out}", flush=True)
        saved = True
    except BaseException as e:
        print(f"[merge] idaapi.save_database raised {type(e).__name__}: {e}", flush=True)

if not saved:
    # Last resort: close with save=True, IDA writes to default .i64 path
    try:
        idapro.close_database(True)
        default_i64 = pathlib.Path(sys.argv[1]).with_suffix(".i64")
        print(f"[merge] saved (close_database) -> {default_i64}", flush=True)
        saved = True
    except BaseException as e:
        print(f"[merge] close_database(True) raised {type(e).__name__}: {e}", flush=True)

if not saved:
    print("[merge] ERROR: all save methods failed", flush=True)
else:
    idapro.close_database()
"""


def merge_shards(binary: str, shard_result_paths: list[str], out_path: str | None = None) -> str:
    """Merge shard JSON results into a master .i64."""
    script_path = Path(tempfile.mktemp(suffix=".py"))
    script_path.write_text(MERGE_LOADER)
    args = [PYTHON, str(script_path), binary] + shard_result_paths
    result = subprocess.run(args, capture_output=True, text=True, cwd=IDA_DIR)
    script_path.unlink(missing_ok=True)
    for line in (result.stdout + result.stderr).splitlines():
        print(line, flush=True)
    return out_path or (binary + "_parallel.i64")


# ── Density-balanced shard partitioning ───────────────────────────────────────

def _density_shards(binary: str, text_start: int, text_end: int,
                    n: int) -> list[tuple[int, int]]:
    """
    Scan the full .text for prologue patterns (GPU if available, else CPU),
    then partition into n shards with equal prologue counts so each worker
    gets roughly the same amount of work.
    Falls back to equal-byte split if scan fails.
    """
    try:
        with open(binary, "rb") as f:
            f.seek(0, 2)
            file_size = f.tell()

        # Read raw .text bytes directly from file using PE section table
        base   = _image_base(binary)
        sects  = _pe_sections(binary)
        rel_s  = text_start - base
        rel_e  = text_end   - base

        # Find the raw file offset for text_start
        raw_bytes = bytearray()
        for name, vaddr, raw_off, raw_size, vsize in sects:
            sect_va_start = vaddr
            sect_va_end   = vaddr + vsize
            if sect_va_end <= rel_s or sect_va_start >= rel_e:
                continue
            # Overlap: read the relevant portion
            overlap_start = max(sect_va_start, rel_s)
            overlap_end   = min(sect_va_end,   rel_e)
            file_off      = raw_off + (overlap_start - sect_va_start)
            byte_count    = min(overlap_end - overlap_start, raw_size - (overlap_start - sect_va_start))
            if byte_count <= 0:
                continue
            with open(binary, "rb") as f:
                f.seek(file_off)
                raw_bytes += f.read(byte_count)

        if not raw_bytes:
            raise RuntimeError("could not read .text bytes")

        data = bytes(raw_bytes)

        # GPU density scan — reuse the same scanner used by workers
        sys.path.insert(0, str(Path(__file__).parent))
        from ida_gpu_accel.config import GPU_ENABLED
        from ida_gpu_accel.x86_64_scanner import _gpu_scan_x86, _x86_prologues_numpy
        try:
            if GPU_ENABLED:
                hits = _gpu_scan_x86(data, text_start)
            else:
                raise RuntimeError("GPU disabled")
        except Exception:
            hits = _x86_prologues_numpy(data, text_start)

        if not hits:
            raise RuntimeError("no prologues found")

        print(f"[parallel_analyze] density scan: {len(hits)} prologues across "
              f"{(text_end-text_start)//1024//1024}MB", flush=True)

        # Partition hits into n equal-count buckets, use bucket boundaries as shard edges
        hits_sorted = sorted(hits)
        per_shard   = max(1, len(hits_sorted) // n)
        shards: list[tuple[int, int]] = []
        prev = text_start
        for i in range(1, n):
            idx = min(i * per_shard, len(hits_sorted) - 1)
            boundary = hits_sorted[idx]
            # Align to 16 bytes
            boundary = boundary & ~0xF
            if boundary <= prev:
                boundary = prev + 0x10
            shards.append((prev, boundary))
            prev = boundary
        shards.append((prev, text_end))
        return shards

    except Exception as e:
        print(f"[parallel_analyze] density scan failed ({e}), using equal split", flush=True)
        shard_size = (text_end - text_start + n - 1) // n
        shards = []
        addr = text_start
        while addr < text_end:
            shards.append((addr, min(addr + shard_size, text_end)))
            addr += shard_size
        return shards


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    binary  = args.binary
    n       = args.workers
    out     = args.out

    # Print accel config (import is best-effort — may not have torch)
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from ida_gpu_accel import status as accel_status
        print(f"[parallel_analyze] accel:\n{accel_status()}")
    except Exception as _e:
        print(f"[parallel_analyze] ida_gpu_accel not available: {_e}")

    print(f"[parallel_analyze] binary: {binary}")
    print(f"[parallel_analyze] workers: {n}")

    # Step 1: Discover .text range
    print("[parallel_analyze] discovering code segment...")
    text_start, text_end = discover_text_range(binary)
    text_size = text_end - text_start
    print(f"[parallel_analyze] .text: {text_start:#x} - {text_end:#x}  ({text_size // 1024 // 1024}MB)")

    # Clean up discovery-pass sidecar files before workers launch
    binary_stem = Path(binary).with_suffix("")
    for ext in (".id0", ".id1", ".nam", ".til", ".id2"):
        Path(str(binary_stem) + ext).unlink(missing_ok=True)

    # Step 2: GPU prologue scan → density-balanced shards
    print("[parallel_analyze] scanning function density for balanced shards...", flush=True)
    shards = _density_shards(binary, text_start, text_end, n)
    sizes  = [e - s for s, e in shards]
    print(f"[parallel_analyze] {len(shards)} shards, "
          f"min {min(sizes)//1024}KB max {max(sizes)//1024}KB "
          f"(density-balanced)", flush=True)

    # Step 3: Run shards in parallel
    # Each worker gets its OWN copy of the binary in a separate temp dir so
    # IDA's sidecar files (.id0/.id1/.nam/.til) don't collide.
    import shutil
    binary_name = Path(binary).name
    tmpdir = Path(tempfile.mkdtemp(prefix="parallel_ida_"))
    result_paths: list[str] = []
    worker_binaries: list[str] = []
    print(f"[parallel_analyze] writing {len(shards)} shard binaries (zeroing out-of-shard sections)...")
    for i, (s_start, s_end) in enumerate(shards):
        wdir = tmpdir / f"worker_{i:02d}"
        wdir.mkdir()
        dst = wdir / binary_name
        make_shard_binary(binary, str(dst), s_start, s_end)
        worker_binaries.append(str(dst))

    futures = []
    t_wall = time.time()
    total_shards = len(shards)
    shard_status = {}   # sid -> {"done": bool, "funcs": int, "wall": float, "error": str|None}
    for i in range(total_shards):
        shard_status[i] = {"done": False, "funcs": 0, "wall": 0.0, "error": None}

    def _render_progress(done: int, total: int, t_start: float, total_funcs: int):
        elapsed = time.time() - t_start
        pct     = done / total if total else 0
        eta_s   = (elapsed / pct - elapsed) if pct > 0 else 0
        bar_w   = 30
        filled  = int(bar_w * pct)
        bar     = "#" * filled + "-" * (bar_w - filled)
        eta_str = f"{int(eta_s//60)}m{int(eta_s%60):02d}s" if eta_s > 60 else f"{eta_s:.0f}s"
        rate    = total_funcs / elapsed if elapsed > 0 else 0
        line = (f"\r  [{bar}] {done}/{total} shards  "
                f"{pct*100:.0f}%  elapsed {elapsed:.0f}s  "
                f"ETA {eta_str}  {total_funcs} funcs  ({rate:.0f} funcs/s)  ")
        print(line, end="", flush=True)

    print(f"[parallel_analyze] launching {total_shards} workers...", flush=True)

    with ThreadPoolExecutor(max_workers=n) as pool:
        for i, (s_start, s_end) in enumerate(shards):
            rpath = str(tmpdir / f"shard_{i:02d}.json")
            result_paths.append(rpath)
            fut = pool.submit(run_shard, worker_binaries[i], s_start, s_end, rpath)
            fut._shard_id = i
            futures.append(fut)

        total_funcs = 0
        done_count  = 0
        _render_progress(0, total_shards, t_wall, 0)

        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as exc:
                r = {"error": str(exc)}
            sid = getattr(fut, "_shard_id", "?")
            done_count += 1

            if "error" in r:
                shard_status[sid]["error"] = r["error"][:80]
            else:
                nf = len(r.get("funcs", []))
                total_funcs += nf
                shard_status[sid]["done"]  = True
                shard_status[sid]["funcs"] = nf
                shard_status[sid]["wall"]  = r.get("wall_s", 0)

            _render_progress(done_count, total_shards, t_wall, total_funcs)

    print()  # newline after progress bar

    # Print per-shard summary
    for sid, s in shard_status.items():
        if s["error"]:
            print(f"  shard {sid}: ERROR {s['error']}", flush=True)
        else:
            print(f"  shard {sid}: {s['funcs']} funcs in {s['wall']:.1f}s", flush=True)

    parallel_wall = time.time() - t_wall
    print(f"\n[parallel_analyze] parallel phase done: {total_funcs} funcs in {parallel_wall:.1f}s wall")

    # Step 4: Merge — clean up any stale legacy IDA database files first so
    # IDA creates a fresh packed .i64 (not reuse old .id0/.id1 format).
    binary_stem_str = str(Path(binary).with_suffix(""))
    for _ext in (".id0", ".id1", ".id2", ".nam", ".til"):
        Path(binary_stem_str + _ext).unlink(missing_ok=True)
        Path(binary + _ext).unlink(missing_ok=True)
    print("[parallel_analyze] merging shards...")
    t_merge = time.time()
    merge_shards(binary, result_paths, out)
    print(f"[parallel_analyze] merge done in {time.time() - t_merge:.1f}s")

    total = time.time() - t_wall
    speedup_est = (total_funcs / max(total_funcs / n, 1)) / total if total > 0 else 1
    print(f"\n[parallel_analyze] total wall: {total:.1f}s for {total_funcs} funcs across {n} workers")

    # Cleanup temp files + worker binary copies + sidecar locks
    for p in result_paths:
        Path(p).unlink(missing_ok=True)
    shutil.rmtree(tmpdir, ignore_errors=True)
    # Clean up discovery-pass sidecar files left next to the original binary
    binary_stem = Path(binary).with_suffix("")
    for ext in (".id0", ".id1", ".nam", ".til", ".id2"):
        Path(str(binary_stem) + ext).unlink(missing_ok=True)


if __name__ == "__main__":
    main()

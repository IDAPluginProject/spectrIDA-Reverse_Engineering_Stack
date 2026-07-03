"""Seed acquisition for fuzzing — the inputs that start the fuzzer inside a parser.

Priority (highest value first):
  1. seeds_dir   — files the AGENT fetched/generated for this format.
  2. carve       — assets embedded IN the target binary itself (games/apps bundle
                   real levels/fonts/images) — the "user needs zero downloads" path.
  3. (proposer)  — atlas's generic input generator, handled in fuzz.hunt as fallback.

Carving scans the binary for known magic signatures and extracts self-delimiting
blobs. It's best-effort: some formats have no length in the header, so we cap each
carved blob at a sane size. Good enough to seed a fuzzer, not a perfect extractor.
"""
from __future__ import annotations

from pathlib import Path

# (name, magic bytes, max carve length). Formats whose parsers commonly hold bugs.
_SIGS = [
    ("png",  b"\x89PNG\r\n\x1a\n", 1 << 20),
    ("jpg",  b"\xff\xd8\xff",       1 << 20),
    ("gif",  b"GIF89a",             1 << 19),
    ("gif87", b"GIF87a",            1 << 19),
    ("riff", b"RIFF",               1 << 20),   # wav/avi/webp
    ("ogg",  b"OggS",               1 << 20),
    ("zip",  b"PK\x03\x04",         1 << 20),
    ("gzip", b"\x1f\x8b\x08",       1 << 20),
    ("bzip2", b"BZh",               1 << 20),
    ("ttf",  b"\x00\x01\x00\x00",   1 << 19),
    ("otf",  b"OTTO",               1 << 19),
    ("woff", b"wOFF",               1 << 19),
    ("bmp",  b"BM",                 1 << 19),
    ("elf",  b"\x7fELF",            1 << 20),
    ("wasm", b"\x00asm",            1 << 20),
    ("pdf",  b"%PDF-",              1 << 20),
    ("xml",  b"<?xml",              1 << 18),
]


def load_seed_dir(seeds_dir: str | None, max_seeds: int = 64) -> list[bytes]:
    """Read seed inputs from a folder the agent populated. Empty if none."""
    if not seeds_dir:
        return []
    d = Path(seeds_dir)
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.iterdir()):
        if f.is_file() and 0 < f.stat().st_size <= (1 << 20):
            try:
                out.append(f.read_bytes())
            except Exception:
                pass
        if len(out) >= max_seeds:
            break
    return out


def _trim_to_end(name: str, blob: bytes) -> bytes:
    """Trim a carved blob to its real end marker when the format has one, so
    seeds are clean assets instead of cap-sized garbage-padded blobs."""
    if name in ("png",) and (i := blob.find(b"IEND")) >= 0:
        return blob[:i + 8]                       # IEND + 4-byte CRC
    if name in ("jpg",) and (i := blob.find(b"\xff\xd9", 2)) >= 0:
        return blob[:i + 2]                       # EOI marker
    if name in ("gif", "gif87") and (i := blob.find(b"\x00\x3b")) >= 0:
        return blob[:i + 2]                       # GIF trailer
    return blob


def carve_seeds(binary_path: str, max_seeds: int = 24,
                min_len: int = 32) -> list[bytes]:
    """Extract embedded assets from the target by magic-byte scan → fuzz seeds.

    The target's own bundled data is the best seed material for its parsers (a
    game's fonts/levels, an app's images). Best-effort and size-capped."""
    p = Path(binary_path)
    if not p.exists() or p.stat().st_size > (400 << 20):   # skip absurdly large
        return []
    data = p.read_bytes()
    seeds: list[bytes] = []
    seen: set[bytes] = set()
    for name, magic, cap in _SIGS:
        start = 0
        while len(seeds) < max_seeds:
            i = data.find(magic, start)
            if i < 0:
                break
            blob = _trim_to_end(name, data[i:i + cap])
            # de-dup on a prefix so we don't collect thousands of near-identical hits
            keyb = blob[:64]
            if len(blob) >= min_len and keyb not in seen:
                seen.add(keyb)
                seeds.append(blob)
            start = i + max(len(magic), 1)
        if len(seeds) >= max_seeds:
            break
    return seeds


def resolve_seeds(binary_path: str, seeds_dir: str | None,
                  carve: bool = True, max_seeds: int = 64) -> tuple[list[bytes], str]:
    """Full resolution: agent seeds first, else carved-from-binary. Returns
    (seeds, source) so callers can report where seeds came from."""
    agent = load_seed_dir(seeds_dir, max_seeds)
    if agent:
        return agent, "seeds_dir"
    if carve:
        carved = carve_seeds(binary_path)
        if carved:
            return carved, "carved-from-binary"
    return [], "none (proposer fallback)"

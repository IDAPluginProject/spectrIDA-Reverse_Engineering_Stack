"""arm32_scanner.py + capstone_scanner._scan_shard_arm32: real ARM32/Thumb-2
support added in 0.2.6 (previously these arches were detected correctly but
skipped entirely -- see CHANGELOG 0.2.5). Pure Capstone, no IDA/idalib
needed."""
from __future__ import annotations

import pytest

capstone = pytest.importorskip("capstone")

from spectrida.analysis.ida_gpu_accel.arm32_scanner import (  # noqa: E402
    is_thumb,
    scan,
    strip_mode,
    thumb_addr,
)
from spectrida.analysis.ida_gpu_accel.capstone_scanner import scan_shard  # noqa: E402


def test_mode_tagging_roundtrips():
    assert is_thumb(thumb_addr(0x1000))
    assert not is_thumb(0x1000)
    assert strip_mode(thumb_addr(0x1234)) == 0x1234
    assert strip_mode(0x1234) == 0x1234


def test_scan_finds_thumb_push_lr_prologue():
    # push {r4, lr}  (16-bit Thumb)
    code = bytes([0x10, 0xB5])
    prologues, _, _, _ = scan(code, 0x1000)
    assert prologues == [thumb_addr(0x1000)]


def test_scan_finds_thumb2_push_w_prologue():
    # push.w {r4-r11, lr}  (32-bit Thumb-2)
    code = bytes([0x2D, 0xE9, 0xF0, 0x4F])
    prologues, _, _, _ = scan(code, 0x2000)
    assert prologues == [thumb_addr(0x2000)]


def test_scan_ignores_push_without_lr():
    # push {r4}  -- mid-function register spill, not a function start
    code = bytes([0x10, 0xB4])
    prologues, _, _, _ = scan(code, 0x3000)
    assert prologues == []


def test_scan_shard_arm32_finds_two_independent_functions():
    func = bytes([0x10, 0xB5, 0x00, 0xBF, 0x10, 0xBD])  # push,nop,pop{r4,pc}
    pad = bytes([0x00, 0xBF]) * 5
    code = func + pad + func
    base = 0x4000

    prologues, _, _, _ = scan(code, base)
    assert len(prologues) == 2

    result = scan_shard(code, base, base, base + len(code), arch="arm32", entry_points=prologues)
    eas = sorted(strip_mode(f.ea) for f in result.funcs)
    assert eas == [base, base + len(func) + len(pad)]
    assert all(is_thumb(f.ea) for f in result.funcs)


def test_scan_shard_arm32_self_seeds_from_entry_points():
    # No GPU/CPU pre-scan path exercised here (entry_points=None) -- confirms
    # scan_shard() wires its own arm32_scanner.scan() call when not given a
    # precomputed seed list (the path shard_worker.py uses with no
    # entries_path global prescan).
    func = bytes([0x10, 0xB5, 0x00, 0xBF, 0x10, 0xBD])
    base = 0x5000
    result = scan_shard(func, base, base, base + len(func), arch="arm32")
    assert len(result.funcs) == 1
    assert strip_mode(result.funcs[0].ea) == base

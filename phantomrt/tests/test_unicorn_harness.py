"""The hard edge: emulate a single function's machine code with Unicorn and catch
input-dependent crashes — proven on BOTH arm64 (Switch) and x86-64 (Windows)."""
import pytest

ks = pytest.importorskip("keystone")
from atlas.analysis.unicorn_harness import EmulationHarness, INPUT_BASE


def _asm(arch_mode, code: str) -> bytes:
    enc, _ = ks.Ks(*arch_mode).asm(code)
    return bytes(enc)


ARM64 = (ks.KS_ARCH_ARM64, ks.KS_MODE_LITTLE_ENDIAN)
X64 = (ks.KS_ARCH_X86, ks.KS_MODE_64)

# straight-line "deref a pointer taken from the input" — crashes iff the input's
# first 8 bytes are an unmapped address. No branches → no keystone labels needed.
ARM64_DEREF = _asm(ARM64, "ldr x1, [x0]; ldr x2, [x1]; ret")
ARM64_BENIGN = _asm(ARM64, "add x0, x0, x1; ret")
X64_DEREF = _asm(X64, "mov rax, [rdi]; mov rax, [rax]; ret")
X64_BENIGN = _asm(X64, "add rax, rsi; ret")

_GOOD_PTR = INPUT_BASE.to_bytes(8, "little")     # points back into mapped input
_BAD_PTR = (0xDEAD_BEEF_0000).to_bytes(8, "little")  # unmapped → fault


def test_arm64_benign_returns_cleanly():
    r = EmulationHarness("arm64").run(ARM64_BENIGN, b"whatever")
    assert r.returned and not r.crashed
    assert r.blocks >= 1


def test_arm64_crash_is_input_dependent():
    h = EmulationHarness("arm64")
    good = h.run(ARM64_DEREF, _GOOD_PTR)      # valid pointer in input → no crash
    bad = h.run(ARM64_DEREF, _BAD_PTR)        # bad pointer in input  → crash
    assert not good.crashed and good.returned
    assert bad.crashed and bad.crash_kind == "read_unmapped"
    assert bad.fault_addr == 0xDEAD_BEEF_0000


def test_x86_64_windows_arch_also_works():
    h = EmulationHarness("x86_64")
    assert h.run(X64_BENIGN, b"x").returned
    good = h.run(X64_DEREF, _GOOD_PTR)
    bad = h.run(X64_DEREF, _BAD_PTR)
    assert not good.crashed
    assert bad.crashed and bad.crash_kind == "read_unmapped"


def test_result_shim_exit_codes():
    bad = EmulationHarness("arm64").run(ARM64_DEREF, _BAD_PTR)
    assert bad.exit_code == 139       # duck-types as a crash for the fuzz/annotate reuse
    good = EmulationHarness("arm64").run(ARM64_BENIGN, b"")
    assert good.exit_code == 0

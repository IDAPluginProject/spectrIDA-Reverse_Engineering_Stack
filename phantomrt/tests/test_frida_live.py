"""Atlas Live (Frida) backend — guarded integration test.

Skips unless: `frida` is importable AND the prebuilt native test target exists.
The target has a global-gated vulnerable path:
    int check(char* s){ if(g_secret!=0x1337) return -1;   // emulation stops here
                        if(s[0]=='A') *(int*)0xdeadbeef;   // crash only when live
                        return strlen(s); }
So live mode proves what emulation can't: state past the gate, and a real crash.
"""
import os
import pytest

frida = pytest.importorskip("frida")

TARGET = os.path.join(os.path.dirname(__file__), "..", "experiments", "live", "target.exe")
ONCE_TARGET = os.path.join(os.path.dirname(__file__), "..", "experiments", "live", "once.exe")
CHECK_RVA = 0x1760

pytestmark = [
    pytest.mark.live,   # spawns real processes; excluded from default run
    pytest.mark.skipif(not os.path.exists(TARGET), reason="native test target.exe not built"),
]


@pytest.fixture(autouse=True)
def _reap_zombies():
    """A crash test can leave a hung target.exe / WerFault child (Windows Error
    Reporting) that stalls the next process spawn. Reap them after each test so
    these live tests don't destabilize the rest of the suite."""
    yield
    import subprocess
    subprocess.run(
        ["taskkill", "/F", "/IM", "target.exe", "/IM", "WerFault.exe"],
        capture_output=True,
    )


def _target():
    from atlas.analysis.frida_live import FridaLiveTarget
    return FridaLiveTarget(os.path.abspath(TARGET), log=lambda *a: None)


@pytest.mark.skipif(not os.path.exists(ONCE_TARGET),
                    reason="once.exe (single-call target) not built")
def test_trace_catches_once_at_startup_call():
    """Regression: the function is called exactly ONCE, early, then the process
    idles. If hooks go in AFTER resume (the old bug), the call is already gone and
    trace() returns 0. Hooks must be installed BEFORE resume."""
    from atlas.analysis.frida_live import FridaLiveTarget
    t = FridaLiveTarget(os.path.abspath(ONCE_TARGET), log=lambda *a: None)
    try:
        t.spawn(rvas=[CHECK_RVA])            # hooks BEFORE resume
        traces = t.trace([CHECK_RVA], seconds=2.0)
        assert len(traces) >= 1              # the single startup call was caught
        assert traces[0].ret == len("hello_once")
    finally:
        t.close()
        import subprocess
        subprocess.run(["taskkill", "/F", "/IM", "once.exe"], capture_output=True)


def test_rva_from_graph_addr():
    from atlas.analysis.frida_live import rva_from_graph_addr
    assert rva_from_graph_addr(0x140001760, 0x140000000) == 0x1760


def test_live_call_reaches_past_gate():
    """check('hello') returns 5 LIVE (g_secret is set) — emulation returns -1."""
    t = _target()
    try:
        t.spawn()
        r = t.fuzz_function(CHECK_RVA, ["hello", "worldXY"], arg_mode="str")
        assert r.crashes == []                 # benign inputs don't crash
        assert 5 in r.returns                  # strlen("hello") — past the g_secret gate
    finally:
        t.close()


def test_live_fuzz_detects_crash_and_reproduces_input():
    """The 'A' input triggers the bad deref; detection + reproducing input are the
    load-bearing artifacts (fault address is best-effort on Windows)."""
    t = _target()
    try:
        t.spawn()
        r = t.fuzz_function(CHECK_RVA, ["ok1", "A_crash", "ok2"], arg_mode="str")
        assert r.crashed
        assert r.respawns >= 1                              # respawned and continued
        crash_inputs = [bytes.fromhex(c["input"]) for c in r.crashes]
        assert any(ci.startswith(b"A") for ci in crash_inputs)   # reproduces the input
    finally:
        t.close()


def test_live_trace_captures_real_execution():
    """Passive trace sees the program's own check() calls running for real."""
    t = _target()
    try:
        t.spawn()
        traces = t.trace([CHECK_RVA], seconds=1.2)
        assert len(traces) >= 1                 # the target's loop calls check() itself
        assert all(tr.rva == CHECK_RVA for tr in traces)
    finally:
        t.close()

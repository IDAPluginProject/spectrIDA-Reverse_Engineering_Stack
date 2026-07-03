"""
Atlas Live — instrument the RUNNING process (Frida), not a vacuum.

Emulation (unicorn_harness) can't build the live state a real function needs, so
state-entangled functions come back `needs_state`. Live mode sidesteps that: the
real process already HAS the globals/heap/objects, so those functions actually run.

Two capabilities:
  * trace(rvas)         — passively hook functions, capture REAL args/returns/coverage
                          as the program runs its own workload.
  * fuzz_function(rva)  — call a function in-process with mutated inputs using the
                          live process state, and catch crashes.

Function identity is by RVA (module.base + offset) — the exact addressing the
spectrIDA graph stores (graph addr - image_base = rva), so a graph function maps
straight onto the live module with no guessing.

Crash model (learned the hard way): recovering in-process after a crash corrupts
the target and hangs. So fuzzing is spawn-per-crash — run inputs fast while the
process survives; when one crashes, record the REPRODUCING INPUT and respawn to
continue. Robust, and only pays the respawn cost when a crash actually happens.

Fault-address honesty: on Windows, reliably capturing the exact fault address from
inside a hardware exception is unreliable (WER interaction + the JS runtime being
in a bad state mid-fault). We tried four ways; all reliably DETECT the crash but
the address often comes back "detached". That's acceptable — the reproducing input
is the real artifact (re-run it under a debugger for the address), same as AFL.
When Frida does surface the address it's recorded; otherwise the input still is.

`frida` is an optional dependency; importing this module without it is fine, you
just can't construct FridaLiveTarget.

Honest limits: the target must actually run on this machine (so NOT Switch NSO);
for packed runtimes (Bun) the native functions are the runtime, not the app logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# The Frida JS agent. Generic + rpc-driven so one agent serves trace and fuzz.
_AGENT = r"""
// On Windows an unhandled access violation triggers Windows Error Reporting,
// which HANGS the crashing process (a WerFault child spawns and blocks). That
// both stalls fuzzing and hides the fault address. Suppress WER, then in the
// exception handler send the fault address, give the async message a beat to
// flush, and die cleanly. We do NOT try to recover in-process (redirecting past
// a fault lands on a corrupt stack → second fault); death + respawn is robust.
try {
    const k32 = 'kernel32.dll';
    const SetErrorMode = new NativeFunction(
        Module.getExportByName(k32, 'SetErrorMode'), 'uint32', ['uint32']);
    SetErrorMode(0x0001 | 0x0002 | 0x8000);   // FAILCRITICALERRORS|NOGPFAULTERRORBOX|NOOPENFILEERRORBOX
    const Sleep = new NativeFunction(
        Module.getExportByName(k32, 'Sleep'), 'void', ['uint32']);
    Process.setExceptionHandler(function (details) {
        send({ t: 'crash', addr: details.address.toString(), kind: details.type });
        Sleep(150);            // let the async crash message flush before we die
        return false;          // WER suppressed → clean fast death, no hang
    });
} catch (e) {
    // non-Windows or missing export: fall back to OS-level detach crash reporting
    Process.setExceptionHandler(function (details) {
        send({ t: 'crash', addr: details.address.toString(), kind: details.type });
        return false;
    });
}

function fnAddr(rva) { return Process.mainModule.base.add(ptr(rva)); }

rpc.exports = {
    base: function () { return Process.mainModule.base.toString(); },

    hook: function (rva) {
        Interceptor.attach(fnAddr(rva), {
            onEnter: function (args) {
                this.p = args[0];
                try { this.s = args[0].readUtf8String(80); } catch (e) { this.s = null; }
            },
            onLeave: function (ret) {
                send({ t: 'call', rva: rva, arg: this.s, ret: ret.toInt32() });
            }
        });
        return true;
    },

    // call fn(char* input) -> int, with live process state present
    callStr: function (rva, s) {
        const fn = new NativeFunction(fnAddr(rva), 'int', ['pointer']);
        return fn(Memory.allocUtf8String(s));
    },

    // call fn(char* buf, size_t len) -> int
    callBuf: function (rva, hex) {
        const bytes = [];
        for (let i = 0; i < hex.length; i += 2) bytes.push(parseInt(hex.substr(i, 2), 16));
        const buf = Memory.alloc(Math.max(bytes.length, 1));
        buf.writeByteArray(bytes);
        const fn = new NativeFunction(fnAddr(rva), 'int', ['pointer', 'uint']);
        return fn(buf, bytes.length);
    }
};
"""


@dataclass
class LiveTrace:
    rva: int
    arg: str | None
    ret: int


@dataclass
class LiveResult:
    """Outcome of live-fuzzing one function."""
    rva: int
    calls: int = 0
    crashes: list = field(default_factory=list)   # [{input, addr, kind}]
    returns: list = field(default_factory=list)    # observed non-crash return values
    respawns: int = 0

    @property
    def crashed(self) -> bool:
        return bool(self.crashes)

    def summary(self) -> dict:
        return {"rva": hex(self.rva), "calls": self.calls,
                "unique_crashes": len(self.crashes), "respawns": self.respawns,
                "crash_inputs": [c["input"] for c in self.crashes]}


class FridaLiveTarget:
    """Attach to / spawn a process and drive it via the Frida agent."""

    def __init__(self, program: str | list[str], log=print):
        import frida  # optional dep — only needed to actually use live mode
        self._frida = frida
        self.program = program if isinstance(program, list) else [program]
        self.log = log
        self.device = frida.get_local_device()
        self.pid = None
        self.session = None
        self.script = None
        self._events: list = []
        self._dead = False
        self._last_crash = None      # set by _on_message the instant a crash arrives

    # ── lifecycle ────────────────────────────────────────────────────────────
    def _on_message(self, message, data):
        if message.get("type") == "send":
            payload = message["payload"]
            self._events.append(payload)
            if payload.get("t") == "crash":
                self._last_crash = payload   # fault address, delivered while alive
        elif message.get("type") == "error":
            self.log(f"[live] agent error: {message.get('description')}")

    def _on_detached(self, reason=None, crash=None, *a):
        self._dead = True
        # Frida hands us an OS-level Crash object when the process actually faulted
        # (reliable fault address, unlike an in-agent handler racing teardown).
        if crash is not None:
            addr = getattr(crash, "address", None)
            self._last_crash = {
                "addr": hex(addr) if isinstance(addr, int) else str(addr),
                "kind": getattr(crash, "signal_name", None) or reason or "crash",
            }

    def spawn(self) -> "FridaLiveTarget":
        self._dead = False
        self.pid = self.device.spawn(self.program)
        self.session = self.device.attach(self.pid)
        self.session.on("detached", self._on_detached)
        self.script = self.session.create_script(_AGENT)
        self.script.on("message", self._on_message)
        self.script.load()
        self.device.resume(self.pid)
        return self

    def close(self):
        try:
            if self.pid is not None:
                self.device.kill(self.pid)
        except Exception:
            pass
        self.pid = self.session = self.script = None

    # ── passive tracing: what really runs, with real args ────────────────────
    def trace(self, rvas: list[int], seconds: float = 2.0) -> list[LiveTrace]:
        if self.script is None:
            self.spawn()
        for rva in rvas:
            try:
                self.script.exports_sync.hook(hex(rva))
            except Exception as e:
                self.log(f"[live] hook {hex(rva)} failed: {e}")
        time.sleep(seconds)
        out = []
        for e in self._events:
            if e.get("t") == "call":
                out.append(LiveTrace(int(e["rva"], 16), e.get("arg"), e.get("ret")))
        return out

    # ── in-process fuzzing with spawn-per-crash ──────────────────────────────
    def fuzz_function(self, rva: int, inputs, arg_mode: str = "str",
                      call_timeout: float = 3.0) -> LiveResult:
        """Call the function at `rva` with each input, using live process state.
        On a crash the process dies and is respawned to continue. `arg_mode`:
        "str" → fn(char*); "buf" → fn(char*, len)."""
        res = LiveResult(rva=rva)
        if self.script is None:
            self.spawn()
        seen_crash = set()

        for inp in inputs:
            if self._dead or self.script is None:
                self._respawn(res)
            payload = inp if isinstance(inp, (bytes, bytearray)) else str(inp).encode()
            self._last_crash = None
            ret, detached = None, False
            try:
                if arg_mode == "buf":
                    ret = self.script.exports_sync.call_buf(hex(rva), payload.hex())
                else:
                    ret = self.script.exports_sync.call_str(hex(rva), payload.decode("latin-1"))
            except Exception:
                detached = True    # session dropped mid-call = the process faulted
                time.sleep(0.05)   # let the 'detached' crash-report callback land

            # a crash is an outright detach; its fault address comes from Frida's
            # OS-level Crash object captured in _on_detached.
            crash = self._last_crash
            if crash or detached:
                key = crash["addr"] if crash else "detached"
                if key not in seen_crash:
                    seen_crash.add(key)
                    res.crashes.append({
                        "input": payload.hex(), "addr": key,
                        "kind": crash["kind"] if crash else "detached",
                    })
                    self.log(f"[live] CRASH @ {key} on {payload[:32]!r} "
                             f"({len(res.crashes)} unique)")
                self._respawn(res)   # post-fault state is corrupt → fresh process
            else:
                res.calls += 1
                if len(res.returns) < 50:
                    res.returns.append(ret)
        return res

    def _respawn(self, res: LiveResult):
        self.close()
        res.respawns += 1
        self._events = []
        self.spawn()


def rva_from_graph_addr(graph_addr: int, image_base: int) -> int:
    """spectrIDA graph stores absolute VAs; a live module is relocated (ASLR).
    The stable identity is the RVA = addr - image_base."""
    return graph_addr - image_base

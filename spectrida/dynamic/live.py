"""Live instrumentation of a RUNNING target (Frida) — for functions that need
real process state (globals/heap/objects) that emulation can't build, so they'd
come back `needs_state`. Only for targets that actually run on this machine
(native PE/ELF) — NOT Switch NSO / other non-runnable images.

Wraps atlas's FridaLiveTarget (spawn-per-crash, WER-aware). Graph addresses are
absolute VAs; a live module is relocated, so we hook by RVA = addr - image_base.
"""
from __future__ import annotations

from .emulate import resolve_binary_path


def _image_base(binary_path: str) -> int:
    """Link-time image base, so RVA = graph_addr - image_base. Reuse atlas's
    format detection (PE/ELF header) via the bridge."""
    from atlas.analysis.spectrida_bridge import EmulatedBinary
    eb = EmulatedBinary(binary_path)
    return eb.image.image_base


def live_trace(graph, tag: str, addresses: list[int], binary_path: str | None = None,
               seconds: float = 3.0, spawn_args: list[str] | None = None) -> dict:
    """Spawn the target, hook the given functions, and capture what really runs
    (real args + returns + call counts). Returns per-function live facts; the
    caller annotates the graph. Does not fuzz — pure observation."""
    from atlas.analysis.frida_live import FridaLiveTarget, rva_from_graph_addr

    path = resolve_binary_path(graph, tag, binary_path)
    base = _image_base(path)
    program = [path, *(spawn_args or [])]

    rvas = [rva_from_graph_addr(a, base) for a in addresses]
    rva_to_addr = {rva_from_graph_addr(a, base): a for a in addresses}

    target = FridaLiveTarget(program)
    try:
        target.spawn()
        traces = target.trace(rvas, seconds=seconds)
    finally:
        target.close()

    per_fn: dict[int, dict] = {}
    for tr in traces:
        addr = rva_to_addr.get(tr.rva, base + tr.rva)
        d = per_fn.setdefault(addr, {"calls": 0, "sample_args": [], "returns": []})
        d["calls"] += 1
        if tr.arg is not None and len(d["sample_args"]) < 5:
            d["sample_args"].append(tr.arg)
        if len(d["returns"]) < 5:
            d["returns"].append(tr.ret)

    return {
        "binary": tag, "binary_path": path, "image_base": hex(base),
        "hooked": [hex(a) for a in addresses],
        "observed": {hex(a): v for a, v in per_fn.items()},
        "total_calls": sum(v["calls"] for v in per_fn.values()),
    }

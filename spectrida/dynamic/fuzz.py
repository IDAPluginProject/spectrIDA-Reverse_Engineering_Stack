"""Coverage-ish crash hunt on a single function via emulation.

Given a function the static side found, feed it many mutated inputs (starting
from agent-provided seeds when available) and collect wild-address faults =
candidate crashes with reproducing inputs. Emulation-based, so it works on
non-runnable targets too (Switch/Android); for state-entangled functions it will
mostly report needs_state (use live_trace there instead).

Seed resolution (the agent↔Atlas seam):
  1. seeds_dir  — files the AGENT fetched/generated for this format (highest value)
  2. proposer   — atlas's built-in input generator (fallback)
(carve-from-binary is a documented future source.)
"""
from __future__ import annotations

from .emulate import resolve_binary_path
from .seeds import resolve_seeds


def hunt(graph, tag: str, addr: int, binary_path: str | None = None,
         seeds_dir: str | None = None, rounds: int = 400, max_insns: int = 8000,
         on_progress=None) -> dict:
    """Emulate-fuzz the function at ``addr``; return crashes + a verdict."""
    import random
    from atlas.analysis.spectrida_bridge import EmulatedBinary
    from atlas.analysis.binary_fuzz import InputProposer, parse_token
    from collections import Counter

    path = resolve_binary_path(graph, tag, binary_path)
    eb = EmulatedBinary(path)
    eb.regions()
    fn = graph.get_function(tag, addr) or {}

    # agent-provided seeds first, else carve embedded assets from the binary itself
    seeds, seed_source = resolve_seeds(path, seeds_dir)
    proposer = InputProposer()
    rng = random.Random(0xC0FFEE)

    def mutate(d: bytes) -> bytes:
        if not d:
            return bytes(rng.randrange(256) for _ in range(rng.randint(1, 64)))
        d = bytearray(d)
        for _ in range(rng.randint(1, 24)):
            i = rng.randrange(len(d)); r = rng.random()
            if r < 0.6:
                d[i] = rng.randrange(256)
            elif r < 0.8 and len(d) < 4096:
                d[i:i] = bytes([rng.randrange(256)])
            elif len(d) > 4:
                del d[i]
        return bytes(d)

    statuses = Counter()
    crashes: dict[str, str] = {}   # fault-addr key -> reproducing input (hex)
    max_blocks = 0

    for it in range(1, rounds + 1):
        if seeds and rng.random() < 0.7:
            payload = mutate(rng.choice(seeds))
        else:
            _, payload = parse_token(proposer.propose(1)[0])
            payload = mutate(payload)
        res = eb.emulate(addr, payload, max_insns=max_insns)
        statuses[res.status] += 1
        max_blocks = max(max_blocks, res.blocks)
        if res.status == "crash":
            key = res.crash_kind + "@" + hex(res.fault_addr)
            if key not in crashes:
                crashes[key] = payload.hex()
        if on_progress and it % 50 == 0:
            on_progress(it, rounds, len(crashes))

    verdict = ("candidate_crash" if crashes else
               "needs_state" if statuses.get("needs_state") else
               "exercised_clean" if statuses.get("clean") else "inconclusive")

    return {
        "binary": tag, "address": hex(addr), "name": fn.get("name"),
        "arch": eb.arch, "verdict": verdict,
        "reachable": max_blocks > 0, "blocks": max_blocks,
        "rounds": rounds, "seeds_used": len(seeds), "seed_source": seed_source,
        "unique_crashes": len(crashes),
        "crash_inputs": crashes,           # {kind@addr: input_hex}
        "status_counts": dict(statuses),
        "binary_path": path,
    }

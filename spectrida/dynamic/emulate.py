"""One-shot emulation of a single function → an honest runtime verdict.

Wraps atlas's EmulatedBinary (Unicorn, chain-emulation + call stubbing, any arch).
Given a function the static side already found, this runs it in isolation and
reports what happens: clean / needs_state / candidate_crash, how far it got, and
how many out-of-chain calls were stubbed — then writes that onto the graph node.

Emulation needs the ORIGINAL binary bytes (the graph only stores the .i64 path),
so ``resolve_binary_path`` finds it: explicit arg → a binary_path prop on the
Binary node → common locations (~/.spectrida, next to the .i64).
"""
from __future__ import annotations

import os
from pathlib import Path


def resolve_binary_path(graph, tag: str, explicit: str | None = None) -> str:
    """Locate the original binary for a graph binary tag.

    The graph stores i64_path (IDA database), not the original file emulation
    needs. Resolution order, first hit wins:
      1. explicit path passed by the caller/agent,
      2. a `binary_path` property on the Binary node (set at analysis time, future),
      3. ~/.spectrida/<tag> and ~/.spectrida/output/<tag>,
      4. a file named <tag> sitting next to the registered .i64.
    """
    if explicit and Path(explicit).exists():
        return explicit

    # read all Binary props (avoids a "property key does not exist" warning when
    # binary_path hasn't been set yet on older nodes)
    with graph.driver.session() as s:
        rec = s.run("MATCH (b:Binary {tag:$t}) RETURN properties(b) AS p", t=tag).single()
    props = (rec["p"] if rec else {}) or {}
    bp = props.get("binary_path")
    ip = props.get("i64_path")
    if bp and Path(bp).exists():
        return bp

    home = Path.home() / ".spectrida"
    for cand in (home / tag, home / "output" / tag):
        if cand.exists():
            return str(cand)

    if ip:
        near = Path(ip).parent / tag
        if near.exists():
            return str(near)

    raise FileNotFoundError(
        f"could not find the original binary for tag '{tag}'. Pass an explicit "
        f"binary_path, or set a binary_path property on its Binary node. "
        f"(i64_path={ip})")


def emulate_one(graph, tag: str, addr: int, binary_path: str | None = None,
                max_insns: int = 6000, rounds: int = 20,
                input_size: int = 96) -> dict:
    """Emulate the function at ``addr`` a few times with fuzzed inputs and return
    an aggregated honest verdict. Does NOT annotate — the caller decides that."""
    from atlas.analysis.spectrida_bridge import EmulatedBinary
    from atlas.analysis.binary_fuzz import InputProposer, parse_token
    from collections import Counter

    path = resolve_binary_path(graph, tag, binary_path)
    eb = EmulatedBinary(path)
    eb.regions()  # map the image once

    fn = graph.get_function(tag, addr) or {}
    proposer = InputProposer()
    statuses = Counter()
    max_blocks = stubbed = 0
    candidate = None

    for _ in range(rounds):
        _, payload = parse_token(proposer.propose(1)[0])
        if payload:
            payload = (payload * (input_size // len(payload) + 1))[:input_size]
        res = eb.emulate(addr, payload, max_insns=max_insns)
        statuses[res.status] += 1
        max_blocks = max(max_blocks, res.blocks)
        stubbed = max(stubbed, res.stubbed_calls)
        if res.status == "crash" and candidate is None:
            candidate = {"input": payload.hex(), "note": res.note}

    if candidate:
        verdict, note = "candidate_crash", candidate["note"]
    elif statuses.get("needs_state"):
        verdict = "needs_state"
        note = ("faults on uninitialized global/this — needs live engine state; "
                "try live_trace on a runnable target, or the agent reasons statically")
    elif statuses.get("clean"):
        verdict, note = "exercised_clean", f"ran to return; up to {stubbed} calls stubbed"
    else:
        verdict, note = "inconclusive", "no clean return within the instruction budget"

    return {
        "binary": tag, "address": hex(addr), "name": fn.get("name"),
        "arch": eb.arch, "format": eb.format,
        "verdict": verdict, "note": note,
        "reachable": max_blocks > 0, "blocks": max_blocks, "stubbed_calls": stubbed,
        "rounds": rounds, "status_counts": dict(statuses),
        "crash_input": candidate["input"] if candidate else None,
        "binary_path": path,
    }

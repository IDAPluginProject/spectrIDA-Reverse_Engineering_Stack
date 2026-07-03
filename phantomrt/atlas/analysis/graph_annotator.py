"""
Atlas -> spectrIDA graph: the annotation edge.

spectrIDA builds the static map (Function/Binary nodes in Neo4j). Atlas runs the
code and learns what actually happens. This writes Atlas's *runtime* findings back
onto the matching Function node as ``atlas_*`` properties, so anything reading the
graph (an LLM, the spectrIDA UI) instantly sees "this function is reachable and
crashes on long input".

Schema it targets (confirmed live):
    (:Function {name, addr, binary, size, pseudocode, disasm, id})
    (:Binary   {tag, i64_path})

Design choices:
  * Match by (binary, addr) primarily — addresses are stable; names get renamed.
  * Only ever SET on EXISTING nodes (never MERGE) — Atlas annotates spectrIDA's
    map, it does not invent functions. A miss returns 0 and is reported, not hidden.
  * All properties are namespaced ``atlas_`` so they never collide with spectrIDA's,
    and can be cleared wholesale.
  * If the graph is unreachable, findings can still be dumped to JSON as a fallback.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


_PRIM = (str, int, float, bool)


def _coerce(v):
    """Neo4j properties must be primitives or lists of primitives; JSON-encode
    anything richer (dicts, mixed lists) so nothing is silently dropped."""
    if isinstance(v, _PRIM) or v is None:
        return v
    if isinstance(v, (list, tuple)) and all(isinstance(x, _PRIM) for x in v):
        return list(v)
    return json.dumps(v, default=str)


class GraphAnnotator:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j",
                 password: Optional[str] = None, driver=None, log=print,
                 prefix: str = "atlas_"):
        self.log = log
        # property namespace: "atlas_" standalone, "dyn_" when driven from spectrIDA.
        self.prefix = prefix
        if driver is not None:
            self._driver = driver
        else:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(uri, auth=(user, password))

    @classmethod
    def from_spectrida_config(cls, path: Optional[str] = None, log=print):
        """Build from ~/.spectrida/config.toml ([graph] password)."""
        import tomllib
        p = Path(path) if path else Path.home() / ".spectrida" / "config.toml"
        cfg = tomllib.loads(p.read_text())
        pw = cfg.get("graph", {}).get("password")
        uri = cfg.get("graph", {}).get("uri", "bolt://localhost:7687")
        return cls(uri=uri, user=cfg.get("graph", {}).get("user", "neo4j"),
                   password=pw, log=log)

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── matching helper ──────────────────────────────────────────────────────
    @staticmethod
    def _match(addr, name):
        if addr is not None:
            return "MATCH (f:Function {binary:$binary, addr:$addr})", {"addr": addr}
        if name is not None:
            return "MATCH (f:Function {binary:$binary, name:$name})", {"name": name}
        raise ValueError("need addr or name to identify the function")

    # ── write ────────────────────────────────────────────────────────────────
    def annotate(self, binary: str, facts: dict, *, addr=None, name=None) -> int:
        """Stamp prefixed runtime facts onto a Function node. Returns #matched."""
        match, key = self._match(addr, name)
        p = self.prefix
        props = {(k if k.startswith(p) else f"{p}{k}"): _coerce(v)
                 for k, v in facts.items()}
        props[f"{p}updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._driver.session() as s:
            n = s.run(f"{match} SET f += $props RETURN count(f) AS n",
                      binary=binary, props=props, **key).single()["n"]
        if n == 0:
            self.log(f"[annotate] no match for binary={binary} {key} — nothing written")
        else:
            self.log(f"[annotate] wrote {len(props)} {p}* props to {n} node(s) "
                     f"(binary={binary} {key})")
        return n

    def annotate_fuzz_run(self, binary: str, fuzz_env, *, addr=None, name=None) -> int:
        """Convenience: turn a BinaryFuzzEnv's findings into function facts."""
        s = fuzz_env.summary()
        crashed = s["unique_crashes"] > 0
        kinds = sorted({k.split(":")[0] for k in s["crash_inputs"]})
        facts = {
            "reachable": True,
            "crashes": crashed,
            "crash_kinds": kinds,
            "functions_covered": s["functions_covered"],
            "unique_crashes": s["unique_crashes"],
            "sample_crash_input": next(iter(s["crash_inputs"].values()), None),
            "verdict": ("crashes on fuzzed input" if crashed
                        else "exercised, no crash found"),
        }
        return self.annotate(binary, facts, addr=addr, name=name)

    # ── read / clear ─────────────────────────────────────────────────────────
    def read(self, binary: str, *, addr=None, name=None) -> dict:
        match, key = self._match(addr, name)
        with self._driver.session() as s:
            rec = s.run(f"{match} RETURN properties(f) AS p",
                        binary=binary, **key).single()
        if not rec:
            return {}
        return {k: v for k, v in rec["p"].items() if k.startswith(self.prefix)}

    def clear(self, binary: str, *, addr=None, name=None) -> int:
        """Remove all prefixed props (leaves spectrIDA's own data untouched)."""
        match, key = self._match(addr, name)
        with self._driver.session() as s:
            rec = s.run(f"{match} RETURN [k IN keys(f) WHERE k STARTS WITH $p] "
                        f"AS ks", binary=binary, p=self.prefix, **key).single()
            if not rec or not rec["ks"]:
                return 0
            removes = ", ".join(f"f.`{k}`" for k in rec["ks"])
            n = s.run(f"{match} REMOVE {removes} RETURN count(f) AS n",
                      binary=binary, **key).single()["n"]
        return n

    # ── fallback when the graph is down ──────────────────────────────────────
    @staticmethod
    def export_json(path: str, findings: dict) -> None:
        Path(path).write_text(json.dumps(findings, indent=2, default=str))

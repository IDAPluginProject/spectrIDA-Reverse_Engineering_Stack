"""Neo4j-backed function-call graph — populated once per binary by
`scripts/populate_graph.py`, queried live by the MCP server (and anything
else that wants fast cross-function lookups without reopening a .i64).

One Function node per (binary, address). CALLS edges mirror IDA xrefs.
Multiple binaries share one Neo4j instance — nodes are namespaced by a
`binary` tag, so two binaries with overlapping virtual addresses never
collide.

Schema:
    (:Function {id, binary, addr, name, size, pseudocode, disasm})
    (:Function)-[:CALLS]->(:Function)
    (:Binary   {tag, i64_path})

`id` is "{binary}:{addr_hex}" — the only thing that needs to be globally unique.
`disasm` is JSON-encoded ([{address, text}, ...]) — Neo4j properties can't
hold a list of maps directly, so it's serialized on write and decoded back
into a real list on read. It's the byte/instruction-level layer pseudocode
can't provide: exact instruction boundaries and operands needed to plan a
patch (pseudocode is reconstructed/approximate and has no address mapping
precise enough for that).
"""
from __future__ import annotations

import json

from neo4j import GraphDatabase


def _fid(binary: str, addr: int) -> str:
    return f"{binary}:{hex(addr)}"


class FunctionGraph:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT function_id IF NOT EXISTS "
                  "FOR (f:Function) REQUIRE f.id IS UNIQUE")
            s.run("CREATE INDEX function_name IF NOT EXISTS "
                  "FOR (f:Function) ON (f.name)")
            s.run("CREATE INDEX function_binary IF NOT EXISTS "
                  "FOR (f:Function) ON (f.binary)")
            s.run("CREATE CONSTRAINT binary_tag IF NOT EXISTS "
                  "FOR (b:Binary) REQUIRE b.tag IS UNIQUE")

    def close(self) -> None:
        self.driver.close()

    # ── binary registry ────────────────────────────────────────────────────

    def register_binary(self, tag: str, i64_path: str,
                        binary_path: str | None = None) -> None:
        """Record which .i64 file backs a `binary` tag, so tools that need
        live IDA access (full pseudocode, rename, fresh analysis) know where
        to look instead of relying solely on the cached graph snapshot.

        ``binary_path`` (optional) records the ORIGINAL binary file too, so the
        dynamic-analysis tools (emulation/live) can find its real bytes exactly
        instead of guessing from ~/.spectrida heuristics."""
        with self.driver.session() as s:
            if binary_path:
                s.run("MERGE (b:Binary {tag: $tag}) "
                      "SET b.i64_path = $path, b.binary_path = $bp",
                      tag=tag, path=i64_path, bp=binary_path)
            else:
                s.run("MERGE (b:Binary {tag: $tag}) SET b.i64_path = $path",
                      tag=tag, path=i64_path)

    def set_binary_path(self, tag: str, binary_path: str) -> None:
        """Record/update the original binary path for an already-registered tag."""
        with self.driver.session() as s:
            s.run("MERGE (b:Binary {tag: $tag}) SET b.binary_path = $bp",
                  tag=tag, bp=binary_path)

    def delete_binary(self, tag: str) -> int:
        """Remove every Function node + edge for `tag`, plus its Binary
        registry entry. Returns count of Function nodes deleted."""
        with self.driver.session() as s:
            n = s.run("MATCH (f:Function {binary: $tag}) RETURN count(f) AS n",
                       tag=tag).single()["n"]
            s.run("MATCH (f:Function {binary: $tag}) DETACH DELETE f", tag=tag)
            s.run("MATCH (b:Binary {tag: $tag}) DELETE b", tag=tag)
            return n

    def get_binary_path(self, tag: str) -> str | None:
        with self.driver.session() as s:
            rec = s.run("MATCH (b:Binary {tag: $tag}) RETURN b.i64_path AS p", tag=tag).single()
            return rec["p"] if rec else None

    def list_binaries(self) -> list[dict]:
        with self.driver.session() as s:
            return [dict(r["b"]) for r in s.run("MATCH (b:Binary) RETURN b")]

    # ── writes ──────────────────────────────────────────────────────────────

    def upsert_functions(self, binary: str, rows: list[dict]) -> None:
        """rows: [{addr:int, name:str, size:int, pseudocode:str, disasm:list}, ...]
        disasm is a list of {address, text} — serialized to JSON before storage."""
        if not rows:
            return
        payload = [
            {"id": _fid(binary, r["addr"]), "addr": r["addr"], "name": r["name"],
             "size": r.get("size", 0), "pseudocode": r.get("pseudocode", ""),
             "disasm": json.dumps(r["disasm"]) if r.get("disasm") else ""}
            for r in rows
        ]
        query = """
        UNWIND $rows AS row
        MERGE (f:Function {id: row.id})
        SET f.binary = $binary, f.addr = row.addr, f.name = row.name,
            f.size = row.size, f.pseudocode = row.pseudocode, f.disasm = row.disasm
        """
        with self.driver.session() as s:
            s.run(query, rows=payload, binary=binary)

    def upsert_calls(self, binary: str, edges: list[tuple[int, int]]) -> None:
        """edges: [(caller_addr, callee_addr), ...]. Creates placeholder
        callee nodes (no name/size/pseudocode yet) if not already populated —
        filled in later when that function is itself processed."""
        if not edges:
            return
        payload = [
            {"caller_id": _fid(binary, a), "caller_addr": a,
             "callee_id": _fid(binary, b), "callee_addr": b}
            for a, b in edges
        ]
        query = """
        UNWIND $edges AS e
        MERGE (a:Function {id: e.caller_id})
        ON CREATE SET a.binary = $binary, a.addr = e.caller_addr
        MERGE (b:Function {id: e.callee_id})
        ON CREATE SET b.binary = $binary, b.addr = e.callee_addr
        MERGE (a)-[:CALLS]->(b)
        """
        with self.driver.session() as s:
            s.run(query, edges=payload, binary=binary)

    # ── reads ───────────────────────────────────────────────────────────────

    def get_function(self, binary: str, addr: int) -> dict | None:
        query = "MATCH (f:Function {id: $id}) RETURN f"
        with self.driver.session() as s:
            rec = s.run(query, id=_fid(binary, addr)).single()
            if not rec:
                return None
            fn = dict(rec["f"])
            if fn.get("disasm"):
                try:
                    fn["disasm"] = json.loads(fn["disasm"])
                except (TypeError, ValueError):
                    fn["disasm"] = []
            else:
                fn["disasm"] = []
            return fn

    def search_by_name(self, binary: str, name_substring: str, limit: int = 20) -> list[dict]:
        cypher = """
        MATCH (f:Function {binary: $binary})
        WHERE f.name CONTAINS $name_substring
        RETURN f.addr AS addr, f.name AS name, f.size AS size LIMIT $limit
        """
        with self.driver.session() as s:
            return [dict(r) for r in s.run(cypher, binary=binary, name_substring=name_substring, limit=limit)]

    def callees(self, binary: str, addr: int) -> list[dict]:
        cypher = """
        MATCH (:Function {id: $id})-[:CALLS]->(callee)
        RETURN callee.addr AS addr, callee.name AS name, callee.size AS size
        """
        with self.driver.session() as s:
            return [dict(r) for r in s.run(cypher, id=_fid(binary, addr))]

    def callers(self, binary: str, addr: int) -> list[dict]:
        cypher = """
        MATCH (caller)-[:CALLS]->(:Function {id: $id})
        RETURN caller.addr AS addr, caller.name AS name, caller.size AS size
        """
        with self.driver.session() as s:
            return [dict(r) for r in s.run(cypher, id=_fid(binary, addr))]

    def trace_chain(self, binary: str, addr: int, depth: int = 2) -> list[dict]:
        """All functions reachable within `depth` calls of addr, deduped."""
        cypher = f"""
        MATCH (start:Function {{id: $id}})-[:CALLS*1..{depth}]->(reached)
        RETURN DISTINCT reached.addr AS addr, reached.name AS name, reached.size AS size
        """
        with self.driver.session() as s:
            return [dict(r) for r in s.run(cypher, id=_fid(binary, addr))]

    def stats(self, binary: str) -> dict:
        cypher = """
        MATCH (f:Function {binary: $binary})
        RETURN count(f) AS total,
               count(CASE WHEN f.name IS NOT NULL AND NOT f.name STARTS WITH 'sub_' THEN 1 END) AS named
        """
        with self.driver.session() as s:
            rec = s.run(cypher, binary=binary).single()
            return {"total": rec["total"], "named": rec["named"]}

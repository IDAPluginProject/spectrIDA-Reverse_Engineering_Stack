#!/usr/bin/env python3
"""
populate_graph.py — walk a .i64 via spectrIDA's API and push everything
(functions + call edges) into Neo4j for the MCP server to query.

Thin CLI wrapper around spectrida.core.populate.populate_graph — see that
module for the actual demangle/skip/AI-naming logic. The same function is
also called directly by the MCP server's analyze_binary tool.

Resumable: writes in batches as it goes, not buffered to the end. Re-running
on the same binary tag just re-MERGEs (idempotent), so an interrupted run can
just be restarted with --skip N to pick up roughly where it left off.

Usage:
    python scripts/populate_graph.py --i64 "path/to/file.i64" --binary among_us \
        --neo4j-pass <password>
"""
import argparse
import asyncio
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

from spectrida.api import open_i64
from spectrida.core.graph import FunctionGraph
from spectrida.core.populate import populate_graph


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i64", required=True)
    ap.add_argument("--binary", required=True, help="tag namespacing this binary's nodes")
    ap.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-pass", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap total functions processed")
    ap.add_argument("--skip", type=int, default=0, help="skip first N functions (resume)")
    ap.add_argument("--min_size", type=int, default=0,
                     help="skip AI-naming functions smaller than this (still indexed for edges)")
    ap.add_argument("--name_chunk", type=int, default=8,
                     help="how many naming calls to fire concurrently (match llama-server --parallel)")
    ap.add_argument("--sample", choices=["sequential", "random"], default="sequential",
                     help="sequential = address order (needed for full coverage runs); "
                          "random = shuffle first (representative sample — sequential address-order "
                          "tends to front-load runtime/SDK init boilerplate, not game logic)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    graph = FunctionGraph(args.neo4j_uri, args.neo4j_user, args.neo4j_pass)
    graph.register_binary(args.binary, args.i64)

    async def on_progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"[*] {done}/{total}")

    async with open_i64(args.i64, verbose=True) as db:
        funcs = await db.list_functions()
        print(f"[*] {len(funcs)} functions in {args.i64}")
        result = await populate_graph(
            db, graph, args.binary,
            limit=args.limit, skip=args.skip, min_size=args.min_size,
            name_chunk=args.name_chunk, sample=args.sample, seed=args.seed,
            on_progress=on_progress,
        )

    print(f"[+] {result['attempted']} AI-naming attempts, {result['named']} named, "
          f"{result['demangled']} demangled")
    stats = graph.stats(args.binary)
    print(f"[+] Done. {stats['named']}/{stats['total']} named in graph for '{args.binary}'")
    graph.close()


if __name__ == "__main__":
    asyncio.run(main())

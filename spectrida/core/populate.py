"""Walk an open .i64 and push functions + call edges into Neo4j, with two
free/cheap passes before any AI call:

  1. Demangle: any _Z-prefixed C++ name gets IDA's own demangler applied
     (ground truth — the compiler's mangling already encodes the real name).
  2. Tiny-function skip (min_size): thunks/stubs get indexed for graph
     edges but never wasted on an AI naming call.

Only what's left after those two — genuinely stripped sub_* functions above
the size floor — goes to the LLM (via llama-server, concurrently chunked).

Used by both scripts/populate_graph.py (standalone CLI) and the MCP server's
analyze_binary tool (chained right after a fresh analysis pass).
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import httpx

from spectrida.config import naming_llama_url

if TYPE_CHECKING:
    from spectrida.api import IDADatabase
    from spectrida.core.graph import FunctionGraph

BATCH_SIZE = 200
PSEUDOCODE_CHARS = 3000   # snippet only — Claude can call get_full_pseudocode live for the rest

# spectrIDA's own naming prompt feeds raw assembly + a NAME:/REASON: format,
# which the validated checkpoint was never trained against (it degenerates
# into repetitive reasoning text on that input shape). Also needs the EXACT
# trained template — an empty forced <think></think> block — or it rambles.
NAMING_SYSTEM = (
    "You are an expert reverse engineer analyzing a stripped game binary.\n"
    "Given decompiled pseudocode, respond with ONLY a single proposed function name, "
    "nothing else — no explanation, no reasoning, just the name."
)


async def name_from_pseudocode(pseudocode: str, http: httpx.AsyncClient, llama_url: str) -> str:
    if not pseudocode.strip():
        return ""
    user = f"Pseudocode:\n```c\n{pseudocode[:2000]}\n```\n\nProposed function name:"
    prompt = (
        f"<|im_start|>system\n{NAMING_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    payload = {"prompt": prompt, "temperature": 0.3, "n_predict": 40}
    resp = await http.post(llama_url, json=payload)
    resp.raise_for_status()
    text = resp.json().get("content", "").strip()
    first_line = text.splitlines()[0].strip() if text else ""
    return first_line[:80]


async def _const(value):
    return value


async def populate_graph(
    db: IDADatabase,
    graph: FunctionGraph,
    binary: str,
    *,
    limit: int | None = None,
    skip: int = 0,
    min_size: int = 0,
    name_chunk: int = 8,
    sample: str = "sequential",
    seed: int = 42,
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
) -> dict:
    """Returns {"total": N, "named": N, "demangled": N, "attempted": N}."""
    funcs = await db.list_functions()

    if sample == "random":
        import random
        random.seed(seed)
        funcs = list(funcs)
        random.shuffle(funcs)

    targets = funcs[skip:]
    if limit:
        targets = targets[:limit]

    llama_url = naming_llama_url()
    func_batch: list[dict] = []
    edge_batch: list[tuple[int, int]] = []
    named_count = 0
    demangled_count = 0
    attempted_count = 0
    done = 0

    async with httpx.AsyncClient(timeout=120) as http:
        # Stage 1: IDA-side work (decompile + xrefs) is sequential — a single
        # idalib worker process can't parallelize. Stage 2: naming calls go
        # to llama-server's parallel slots concurrently, chunked so the slow
        # IDA stage and the LLM stage don't block each other more than needed.
        for chunk_start in range(0, len(targets), name_chunk):
            chunk = targets[chunk_start: chunk_start + name_chunk]

            # _Z = Itanium (GCC/Clang ELF/NSO), ? = MSVC (Windows PE) — IDA's
            # demangle_name() auto-detects which one a given binary actually
            # uses, but only if we actually hand it the mangled names.
            mangled_names = [f["name"] for f in chunk if f["name"].startswith(("_Z", "?"))]
            demangled = await db.demangle(mangled_names) if mangled_names else {}
            demangled_count += len(demangled)

            pending: list[dict] = []
            for f in chunk:
                addr = f["start"]
                name = demangled.get(f["name"], f["name"])

                pseudocode = ""
                try:
                    code = await db.decompile(addr)
                    pseudocode = (code or "")[:PSEUDOCODE_CHARS]
                except Exception:
                    pass

                edges = []
                try:
                    callees = await db.xrefs_from(addr)
                    for c in callees:
                        callee_addr = c["address"] if isinstance(c["address"], int) else int(c["address"], 16)
                        edges.append((addr, callee_addr))
                except Exception:
                    pass

                disasm = []
                try:
                    disasm = await db.disasm(addr)
                except Exception:
                    pass

                needs_naming = name.lower().startswith("sub_") and f.get("size", 0) >= min_size
                pending.append({"addr": addr, "name": name, "size": f.get("size", 0),
                                "pseudocode": pseudocode, "disasm": disasm, "edges": edges,
                                "needs_naming": needs_naming})

            attempted_count += sum(1 for p in pending if p["needs_naming"])
            naming_results = await asyncio.gather(
                *[name_from_pseudocode(p["pseudocode"], http, llama_url) if p["needs_naming"]
                  else _const("") for p in pending],
                return_exceptions=True,
            )

            for p, new_name in zip(pending, naming_results, strict=True):
                if not isinstance(new_name, Exception) and new_name:
                    p["name"] = new_name
                    named_count += 1
                func_batch.append({"addr": p["addr"], "name": p["name"],
                                   "size": p["size"], "pseudocode": p["pseudocode"],
                                   "disasm": p["disasm"]})
                edge_batch.extend(p["edges"])

            done += len(chunk)
            if len(func_batch) >= BATCH_SIZE:
                graph.upsert_functions(binary, func_batch)
                graph.upsert_calls(binary, edge_batch)
                func_batch, edge_batch = [], []

            if on_progress:
                await on_progress(done, len(targets))

    if func_batch or edge_batch:
        graph.upsert_functions(binary, func_batch)
        graph.upsert_calls(binary, edge_batch)

    return {"total": len(targets), "named": named_count,
            "demangled": demangled_count, "attempted": attempted_count}

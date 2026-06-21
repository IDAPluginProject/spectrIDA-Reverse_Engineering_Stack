"""spectrIDA MCP server — lets Claude (or any MCP client) query analyzed
binaries directly: search functions, walk callers/callees, pull pseudocode,
rename, or kick off a fresh analysis pass on a new binary.

Two tiers of access, by design:
  - Fast/cached: search_functions, get_function, get_callees, get_callers,
    trace_chain — all hit the Neo4j graph built by scripts/populate_graph.py.
    Cheap, no IDA process involved, safe to call a lot.
  - Live/authoritative: get_full_pseudocode, rename_function, analyze_binary —
    these open the real .i64 via idalib. Slower, but ground truth (the cached
    graph stores a truncated pseudocode snippet; these don't).

get_function deliberately returns inline {address, name, size} for callees
AND callers in the same response — that's the whole point. The agent decides
whether to chain (call get_function again on a callee) by looking at whether
it's still sub_* right there in the result, instead of needing a separate
round trip just to find out there's nothing more to see.
"""
from __future__ import annotations

import asyncio
import re
import time
import uuid

from mcp.server.fastmcp import Context, FastMCP

from spectrida import config
from spectrida.api import IDADatabase
from spectrida.core.backend import RealBackend
from spectrida.core.graph import FunctionGraph

mcp = FastMCP("spectrida")

_SHARD_PROGRESS_RE = re.compile(r"(\d+)/(\d+) shards")

_graph: FunctionGraph | None = None
_live: dict[str, IDADatabase] = {}   # binary tag -> open live handle, opened lazily
_jobs: dict[str, dict] = {}          # job_id -> {"status", "result"|"error", "progress", "created"}


def _g() -> FunctionGraph:
    global _graph
    if _graph is None:
        _graph = FunctionGraph(config.graph_uri(), config.graph_user(), config.graph_password())
    return _graph


def _norm_addr(address: str) -> int:
    return int(address, 16) if isinstance(address, str) and address.startswith("0x") else int(address)


def _hexify(d: dict) -> dict:
    """Graph queries return addr as a raw int (correct for Cypher comparisons) —
    but a bare decimal int is hard for an LLM to read or match against the hex
    addresses it already has. Reformat to hex at the tool boundary only."""
    if "addr" in d and isinstance(d["addr"], int):
        d["address"] = hex(d.pop("addr"))
    return d


async def _live_db(binary: str) -> IDADatabase:
    """Lazily open (and cache) a live IDA handle for `binary`'s registered .i64.
    Reused across calls in this server's lifetime — reopening a large .i64 is slow.
    Closed explicitly at server shutdown via _close_all_live()."""
    if binary in _live:
        return _live[binary]
    path = _g().get_binary_path(binary)
    if not path:
        raise ValueError(f"no .i64 registered for binary tag '{binary}' — run populate_graph.py first, "
                          f"or call analyze_binary() on a fresh file")
    backend = RealBackend(path)
    await backend.ensure_open()
    db = IDADatabase(backend)
    _live[binary] = db
    return db


async def _close_all_live() -> None:
    for db in _live.values():
        await db.close()
    _live.clear()


async def _heartbeat(ctx: Context | None, message: str, interval_s: float = 12) -> None:
    """Loop sending progress pings while a long single-step operation (e.g.
    one unsharded NSO analysis pass) has no natural sub-progress events of
    its own — MCP clients use these to know a call is still alive and not
    apply their own idle/stuck-call timeout. Cancelled by the caller once the
    real work finishes; that cancellation is expected, not an error."""
    if not ctx:
        return
    tick = 0
    while True:
        await asyncio.sleep(interval_s)
        tick += 1
        await ctx.report_progress(tick, None, message=message)


@mcp.tool()
async def doctor() -> dict:
    """Check the health of every external dependency this server needs
    (llama-server for AI naming, Neo4j for the graph, idalib for IDA access)
    WITHOUT starting anything. Call this first if a tool call fails or
    behaves oddly — it tells you exactly what's down before you go digging.
    Use start_all() to actually fix what's reported missing/down."""
    from spectrida.core import services

    return {
        "llama_server": {"running": await services.llama_server_running(),
                          "configured": services.llama_server_configured()},
        "neo4j": {"running": await services.neo4j_running(),
                  "configured": services.neo4j_configured()},
        "idalib": {"configured": services.idalib_ok()},
    }


@mcp.tool()
async def start_all() -> dict:
    """Start every external dependency that's down but configured
    (llama-server for AI naming, Neo4j for the graph). Safe to call even if
    some/all are already running — it's a no-op for anything already up.
    idalib doesn't run as a persistent service so there's nothing to start
    for it; analyze_binary/get_full_pseudocode/etc. open it on demand.

    GPU model loads and JVM boots can take 1-5+ minutes, so — like
    analyze_binary — this returns IMMEDIATELY with a job_id. Use
    poll_analysis(job_id) to check progress and get the final result once
    status='done'. A service reported not-running AND not-configured in the
    final result means you need to set its path in ~/.spectrida/config.toml's
    [services] section first."""
    from spectrida.core import services

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "status": "running", "binary": "(services)",
        "progress": "starting services...", "created": time.time(),
        "result": None, "error": None,
    }

    async def _run() -> None:
        job = _jobs[job_id]
        try:
            job["progress"] = "starting llama-server (GPU model load can take a while)..."
            llama_ok = await services.ensure_llama_server()
            job["progress"] = "starting neo4j..."
            neo4j_ok = await services.ensure_neo4j()
            job["status"] = "done"
            job["result"] = {
                "llama_server": {"running": llama_ok, "configured": services.llama_server_configured()},
                "neo4j": {"running": neo4j_ok, "configured": services.neo4j_configured()},
            }
            job["progress"] = "complete"
        except Exception as exc:
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"

    asyncio.create_task(_run())
    return {"job_id": job_id, "status": "started",
            "hint": f"call poll_analysis('{job_id}') to check progress — can take a few minutes"}


@mcp.tool()
async def deindex_binary(binary: str) -> dict:
    """Remove a binary's nodes/edges from the graph (and its Binary registry
    entry) — use to clear out a bad/stale run before re-populating, or to free
    up Neo4j. Does NOT delete the .i64 file itself."""
    if binary in _live:
        await _live.pop(binary).close()
    deleted = _g().delete_binary(binary)
    return {"binary": binary, "functions_deleted": deleted}


@mcp.tool()
def list_binaries() -> list[dict]:
    """List every binary that's been indexed into the graph, with its tag,
    backing .i64 path, and naming coverage stats. Call this first if you
    don't already know the binary tag to use in other calls."""
    binaries = _g().list_binaries()
    for b in binaries:
        stats = _g().stats(b["tag"])
        b.update(stats)
    return binaries


@mcp.tool()
def search_functions(binary: str, query: str, limit: int = 20) -> list[dict]:
    """Search for functions by substring in their name within `binary`.
    Use this to find a starting point — e.g. search_functions('among_us', 'damage')
    to find combat-related code before you know any addresses."""
    return [_hexify(r) for r in _g().search_by_name(binary, query, limit)]


@mcp.tool()
def get_function(binary: str, address: str) -> dict:
    """Get full info on one function: name, size, a cached pseudocode snippet
    (may be truncated — call get_full_pseudocode for the complete body),
    cached disassembly (disasm: [{address, text}, ...] — the instruction-level
    layer pseudocode can't give you: exact instruction boundaries and operand
    bytes, needed before planning any actual byte/instruction patch), and its
    callees/callers inline as {address, name, size}.

    Use the inline callees/callers to decide whether to chain: if a callee is
    still named sub_* and looks load-bearing, call get_function on it too.
    If everything around it is already meaningfully named, you probably have
    enough context already — no need to keep digging. A callee with
    name: null hasn't been indexed yet — call get_full_pseudocode on its
    address to look at it live instead of relying on the cached graph.
    """
    addr = _norm_addr(address)
    fn = _g().get_function(binary, addr)
    if fn is None:
        return {"error": f"no function at {address} in '{binary}'"}
    fn["callees"] = [_hexify(c) for c in _g().callees(binary, addr)]
    fn["callers"] = [_hexify(c) for c in _g().callers(binary, addr)]
    return _hexify(fn)


@mcp.tool()
def get_callees(binary: str, address: str) -> list[dict]:
    """Just the functions called by `address` — {address, name, size} each.
    Lighter than get_function when you only need to see what's downstream."""
    return [_hexify(c) for c in _g().callees(binary, _norm_addr(address))]


@mcp.tool()
def get_callers(binary: str, address: str) -> list[dict]:
    """Just the functions that call `address` — {address, name, size} each.
    Useful for figuring out where/how a function is actually used."""
    return [_hexify(c) for c in _g().callers(binary, _norm_addr(address))]


@mcp.tool()
def trace_chain(binary: str, address: str, depth: int = 2) -> list[dict]:
    """Every function reachable within `depth` calls of `address`, deduplicated.
    Use this instead of repeated get_callees calls when you want the whole
    neighborhood at once (e.g. "what does this subsystem touch within 2 hops")."""
    return [_hexify(c) for c in _g().trace_chain(binary, _norm_addr(address), depth)]


@mcp.tool()
async def get_full_pseudocode(binary: str, address: str) -> str:
    """Full, untruncated decompiled pseudocode for one function, fetched live
    from the actual .i64 (not the cached snippet in the graph). Slower than
    get_function but authoritative — use when the cached snippet cut off
    mid-function and you need to see the rest."""
    db = await _live_db(binary)
    return await db.decompile(_norm_addr(address))


@mcp.tool()
async def rename_function(binary: str, address: str, new_name: str) -> dict:
    """Rename a function in the live .i64 AND update the cached graph to
    match, so future queries see the new name too. Use this once you've
    actually figured out what a function does — it's a real, persisted edit,
    not a suggestion."""
    db = await _live_db(binary)
    addr = _norm_addr(address)
    ok = await db.rename(addr, new_name)
    if ok:
        info = await db.info(addr) or {}
        pseudocode = await db.decompile(addr)
        disasm = await db.disasm(addr)
        _g().upsert_functions(binary, [{
            "addr": addr, "name": new_name,
            "size": info.get("size", 0), "pseudocode": pseudocode, "disasm": disasm,
        }])
    return {"renamed": ok, "address": address, "new_name": new_name}


@mcp.tool()
async def populate_binary(
    binary: str, limit: int | None = None, min_size: int = 0,
) -> dict:
    """Re-populate the Neo4j graph for an already-analyzed binary with full
    control over the naming pass. Use this when the initial analyze_binary
    only indexed the first N functions (default 2000) and you need everything
    in the graph. Set limit=None for all functions, min_size=0 to include
    even tiny thunks/stubs.

    This only runs the demangle + AI-naming pass on the existing .i64 — it
    does NOT re-run the slow parallel analysis.

    Returns immediately with a job_id — use poll_analysis() to check status."""
    from spectrida.core.populate import populate_graph

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "status": "running",
        "binary": binary,
        "progress": "demangling + AI naming...",
        "created": time.time(),
        "result": None,
        "error": None,
    }

    async def _run() -> None:
        job = _jobs[job_id]
        try:
            db = await _live_db(binary)

            async def _on_progress(done: int, total: int) -> None:
                job["progress"] = f"{done}/{total} functions processed"

            result = await populate_graph(
                db, _g(), binary,
                limit=limit,
                skip=0,
                min_size=min_size,
                name_chunk=8,
                on_progress=_on_progress,
            )
            job["status"] = "done"
            job["result"] = result
            job["progress"] = f"complete: {result}"
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
            job["progress"] = f"failed: {tb[-300:]}"

    asyncio.create_task(_run())
    return {
        "job_id": job_id,
        "status": "started",
        "binary": binary,
        "hint": f"call poll_analysis('{job_id}') to check progress",
    }


@mcp.tool()
async def list_jobs() -> list[dict]:
    """List all background analysis jobs (running, done, or failed).
    Each entry has job_id, status, binary tag, and a progress summary.
    Use poll_analysis(job_id) to get the full result once status is 'done'."""
    return [
        {
            "job_id": jid,
            "status": j["status"],
            "binary": j.get("binary", "?"),
            "progress": j.get("progress", ""),
            "created": j.get("created", 0),
        }
        for jid, j in _jobs.items()
    ]


@mcp.tool()
async def poll_analysis(job_id: str) -> dict:
    """Check the status of a background analysis job kicked off by
    analyze_binary. Returns the full result if done, or current progress
    if still running. Poll this every few seconds until status is 'done'
    or 'error' — the analysis can take minutes for large binaries."""
    job = _jobs.get(job_id)
    if not job:
        return {"error": f"no job '{job_id}' — it may have been cleaned up or never existed"}
    if job["status"] == "running":
        return {"job_id": job_id, "status": "running", "progress": job.get("progress", ""),
                "binary": job.get("binary", "?")}
    if job["status"] == "error":
        return {"job_id": job_id, "status": "error", "error": job.get("error", "unknown"),
                "binary": job.get("binary", "?")}
    return {"job_id": job_id, "status": "done", **job["result"]}


@mcp.tool()
async def analyze_binary(
    path: str, binary: str, workers: int | None = None,
    populate: bool = True, populate_limit: int | None = 2000,
    populate_min_size: int = 20, ctx: Context | None = None,
) -> dict:
    """Kick off spectrIDA's parallel analysis pipeline on a fresh binary
    (DLL/EXE/NSO/...) — this can take MINUTES for large binaries, so it
    returns IMMEDIATELY with a job_id. Use poll_analysis(job_id) to check
    progress and get the final result once status='done'.

    The pipeline: discover code segments → density-balanced sharding →
    parallel idalib workers → merge into one .i64 → (optionally) populate
    the Neo4j graph with AI-named functions.

    populate_limit caps how many functions get the AI-naming pass — raise
    it for a fuller pass, or set populate=False to skip naming entirely.
    Call this ONCE per binary, then poll for the result."""
    from spectrida.core.pipeline import run_analysis

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "status": "running",
        "binary": binary,
        "progress": "queued",
        "created": time.time(),
        "result": None,
        "error": None,
    }

    async def _run() -> None:
        job = _jobs[job_id]
        try:
            # ── phase 1: parallel analysis ──
            job["progress"] = "discovering code segments..."
            result = await run_analysis(path, workers, on_line=None)

            if "error" in result:
                job["status"] = "error"
                job["error"] = result["error"]
                return

            i64_path = result.get("i64")
            if not i64_path:
                job["status"] = "error"
                job["error"] = "analysis finished but no .i64 path was reported"
                return

            _g().register_binary(binary, i64_path)
            out = {
                "i64_path": i64_path,
                "function_count": result.get("funcs"),
                "elapsed_seconds": result.get("elapsed"),
                "binary": binary,
            }

            # ── phase 2: populate graph ──
            if populate:
                from spectrida.core.populate import populate_graph

                job["progress"] = "populating graph (AI naming)..."
                db = await _live_db(binary)

                async def _pop_progress(done: int, total: int) -> None:
                    job["progress"] = f"naming {done}/{total} functions"

                pop_result = await populate_graph(
                    db, _g(), binary,
                    limit=populate_limit,
                    min_size=populate_min_size,
                    on_progress=_pop_progress,
                )
                out["populate"] = pop_result

            job["status"] = "done"
            job["result"] = out
            job["progress"] = "complete"

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
            job["progress"] = f"failed: {tb[-400:]}"

    asyncio.create_task(_run())

    return {
        "job_id": job_id,
        "status": "started",
        "binary": binary,
        "hint": "call poll_analysis('" + job_id + "') to check progress — this will take minutes",
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()

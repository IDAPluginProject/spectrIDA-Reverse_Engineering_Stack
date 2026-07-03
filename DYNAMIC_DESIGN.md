# spectrIDA Dynamic Analysis — module + MCP design

Goal: add Atlas's dynamic corner (emulate / live-instrument / fuzz) to spectrIDA as
a **clean, self-contained module** with **thin MCP tool wrappers** — no rewrite of the
existing (already-clean) server. The agent driving the MCP server is the reasoning
corner; these tools are how it gets runtime facts onto the graph and how it feeds
Atlas seeds.

Principle: **separation, not rewrite.** All logic lives in `spectrida/dynamic/`; the
existing `mcp_server.py` only gains a handful of `@mcp.tool()` wrappers that call in.

---

## 1. Module layout — `spectrida/dynamic/`

```
spectrida/dynamic/
  __init__.py     # lazy import + friendly "pip install spectrida[atlas]" guard
  emulate.py      # one-shot emulation of a function  → verdict
  fuzz.py         # coverage-guided crash-hunt campaign (job-based)
  live.py         # Frida live instrumentation (trace + in-process fuzz)
  seeds.py        # seed acquisition: folder / carve-from-binary / bundled corpus
  annotate.py     # write dyn_* runtime facts onto graph Function nodes
```

Each wraps atlas code already built + tested (kept in the `atlas` package, pulled in
via the optional `[atlas]` extra):
- `emulate.py` → `atlas.analysis.spectrida_bridge.EmulatedBinary`
- `fuzz.py`    → the coverage-guided driver (`atlas.analysis.binary_fuzz` + guided loop)
- `live.py`    → `atlas.analysis.frida_live.FridaLiveTarget`
- `annotate.py`→ `atlas.analysis.graph_annotator.GraphAnnotator` (inject `_g().driver`)

### `__init__.py` — the availability guard (the whole "optional" story)
```python
"""Dynamic analysis. Requires the optional `atlas` extra."""
try:
    import atlas.analysis  # noqa
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

def require():
    if not AVAILABLE:
        raise RuntimeError(
            "Dynamic analysis needs Atlas. Install it:  pip install \"spectrida[atlas]\"")
```

---

## 2. Seed handling — `seeds.py` (the agent↔Atlas seam)

The design point you raised: the reasoning agent provides seeds. This module makes
that a first-class, three-source resolution — the agent's own fetched files take
priority, with automatic fallbacks so it also works unattended.

```python
def resolve_seeds(binary_path: str, seeds_dir: str | None,
                  fmt_hint: str | None = None) -> list[bytes]:
    """Seed corpus for a fuzz target, in priority order:
      1. seeds_dir      — a folder the AGENT populated (fetched/generated samples).
      2. carve_from(binary_path) — assets embedded in the target itself
         (magic-byte scan: PNG/TTF/OTTO/RIFF/zip/... → the target's own inputs).
      3. bundled(fmt_hint) — a tiny built-in corpus for common formats.
    Deduped; empty list is allowed (fuzzer will start from random, weakest case)."""
```

- **`seeds_dir` is the seam.** The agent reads a function's pseudocode via
  `get_function`, decides "this parses OpenType", fetches OTF samples with its own
  tools, writes them to a folder, and passes the path. No inline blobs through MCP.
- **`carve_from`** is the "user needs zero downloads" path — most real targets embed
  valid inputs (games bundle assets); carve them as seeds.
- Proven necessity: TTF-only seeds plateaued at 727 edges; agent-fetched OTF/Type1/BDF
  seeds → 4013 edges (+3286), turning a dead plateau into real exploration.

---

## 3. Thin MCP tools (add to `mcp_server.py`)

Mirror the existing patterns exactly: `@mcp.tool()` async, `_g()` for the graph,
`_norm_addr`/`_hexify` at the boundary, and the **existing `_jobs` + `poll_analysis`
job pattern** for anything long-running (reuse — don't reinvent). Each guards with
`dynamic.require()` so a missing extra gives the friendly install message.

```python
@mcp.tool()
async def emulate_function(binary: str, address: str) -> dict:
    """Chain-emulate ONE function (no OS, any arch) and report what happens when it
    runs: clean / needs_state / candidate_crash, coverage, stubbed calls. Writes the
    verdict onto the graph node (dyn_*). Fast, synchronous. Good for triage."""

@mcp.tool()
async def hunt_crashes(binary: str, address: str, seeds_dir: str = "",
                       budget_seconds: int = 300) -> dict:
    """Coverage-guided crash hunt on a function. Seeds resolve via seeds.resolve_seeds
    (agent's seeds_dir → carved-from-binary → bundled). Long-running → returns a
    job_id; poll with poll_analysis(). Annotates crashes + reproducing inputs onto
    the graph. This is the agent→Atlas seed-fed loop."""

@mcp.tool()
async def live_trace(binary: str, addresses: list[str], spawn: bool = True,
                     seconds: int = 3) -> dict:
    """Attach/spawn the RUNNING target (Frida) and capture real args/returns/coverage
    for the given functions — for state-entangled functions emulation returns
    needs_state on. Native, runnable targets only (not Switch). Annotates dyn_live_*."""
```

Notes:
- `emulate_function` sync (ms–seconds); `hunt_crashes` job-based (minutes);
  `live_trace` sync-ish (bounded by `seconds`).
- **No LLM tool.** The MCP-consuming agent IS the reasoning corner; it reads these
  results + `get_function` and reasons. spectrida-re/llama stay naming-only.

---

## 4. Annotation — `annotate.py`

Reuse `GraphAnnotator` with the graph's existing driver (no second Neo4j connection):
```python
from atlas.analysis.graph_annotator import GraphAnnotator
def annotator():
    return GraphAnnotator(driver=_g().driver, prefix="dyn_")  # product-native prefix
```
Props land on the `Function` node so `get_function` surfaces them for the next agent:
`dyn_status`, `dyn_reachable`, `dyn_blocks`, `dyn_crash_input`, `dyn_live_args`, …

---

## 5. Packaging

- `spectrida/pyproject.toml`: `[project.optional-dependencies] atlas = ["atlas-dynre>=0.1"]`.
- Onboarding offers it (opt-in, recommended). Absent extra → tools return the friendly
  install message, base spectrIDA unaffected (no torch/frida in the core install).

---

## 6. Build order (small, verifiable steps)
1. `dynamic/__init__.py` guard + `emulate.py` + `emulate_function` tool  → smoke test
   on `main.nso` (verdict lands on graph).
2. `annotate.py` (reuse GraphAnnotator) — confirm `get_function` shows `dyn_*`.
3. `seeds.py` (folder + carve) + `fuzz.py` + `hunt_crashes` (job-based) — confirm the
   agent→seeds_dir→crash loop end to end.
4. `live.py` + `live_trace` — confirm real args on a runnable native target.
5. Packaging + onboarding opt-in.

Each step is independently testable and leaves the server working — no big-bang rewrite.

---

## What NOT to do
- Don't build a fresh MCP server. The existing one is clean (FastMCP, good separation,
  job pattern, heartbeats). A rewrite trades working+tested code for rewrite risk. The
  clean boundary you want comes from the `dynamic/` module, not a new server.
- Don't pass seed blobs through MCP calls. Files on disk (agent writes, tool reads a
  path) is simpler, debuggable, and avoids giant tool payloads.
- Don't put an LLM in the loop. The agent calling the tools is the reasoner.

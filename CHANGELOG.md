# Changelog

## 0.2.0 — the ghost learns to talk back (chapter 2)

- **MCP server** (`spectrida mcp`) — Claude (or any MCP client) can search/read/chain through
  every analyzed binary directly: `search_functions`, `get_function` (pseudocode + full
  disassembly + inline callers/callees, for chaining without extra round trips),
  `get_callees`/`get_callers`/`trace_chain`, `get_full_pseudocode`, `rename_function`.
- **`spectrida install mcp`** — registers the server with Claude Code and pi automatically;
  no manual JSON, no separate `pip install` for the `mcp`/`neo4j` extras.
- **Neo4j-backed function graph** (`scripts/populate_graph.py`, or auto-chained from
  `analyze_binary`) — demangles every mangled name (Itanium *and* MSVC, both via IDA's own
  ABI-detecting demangler — previously only Itanium ever reached it), skips tiny thunks, sends
  only genuinely-stripped functions to the model, and now stores disassembly per function too
  (exact instruction boundaries/operands — the layer pseudocode can't give you).
- **`analyze_binary`** — one tool call takes a never-before-seen binary through the whole
  pipeline (parallel analysis → demangle → AI naming → graph), as a background job polled via
  `poll_analysis`/`list_jobs` so a multi-minute run never blocks the conversation. Reports live
  shard progress via MCP progress notifications.
- **NSO support** for the parallel analyzer — previously PE-only; equal-width parallel sharding
  (no PE-style file-zeroing, idalib's own loader handles decompression per worker).
- **`doctor`/`start_all`/`deindex_binary`** — check or boot llama-server + Neo4j from inside the
  conversation; clear out a bad/stale graph run without touching the `.i64`.
- Fixed: a cancelled `analyze_binary` call no longer orphans the analyzer subprocess (and its
  own shard-worker children); the analyzer no longer inherits the MCP transport's stdin pipe
  (a real hang vector specific to running under an MCP server, not standalone).

## 0.1.0 — first ghost

- Parallel sharded IDA analysis (Capstone recursive descent + idalib merge).
- AI function naming via a local Ollama model, streamed token-by-token.
- Terminal UI: virtualized function browser, syntax-highlighted disasm, decompiler view,
  call-chain explorer, inline rename, command palette.
- First-run onboarding wizard (humorous, skippable) that helps set up Ollama + the model.
- Demo mode (`spectrida --demo`) — runs the whole TUI with no IDA/Ollama.
- Config-driven everything (`~/.spectrida/config.toml` + env vars); no hardcoded paths.

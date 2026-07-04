<div align="center">

# 👻 spectrIDA Desktop

**Ghost through binaries — now with a face.**

A dark, spectral desktop workbench for spectrIDA: drop a binary, watch it index,
then browse every named function — pseudocode, disassembly, callers/callees — and
run the phantomrt dynamic tools (emulate / hunt crashes / live trace) on any
function, verdict stamped right on the graph.

</div>

```
┌ Binaries ┐┌ Functions ────────┐┌ Achievement::Achievement(AchievementInfo const*) ┐
│ main.nso ││ Achievement::…    ││ 0x7100378264   12 B   CLEAN                       │
│ 74,791   ││ AchievementHolder ││ [⚡ Emulate] [🩸 Hunt] [📡 Live]                  │
│ Spotify… ││ …calcMoonGet(…)   ││ __int64 __fastcall Achievement::Achievement(...)  │
└──────────┘└───────────────────┘└──────────────────────────────────────────────────┘
```

## What it is

The TUI is fast but terminal-bound. This is the same power in a real window:

- **Index** — drag a binary onto the ghost (or browse), watch the parallel pipeline
  run with a live progress bar, then it lands in the graph.
- **Browse** — a three-pane RE workbench: binaries → searchable function list →
  detail (pseudocode / disassembly / xrefs). Search by name or `0xADDRESS`.
- **Dynamic** — one click runs phantomrt on the selected function: **Emulate**
  (verdict: clean / needs-state / candidate-crash), **Hunt crashes** (fuzz +
  reproducing inputs), **Live trace** (Frida real args/returns). The verdict is
  written back onto the graph node and shown as a chip — so the next look, in the
  UI *or* an MCP agent, already sees it.

Everything runs **locally**. The UI is just a face on spectrIDA's own graph +
analysis + phantomrt modules — no logic lives here that isn't in the library.

## Architecture

```
Electron renderer  ──HTTP──▶  FastAPI backend  ──▶  spectrIDA graph / pipeline / phantomrt
 (this beautiful UI)          (desktop/backend)      (Neo4j · idalib · unicorn/frida)
```

- `main.js` boots the Python backend and manages its lifecycle.
- `backend/server.py` is a thin FastAPI layer over `spectrida.core.graph`,
  `spectrida.core.pipeline`, and `spectrida.dynamic` (phantomrt).
- `renderer/` is the workbench (vanilla HTML/CSS/JS — no framework, no slop).

## Run it

```bash
cd desktop
npm install          # electron
npm start            # boots backend + opens the window
```

Requirements: everything spectrIDA needs (IDA idalib, Neo4j running), plus Node 18+.
Dynamic tools need the `phantomrt` extra (`pip install "spectrida[atlas]"`). Set
`SPECTRIDA_PYTHON` if your interpreter isn't the default.

## Honest notes

- v0.1 — it drives the real pipeline and real graph; it is not yet packaged into a
  signed installer (run from source).
- Indexing a large binary still takes minutes (that's IDA, not the UI) — the
  progress bar keeps you company; the merge phase is the slow part.
- Dynamic verdicts carry phantomrt's honesty: `needs_state` and `candidate_crash`
  mean what they say — a candidate is a lead to verify, not a confirmed bug.

*The terminal ghost reads your binaries. This one lets you watch it work.* 👻

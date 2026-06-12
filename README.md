<div align="center">

# 👻 spectrIDA

**Ghost through binaries.**

Parallel IDA Pro analysis + AI function naming, in a terminal TUI that actually has a personality.

</div>

```
spectrida analyze GameAssembly.dll
```

```
◈  spectrIDA  ▸  GameAssembly.dll

  ✓ 00  ✓ 01  ✓ 02  ✓ 03  ▸ 04  · 05  · 06  · 07
  ✓ 08  ✓ 09  ✓ 10  ✓ 11  ✓ 12  ✓ 13  ▸ 14  · 15

  14/16 shards  │  141,203 functions found
  ████████████████████████████░░░░  89%  ~4s remaining
```

---

## What it is

IDA Pro's auto-analysis is single-threaded. On a 34 MB il2cpp DLL that's *minutes* — sometimes a lot
of them. spectrIDA splits the binary into N shards, runs Capstone recursive descent across all of
them in parallel, merges into one `.i64`, then lets a fine-tuned Qwen3 model **name every function** —
all from one terminal UI.

It is not Ghidra. It is not going to win a Pwnie. It does one annoying thing (slow analysis + naming)
fast, and it's genuinely fun to use. That's the whole pitch.

**147,288 functions · 56 seconds · 16 workers**

## Features

- **Parallel sharded analysis** — splits the binary into address-space shards, runs them in parallel
  via idalib, merges into a single `.i64`.
- **AI function naming** — a fine-tuned Qwen3-8B model ([on HuggingFace](https://huggingface.co/gdfhhjk/spectrida-re-gguf))
  runs locally via Ollama and streams names token-by-token. Batch-name a whole selection at once.
- **A TUI that's actually nice** — virtualized function browser (handles 147k+), syntax-highlighted
  disasm, decompiler view, **call-chain explorer**, inline rename, command palette.
- **A first-run wizard with a sense of humor** — helps you install Ollama + the model, then gets out
  of your way (skippable, never nags twice).
- **Demo mode** (`spectrida --demo`) — try the whole TUI with **zero setup** (no IDA, no Ollama).

## Requirements

- **IDA Pro 9.x** with idalib (ships with IDA Pro)
- **Python 3.10+**
- **Ollama** — [install](https://ollama.com/download) (the wizard will walk you through it)

## Install

```bash
pip install spectrida
spectrida          # first run launches the friendly setup wizard
```

Or skip the ceremony:

```bash
spectrida --demo   # see the TUI right now, no setup
```

## Usage

```bash
spectrida analyze path/to/binary.dll   # parallel analysis, then opens the browser
spectrida open path/to/database.i64    # open an existing .i64
spectrida onboard                      # re-run the setup wizard
spectrida --demo                       # canned demo, no IDA/Ollama
```

### Keybinds

| Key | Action |
|-----|--------|
| `N` | Name selected function (AI, streams live) |
| `R` | Rename (pre-fills the AI suggestion) |
| `D` | Toggle decompiled pseudocode |
| `C` | Call chain (callers / callees) |
| `/` | Fuzzy search |
| `?` | Help |
| `Q` | Quit |

## The model

`spectrida-re` is a Qwen3-8B fine-tuned for reverse engineering — reading assembly, naming functions,
tracing call chains. It's trained with **neuron-targeted SFT + GRPO** (only the RE-relevant neurons are
tuned, which keeps it from forgetting everything else). Pull it:

```bash
ollama pull hf.co/gdfhhjk/spectrida-re-gguf
```

## License

MIT. If it works for you, cool. If it doesn't, blame the GGUF quantization, not us. 👻

# Changelog

## 0.1.0 — first ghost

- Parallel sharded IDA analysis (Capstone recursive descent + idalib merge).
- AI function naming via a local Ollama model, streamed token-by-token.
- Terminal UI: virtualized function browser, syntax-highlighted disasm, decompiler view,
  call-chain explorer, inline rename, command palette.
- First-run onboarding wizard (humorous, skippable) that helps set up Ollama + the model.
- Demo mode (`spectrida --demo`) — runs the whole TUI with no IDA/Ollama.
- Config-driven everything (`~/.spectrida/config.toml` + env vars); no hardcoded paths.

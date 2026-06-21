"""`spectrida install mcp` — wires the MCP server into Claude Code and/or pi
automatically: no manual JSON editing, no separate pip step for the graph
extras. Safe to re-run; every step is idempotent (re-registering an already-
registered server, or re-merging an already-present config entry, is a no-op
or a harmless overwrite of the same values).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from spectrida import voice

_SERVER_ENTRY = {"command": sys.executable, "args": ["-m", "spectrida.mcp_server"]}


def _ensure_packages() -> str:
    try:
        import mcp  # noqa: F401
        import neo4j  # noqa: F401
        return "deps: mcp + neo4j already present"
    except ImportError:
        pass
    print("  fetching mcp + neo4j (the parts a bare `pip install spectrida` skips)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "mcp>=1.0", "neo4j>=5.0"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return "deps: installed mcp + neo4j"
    return f"deps: pip install failed — {result.stderr.strip()[-200:]}"


def _configure_claude_code() -> str:
    claude = shutil.which("claude")
    if not claude:
        return "claude-code: not found on PATH, skipped (install the Claude Code CLI first)"
    try:
        result = subprocess.run(
            [claude, "mcp", "add", "-s", "user", "spectrida", "--",
             sys.executable, "-m", "spectrida.mcp_server"],
            capture_output=True, text=True, timeout=30,
        )
        out = (result.stdout + result.stderr).lower()
        if result.returncode == 0 or "already exists" in out:
            return "claude-code: registered (restart Claude Code to pick it up)"
        return f"claude-code: failed — {(result.stderr or result.stdout).strip()[:200]}"
    except Exception as e:
        return f"claude-code: error — {e}"


def _configure_pi() -> str:
    pi = shutil.which("pi")
    if not pi:
        return "pi: not found on PATH, skipped"
    try:
        listed = subprocess.run([pi, "list"], capture_output=True, text=True, timeout=30).stdout
        if "pi-mcp-adapter" not in listed:
            subprocess.run([pi, "install", "npm:pi-mcp-adapter"], capture_output=True, text=True, timeout=180)
    except Exception:
        pass  # adapter install is best-effort; the config write below still helps if it's already present

    cfg_path = Path.home() / ".pi" / "agent" / "mcp.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    cfg.setdefault("mcpServers", {})
    cfg["mcpServers"]["spectrida"] = dict(_SERVER_ENTRY)
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return "pi: registered (restart pi to pick it up)"


def install_mcp() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # emoji on Windows consoles default to cp1252
    except AttributeError:
        pass
    print(f"\U0001f47b {voice.quip('mcp_install')}\n")
    print(f"  {_ensure_packages()}")
    for line in (_configure_claude_code(), _configure_pi()):
        mark = "✓" if "registered" in line or "already present" in line else (
            "·" if "skipped" in line else "✗")
        print(f"  {mark} {line}")
    print(f"\n{voice.quip('goodbye')}")

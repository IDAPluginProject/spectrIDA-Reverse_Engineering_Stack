"""Service checks for Ollama + idalib — used by the CLI and the onboarding wizard."""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import httpx

from spectrida.config import idalib_dir, ollama_model, ollama_url

# ── Ollama ──────────────────────────────────────────────────────────────────

def ollama_installed() -> bool:
    return shutil.which("ollama") is not None


def ollama_install_hint() -> str:
    if sys.platform == "win32":
        return "winget install Ollama.Ollama   (or download from https://ollama.com/download)"
    if sys.platform == "darwin":
        return "brew install ollama   (or download from https://ollama.com/download)"
    return "curl -fsSL https://ollama.com/install.sh | sh"


async def ollama_running() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            return (await c.get(f"{ollama_url()}/api/tags")).status_code == 200
    except Exception:
        return False


async def ensure_ollama() -> bool:
    """True if Ollama is reachable; tries to start `ollama serve` if not."""
    if await ollama_running():
        return True
    if not ollama_installed():
        return False
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return False
    for _ in range(20):
        await asyncio.sleep(0.5)
        if await ollama_running():
            return True
    return False


async def model_present(model: str | None = None) -> bool:
    model = model or ollama_model()
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            tags = (await c.get(f"{ollama_url()}/api/tags")).json()
        names = [m.get("name", "") for m in tags.get("models", [])]
        return any(model in n for n in names)
    except Exception:
        return False


async def ensure_model_loaded() -> bool:
    """Warm the model so the first real inference isn't cold."""
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            await c.post(f"{ollama_url()}/api/generate", json={
                "model": ollama_model(), "prompt": "hi", "stream": False,
                "options": {"num_predict": 1},
            })
        return True
    except Exception:
        return False


# ── idalib ──────────────────────────────────────────────────────────────────

def idalib_ok(path: str | None = None) -> bool:
    """Cheap validity check that `path` looks like an IDA install with idalib."""
    p = Path(path or idalib_dir())
    if not path and not idalib_dir():
        return False
    if not p.is_dir():
        return False
    markers = ["idalib.dll", "libidalib.so", "libidalib.dylib", "idapro.py"]
    return any((p / m).exists() for m in markers) or any(p.glob("**/idapro.py"))

"""Ollama streaming client for function naming."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from spectrida.config import ollama_model, ollama_url

_SYSTEM = (
    "You are an expert reverse engineer specialising in C++ game binaries. "
    "Given x86-64 assembly and call-chain context, output a concise snake_case "
    "function name followed by a SHORT reasoning (3-5 sentences max). "
    "Format:\nNAME: <name>\nREASON: <reasoning>"
)


def _insn_line(i: dict) -> str:
    # disasm rows are {"address", "text"}; fall back to mnemonic/op_str if present
    text = i.get("text") or f"{i.get('mnemonic', '')}  {i.get('op_str', '')}".strip()
    return f"  {i.get('address', ''):>16}  {text}"


def _build_prompt(insns: list[dict], callees: list[str], callers: list[str]) -> str:
    asm_lines = "\n".join(_insn_line(i) for i in insns[:80])
    return (
        f"Calls: {', '.join(callees[:8]) or 'none'}\n"
        f"Called by: {', '.join(callers[:8]) or 'none'}\n\n"
        f"Assembly:\n{asm_lines}\n\n"
        "Name this function:"
    )


async def stream_name(
    insns: list[dict],
    callees: list[str],
    callers: list[str],
) -> AsyncIterator[str]:
    """Yield response tokens one at a time as the model writes the name + reason."""
    payload = {
        "model": ollama_model(),
        "system": _SYSTEM,
        "prompt": _build_prompt(insns, callees, callers),
        "stream": True,
        "options": {"temperature": 0.2, "num_predict": 256},
    }
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{ollama_url()}/api/generate", json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])
                if chunk.get("response"):
                    yield chunk["response"]
                if chunk.get("done"):
                    break


async def name_function(insns: list[dict], callees: list[str], callers: list[str]) -> str:
    """Non-streaming convenience used by batch mode — returns the extracted name."""
    full = "".join([tok async for tok in stream_name(insns, callees, callers)])
    return extract_name(full) or ""


def extract_name(full_text: str) -> str | None:
    for line in full_text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("NAME:"):
            rest = stripped[5:].strip()
            candidate = rest.split()[0] if rest else ""
            if candidate and candidate.isidentifier():
                return candidate
    return None

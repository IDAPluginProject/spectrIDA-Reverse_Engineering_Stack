"""Canned data so the TUI runs with no IDA and no Ollama (for --demo + the tutorial)."""
from __future__ import annotations

import asyncio

# A tiny fake il2cpp-ish database. Some functions are named; the sub_* ones are
# there for you to "name" with the (fake) model during the demo/tutorial.
FUNCTIONS: list[dict] = [
    {"name": "GameManager$$Update",      "start": 0x140001000, "end": 0x1400010C0, "size": 192},
    {"name": "Player$$TakeDamage",       "start": 0x140001100, "end": 0x140001210, "size": 272},
    {"name": "Player$$Respawn",          "start": 0x140001220, "end": 0x1400012E0, "size": 192},
    {"name": "sub_1400013A0",            "start": 0x1400013A0, "end": 0x140001460, "size": 192},
    {"name": "Enemy$$Attack",            "start": 0x140001480, "end": 0x140001560, "size": 224},
    {"name": "sub_140001600",            "start": 0x140001600, "end": 0x1400016A0, "size": 160},
    {"name": "Inventory$$AddItem",       "start": 0x140001700, "end": 0x1400017F0, "size": 240},
    {"name": "sub_140001820",            "start": 0x140001820, "end": 0x1400018B0, "size": 144},
    {"name": "SaveSystem$$Serialize",    "start": 0x140001900, "end": 0x140001A40, "size": 320},
    {"name": "sub_140001A80",            "start": 0x140001A80, "end": 0x140001B20, "size": 160},
    {"name": "NetworkClient$$SendPacket","start": 0x140001B40, "end": 0x140001C80, "size": 320},
    {"name": "sub_140001D00",            "start": 0x140001D00, "end": 0x140001D90, "size": 144},
]

_DISASM = {
    0x1400013A0: [
        ("0x1400013a0", "push    rbp"),
        ("0x1400013a1", "mov     rbp, rsp"),
        ("0x1400013a4", "movss   xmm0, dword ptr [rcx+0x40]"),
        ("0x1400013a9", "subss   xmm0, dword ptr [rdx]"),
        ("0x1400013ad", "movss   dword ptr [rcx+0x40], xmm0"),
        ("0x1400013b2", "comiss  xmm0, dword ptr [rip+0x1c4a]"),
        ("0x1400013ba", "ja      0x1400013d0"),
        ("0x1400013bc", "call    Player$$Respawn"),
        ("0x1400013c1", "xor     eax, eax"),
        ("0x1400013c3", "pop     rbp"),
        ("0x1400013c4", "ret"),
    ],
}
_DEFAULT_DISASM = [
    ("0x140000000", "push    rbp"),
    ("0x140000001", "mov     rbp, rsp"),
    ("0x140000004", "mov     rax, qword ptr [rcx]"),
    ("0x140000007", "test    rax, rax"),
    ("0x14000000a", "je      0x140000020"),
    ("0x14000000c", "call    qword ptr [rax+0x18]"),
    ("0x14000000f", "pop     rbp"),
    ("0x140000010", "ret"),
]

# callee links keyed by function start
_XREFS_FROM = {
    0x1400013A0: [{"address": "0x140001220", "name": "Player$$Respawn"}],
    0x140001100: [{"address": "0x1400013a0", "name": "sub_1400013A0"}],
    0x140001000: [{"address": "0x140001100", "name": "Player$$TakeDamage"},
                  {"address": "0x140001480", "name": "Enemy$$Attack"}],
}
_XREFS_TO = {
    0x140001220: [{"address": "0x1400013a0", "name": "sub_1400013A0"}],
    0x1400013A0: [{"address": "0x140001100", "name": "Player$$TakeDamage"}],
    0x140001100: [{"address": "0x140001000", "name": "GameManager$$Update"}],
}

# what the (fake) model "decides" sub_* functions should be called
_DEMO_NAMES = {
    0x1400013A0: ("apply_fall_damage",
                  "Subtracts a delta from a float health field at [rcx+0x40], clamps, "
                  "and calls Player$$Respawn when it drops below zero. Classic damage tick."),
    0x140001600: ("normalize_vector3", "Reads three floats, computes inverse sqrt of the sum of squares, scales."),
    0x140001820: ("hash_string_fnv", "FNV-1a loop over a byte buffer — multiply by prime, xor next byte."),
    0x140001A80: ("clamp_health", "min/max guard on a float field, writes it back."),
    0x140001D00: ("crc32_block", "Table-driven CRC over a length-prefixed buffer."),
}


def _norm(addr) -> int:
    if isinstance(addr, int):
        return addr
    return int(addr, 16) if str(addr).startswith("0x") else int(addr)


def disasm(addr) -> list[dict]:
    rows = _DISASM.get(_norm(addr), _DEFAULT_DISASM)
    return [{"address": a, "text": t} for a, t in rows]


def decompile(addr) -> str:
    a = _norm(addr)
    if a in _DEMO_NAMES:
        return ("// (demo) reconstructed pseudocode\n"
                "void __fastcall demo(Entity *e, float *delta) {\n"
                "    e->health -= *delta;\n"
                "    if (e->health < 0.0)\n"
                "        Player__Respawn(e);\n"
                "}\n")
    return "// (demo) no pseudocode for this one — try a sub_ function."


def xrefs_from(addr) -> list[dict]:
    return _XREFS_FROM.get(_norm(addr), [])


def xrefs_to(addr) -> list[dict]:
    return _XREFS_TO.get(_norm(addr), [])


async def stream_name(addr):
    """Fake token-by-token model output for the demo model pane."""
    name, reason = _DEMO_NAMES.get(_norm(addr), ("demo_function", "A perfectly cromulent function. (demo mode — no real model running.)"))
    text = f"NAME: {name}\nREASON: {reason}"
    for chunk in text.split(" "):
        await asyncio.sleep(0.03)
        yield chunk + " "

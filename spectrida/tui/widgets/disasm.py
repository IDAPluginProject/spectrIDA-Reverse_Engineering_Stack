"""Disassembly / decompiler pane + the assembly syntax highlighter."""
from __future__ import annotations

import re

from rich.style import Style
from rich.text import Text
from textual.widgets import RichLog

_REG = re.compile(
    r"\b(r[a-z0-9]{1,3}|e[a-z]{2}|[abcd][lhx]|sil|dil|bpl|spl|xmm\d+|ymm\d+)\b"
)
_HEX = re.compile(r"\b(0x[0-9a-fA-F]+)\b")


def is_sub(name: str) -> bool:
    return name.startswith("sub_") or name.startswith("j_") or name.startswith("nullsub_")


def fmt_size(size: int) -> str:
    if size <= 0:
        return ""
    return f"{size}b" if size < 1024 else f"{size // 1024}kb"


def highlight(address: str, text: str) -> Text:
    t = Text()
    t.append(f"{address:>14}  ", Style(color="#374151"))
    parts = text.split(None, 1)
    mn = parts[0] if parts else ""
    rest = parts[1] if len(parts) > 1 else ""
    ml = mn.lower()
    if ml.startswith("call"):
        col = "#f97316"
    elif ml.startswith("j"):
        col = "#fbbf24"
    elif ml in ("ret", "retn", "retf", "leave"):
        col = "#ef4444"
    elif ml in ("push", "pop"):
        col = "#8b5cf6"
    elif ml in ("mov", "lea", "movzx", "movsx", "movss", "movups", "xchg"):
        col = "#00d4ff"
    elif ml in ("xor", "and", "or", "not", "shl", "shr", "sar", "rol", "ror", "test"):
        col = "#ec4899"
    else:
        col = "#c9d1d9"
    t.append(f"{mn:<8}", Style(color=col, bold=ml.startswith(("call", "j")) or ml in ("ret", "retn")))
    # operands: colour registers + hex
    pos = 0
    for m in sorted([*_REG.finditer(rest), *_HEX.finditer(rest)], key=lambda x: x.start()):
        if m.start() < pos:
            continue
        t.append(rest[pos:m.start()], Style(color="#9ca3af"))
        is_hex = m.group(0).startswith("0x")
        t.append(m.group(0), Style(color="#f97316" if is_hex else "#fbbf24"))
        pos = m.end()
    t.append(rest[pos:], Style(color="#9ca3af"))
    return t


class DisasmPane(RichLog):
    def __init__(self, **kw):
        super().__init__(highlight=False, markup=False, wrap=False, **kw)

    def show_disasm(self, insns: list[dict]) -> None:
        self.clear()
        if not insns:
            self.write(Text("no disassembly. spooky.", Style(color="#475569")))
            return
        for i in insns:
            self.write(highlight(i.get("address", ""), i.get("text", "")))

    def show_decompile(self, code: str) -> None:
        self.clear()
        for line in (code or "// nothing here").splitlines():
            s = line.lstrip()
            if s.startswith("//"):
                col = "#475569"
            elif any(s.startswith(k) for k in ("if", "for", "while", "return", "else", "switch")):
                col = "#ec4899"
            elif "(" in line and ")" in line and ";" not in line.split("(")[0]:
                col = "#00d4ff"
            else:
                col = "#c9d1d9"
            self.write(Text(line, Style(color=col)))

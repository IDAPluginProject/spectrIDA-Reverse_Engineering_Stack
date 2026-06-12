"""Bottom status bar — live info on the left, a rotating ghost quip on the right."""
from __future__ import annotations

from rich.style import Style
from rich.text import Text
from textual.widgets import Static

from spectrida import voice


class StatusBar(Static):
    def __init__(self, **kw):
        super().__init__(id="statusbar", **kw)
        self._info = ""
        self._quip = voice.quip("idle")

    def on_mount(self) -> None:
        self.set_interval(20, self._tick)

    def _tick(self) -> None:
        self._quip = voice.quip("idle")
        self.refresh()

    def set_info(self, info: str) -> None:
        self._info = info
        self.refresh()

    def render(self) -> Text:
        t = Text()
        t.append(self._info, Style(color="#64748b"))
        w = self.size.width or 80
        pad = max(1, w - len(self._info) - len(self._quip) - 2)
        t.append(" " * pad)
        t.append(self._quip, Style(color="#475569", italic=True))
        return t

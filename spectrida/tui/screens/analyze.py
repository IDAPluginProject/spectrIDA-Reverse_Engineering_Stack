"""Progress screen shown while the parallel analyzer runs."""
from __future__ import annotations

import asyncio
import re

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import RichLog, Static

from spectrida import voice
from spectrida.core.pipeline import run_analysis
from spectrida.tui.widgets.statusbar import StatusBar

_SHARD_DONE = re.compile(r"shard (\d+):\s*([\d,]+) funcs")
_PROG = re.compile(r"(\d+)/(\d+) shards")
_FUNCS = re.compile(r"([\d,]+) funcs")
_LAUNCH = re.compile(r"launching (\d+) workers")


class AnalyzeScreen(Screen):
    def __init__(self, binary: str, workers: int | None = None):
        super().__init__()
        self._binary = binary
        self._workers = workers or 16
        self._n_funcs = 0

    def compose(self) -> ComposeResult:
        from pathlib import Path
        yield Static(f"◈  spectrIDA  ▸  analyzing  {Path(self._binary).name}", id="analyze-binary")
        with Vertical(id="shard-grid"):
            for i in range(16):
                yield Static(f"{i:02d}", classes="shard-cell", id=f"shard-{i}")
        yield Static("", id="analyze-stats")
        yield RichLog(id="analyze-log", highlight=False, markup=False, auto_scroll=True)
        yield StatusBar()

    def _spawn(self, coro):
        t = asyncio.create_task(coro)
        self._tasks = getattr(self, '_tasks', set())
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return t

    def on_mount(self) -> None:
        self.query_one(StatusBar).set_info(f"sharding {self._binary}")
        self.call_after_refresh(lambda: self._spawn(self._run()))

    def _set_shard(self, idx: int, state: str) -> None:
        if 0 <= idx < 16:
            cell = self.query_one(f"#shard-{idx}", Static)
            cell.set_classes(f"shard-cell {state}")
            cell.update({"running": "▸", "done": "✓", "error": "✗"}.get(state, f"{idx:02d}") + f" {idx:02d}")

    async def _run(self) -> None:
        log = self.query_one("#analyze-log", RichLog)
        stats = self.query_one("#analyze-stats", Static)
        log.write(Text(f"  {voice.quip('analyzing')}", Style(color="#8b5cf6", italic=True)))

        async def on_line(line: str) -> None:
            log.write(Text(line, Style(color="#3a4456")))
            if (m := _LAUNCH.search(line)):
                for i in range(min(int(m.group(1)), 16)):
                    self._set_shard(i, "running")
            if (m := _SHARD_DONE.search(line)):
                self._set_shard(int(m.group(1)), "done")
            if (m := _FUNCS.search(line)):
                self._n_funcs = max(self._n_funcs, int(m.group(1).replace(",", "")))
            done = "?"
            if (m := _PROG.search(line)):
                done = m.group(1)
            stats.update(f"  {done} shards  │  {self._n_funcs:,} functions found")

        result = await run_analysis(self._binary, self._workers, on_line)

        if result.get("error") or not result.get("i64"):
            stats.update(f"  ✗ {result.get('error', 'analysis failed')}")
            log.write(Text(f"  {voice.quip('error')}", Style(color="#ef4444")))
            return

        i64 = result["i64"]
        stats.update(f"  ✓ done · {result.get('funcs', self._n_funcs):,} funcs · "
                     f"{result.get('elapsed', '?')}s · opening browser…")
        from spectrida.core.backend import RealBackend
        from spectrida.tui.screens.browser import BrowserScreen
        backend = RealBackend(i64)
        try:
            await backend.open()
        except Exception as e:
            stats.update(f"  ✗ opened analysis but idalib failed: {e}")
            return
        await self.app.switch_screen(BrowserScreen(backend))

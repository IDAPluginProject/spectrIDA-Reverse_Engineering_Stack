"""spectrIDA TUI app shell — routing + first-run onboarding gate."""
from __future__ import annotations

from textual.app import App


def _patch_textual_unmount_bug() -> None:
    # Textual 8.x raises AttributeError from Widget._on_unmount on some teardown
    # paths. Swallow it so quitting is always clean.
    try:
        from textual import widget as _w
        _orig = _w.Widget._on_unmount

        def _safe(self):
            try:
                _orig(self)
            except AttributeError:
                pass
        _w.Widget._on_unmount = _safe
    except Exception:
        pass


_patch_textual_unmount_bug()


class SpectrIDAApp(App):
    CSS_PATH = "styles.tcss"
    TITLE = "spectrIDA"

    def __init__(self, *, demo: bool = False, i64: str | None = None,
                 binary: str | None = None, workers: int | None = None):
        super().__init__()
        self._demo = demo
        self._i64 = i64
        self._binary = binary
        self._workers = workers

    def on_mount(self) -> None:
        # Onboarding is handled as a text flow at the CLI before the app launches.
        self.push_screen(self._build_screen())

    def _build_screen(self):
        # Screens open their own backend lazily (in their worker), so this stays sync.
        from spectrida.core.backend import DemoBackend, RealBackend
        from spectrida.tui.screens.analyze import AnalyzeScreen
        from spectrida.tui.screens.browser import BrowserScreen

        if self._binary:
            return AnalyzeScreen(self._binary, self._workers)
        if self._i64 and not self._demo:
            return BrowserScreen(RealBackend(self._i64))
        return BrowserScreen(DemoBackend())

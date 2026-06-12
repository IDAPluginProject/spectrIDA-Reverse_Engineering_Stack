"""Modal dialogs — rename + help overlay."""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, Static

from spectrida import voice


class RenameDialog(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "cancel")]

    def __init__(self, current: str, suggested: str | None = None):
        super().__init__()
        self._current = current
        self._suggested = suggested

    def compose(self) -> ComposeResult:
        with Vertical(id="rename-dialog"):
            yield Label(" ✎  rename function", id="rename-title")
            yield Input(value=self._suggested or self._current,
                        placeholder="new_function_name", id="rename-input")
            yield Label("↵ confirm   ·   esc cancel", id="dialog-hint")

    @on(Input.Submitted)
    def _submit(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)


class HelpScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape,question_mark,q", "dismiss", "close")]

    _KEYS = [
        ("N", "name the selected function (AI)"),
        ("R", "rename (pre-fills the AI suggestion)"),
        ("D", "toggle decompiled pseudocode"),
        ("C", "call chain — callers / callees"),
        ("B", "batch-name selected sub_ functions"),
        ("/", "fuzzy search"),
        ("ctrl+p", "command palette"),
        ("Q", "quit"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Label(" ?  spectrIDA — keys", id="help-title")
            body = "\n".join(f"  [b cyan]{k:<7}[/]  {d}" for k, d in self._KEYS)
            yield Static(body, id="help-body", markup=True)
            yield Static(f"\n  [dim]{voice.quip('idle')}[/]", markup=True)
            yield Label("esc / ? to close", id="dialog-hint")

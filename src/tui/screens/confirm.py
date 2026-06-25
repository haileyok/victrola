"""Reusable confirmation dialog screen for destructive actions."""

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Middle
from textual.screen import ModalScreen
from textual.widgets import Footer, Label


class ConfirmScreen(ModalScreen[bool]):
    """A modal confirmation dialog. Returns True (confirm) or False (cancel)."""

    BINDINGS = [
        Binding("y,enter", "confirm", "Yes"),
        Binding("n,escape", "cancel", "No"),
    ]

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._message = message

    def compose(self) -> ComposeResult:
        with Center():
            with Middle():
                yield Label(self._message, id="confirm-message")
                yield Label("[bold]Y[/bold] to confirm · [bold]N[/bold] to cancel", markup=True)
        yield Footer()

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

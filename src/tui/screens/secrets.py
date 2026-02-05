import logging
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView

logger = logging.getLogger(__name__)


class SecretListScreen(Screen):
    """Screen for managing secrets (name → value pairs stored on PDS).

    Secrets are injected as environment variables into custom tool Deno processes.
    The agent can see secret names but never their values.
    """

    BINDINGS = [
        Binding("n", "new_secret", "New"),
        Binding("d", "delete_secret", "Delete"),
        Binding("escape", "back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(name="Secrets")
        yield ListView(id="secret-list")
        yield Footer()

    async def on_mount(self) -> None:
        await self._refresh_secrets()

    def _get_manager(self) -> Any:
        executor = getattr(self.app, "executor", None)
        if executor is None:
            return None
        return getattr(executor, "_secret_manager", None)

    async def _refresh_secrets(self) -> None:
        list_view = self.query_one("#secret-list", ListView)
        await list_view.clear()

        manager = self._get_manager()
        if manager is None:
            await list_view.append(
                ListItem(Label("Secret manager not available (no PDS connection)."))
            )
            return

        # reload from PDS
        try:
            await manager.load_secrets()
        except Exception:
            logger.exception("Failed to reload secrets")

        names = manager.list_secret_names()
        if not names:
            await list_view.append(
                ListItem(
                    Label(
                        "No secrets configured. Press [bold]n[/bold] to add one.",
                        markup=True,
                    )
                )
            )
            return

        for name in names:
            value = manager.get_secret(name) or ""
            masked = "*" * min(len(value), 8) + "..." if len(value) > 8 else "*" * len(value)
            label = f"{name}  =  {masked}"
            item = ListItem(Label(label))
            item._secret_name = name  # type: ignore[attr-defined]
            await list_view.append(item)

    async def action_new_secret(self) -> None:
        """Push a simple input screen to add a secret."""
        self.app.push_screen(SecretInputScreen(on_save=self._on_secret_saved))

    async def _on_secret_saved(self, name: str, value: str) -> None:
        manager = self._get_manager()
        if manager is None:
            self.notify("No secret manager", severity="error")
            return

        try:
            result = await manager.set_secret(name, value)
            self.notify(result)
            await self._refresh_secrets()
        except Exception:
            logger.exception("Failed to save secret")
            self.notify("Failed to save secret", severity="error")

    async def action_delete_secret(self) -> None:
        manager = self._get_manager()
        if manager is None:
            self.notify("No secret manager", severity="error")
            return

        list_view = self.query_one("#secret-list", ListView)
        if list_view.highlighted_child is None:
            return
        name = getattr(list_view.highlighted_child, "_secret_name", None)
        if not name:
            return

        try:
            result = await manager.delete_secret(name)
            self.notify(result)
            await self._refresh_secrets()
        except Exception:
            logger.exception("Failed to delete secret")
            self.notify("Failed to delete secret", severity="error")

    def action_back(self) -> None:
        self.app.pop_screen()


class SecretInputScreen(Screen):
    """Simple input screen for entering a secret name and value."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        on_save: Any = None,
        prefill_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._on_save = on_save
        self._prefill_name = prefill_name

    def compose(self) -> ComposeResult:
        yield Header(name="Add Secret")
        yield Label("Secret name (e.g. OPENAI_API_KEY):", id="name-label")
        yield Input(
            value=self._prefill_name or "",
            placeholder="SECRET_NAME",
            id="secret-name-input",
        )
        yield Label("Secret value:", id="value-label")
        yield Input(placeholder="secret value", password=True, id="secret-value-input")
        yield Label("Press Enter in the value field to save.", id="hint-label")
        yield Footer()

    def on_mount(self) -> None:
        # If name is prefilled, jump straight to the value field
        if self._prefill_name:
            self.query_one("#secret-value-input", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "secret-name-input":
            # move focus to value input
            self.query_one("#secret-value-input", Input).focus()
            return

        if event.input.id == "secret-value-input":
            name_input = self.query_one("#secret-name-input", Input)
            value_input = self.query_one("#secret-value-input", Input)
            name = name_input.value.strip()
            value = value_input.value.strip()

            if not name:
                self.notify("Name is required", severity="error")
                return
            if not value:
                self.notify("Value is required", severity="error")
                return

            self.app.pop_screen()
            if self._on_save:
                await self._on_save(name, value)

    def action_cancel(self) -> None:
        self.app.pop_screen()

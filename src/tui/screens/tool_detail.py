"""Detailed review screen for a single custom tool."""

import json
import logging
from typing import Any

from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from src.tui.screens.secrets import SecretInputScreen

logger = logging.getLogger(__name__)


class ToolDetailScreen(Screen):
    """Read + review view for a single custom tool, with approve/revoke/delete.

    On approve: if the tool references any secrets that aren't yet set,
    detours through the secret input flow for each missing secret before
    completing approval.
    """

    BINDINGS = [
        Binding("a", "approve", "Approve"),
        Binding("r", "revoke", "Revoke"),
        Binding("d", "delete", "Delete"),
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, tool_name: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tool_name = tool_name
        self._pending_missing_secrets: list[str] = []

    def _manager(self) -> Any:
        executor = getattr(self.app, "executor", None)
        if executor is None:
            return None
        return getattr(executor, "_custom_tool_manager", None)

    def _secret_manager(self) -> Any:
        executor = getattr(self.app, "executor", None)
        if executor is None:
            return None
        return getattr(executor, "_secret_manager", None)

    def compose(self) -> ComposeResult:
        yield Header(name=f"Tool: {self._tool_name}")
        yield VerticalScroll(id="tool-detail-body")
        yield Footer()

    async def on_mount(self) -> None:
        await self._populate()

    async def _populate(self) -> None:
        body = self.query_one("#tool-detail-body", VerticalScroll)
        await body.remove_children()

        manager = self._manager()
        if manager is None:
            await body.mount(Static("Custom tool manager not available."))
            return

        tool = manager.get_tool(self._tool_name)
        if tool is None:
            await body.mount(Static(f"Tool '{self._tool_name}' not found."))
            return

        status = "approved" if tool.approved else "pending review"
        await body.mount(Static(f"[b]Status:[/b] {status}", markup=True))
        await body.mount(Static(f"[b]Description:[/b] {tool.description}", markup=True))

        # parameters
        await body.mount(Static("[b]Parameters:[/b]", markup=True))
        params_json = json.dumps(tool.parameters, indent=2)
        await body.mount(
            Static(
                Syntax(params_json, "json", theme="monokai", word_wrap=True),
                classes="tool-detail-code",
            )
        )

        # secrets: show which are set vs missing
        sm = self._secret_manager()
        known = set(sm.list_secret_names()) if sm is not None else set()
        if tool.secrets:
            lines = ["[b]Secrets:[/b]"]
            for name in tool.secrets:
                badge = "[green]set[/green]" if name in known else "[yellow]missing[/yellow]"
                lines.append(f"- `{name}` {badge}")
            await body.mount(Static("\n".join(lines), markup=True))

        # code (syntax-highlighted)
        await body.mount(Static("[b]Code:[/b]", markup=True))
        await body.mount(
            Static(
                Syntax(tool.code, "typescript", theme="monokai", word_wrap=True),
                classes="tool-detail-code",
            )
        )

    def _missing_secrets(self) -> list[str]:
        manager = self._manager()
        sm = self._secret_manager()
        if manager is None or sm is None:
            return []
        tool = manager.get_tool(self._tool_name)
        if tool is None:
            return []
        known = set(sm.list_secret_names())
        return [s for s in tool.secrets if s not in known]

    async def action_approve(self) -> None:
        missing = self._missing_secrets()
        if missing:
            self._pending_missing_secrets = list(missing)
            await self._prompt_next_secret()
            return
        await self._do_approve()

    async def _prompt_next_secret(self) -> None:
        """Walk through missing secrets one at a time, then approve."""
        if not self._pending_missing_secrets:
            await self._do_approve()
            return

        next_name = self._pending_missing_secrets[0]

        async def on_saved(name: str, value: str) -> None:
            sm = self._secret_manager()
            if sm is None:
                self.notify("Secret manager not available", severity="error")
                return
            try:
                await sm.set_secret(name, value)
            except Exception:
                logger.exception("Failed to save secret during approval flow")
                self.notify(f"Failed to save secret {name}", severity="error")
                return
            # move on to the next missing secret (or approve)
            if self._pending_missing_secrets and self._pending_missing_secrets[0] == name:
                self._pending_missing_secrets.pop(0)
            await self._prompt_next_secret()

        self.app.push_screen(
            SecretInputScreen(on_save=on_saved, prefill_name=next_name)
        )

    async def _do_approve(self) -> None:
        manager = self._manager()
        if manager is None:
            self.notify("Custom tool manager not available", severity="error")
            return
        try:
            result = await manager.approve_tool(self._tool_name)
            self.notify(result)
            await self._populate()
        except Exception:
            logger.exception("Failed to approve tool")
            self.notify("Failed to approve tool", severity="error")

    async def action_revoke(self) -> None:
        manager = self._manager()
        if manager is None:
            return
        try:
            result = await manager.revoke_tool(self._tool_name)
            self.notify(result)
            await self._populate()
        except Exception:
            logger.exception("Failed to revoke tool")
            self.notify("Failed to revoke tool", severity="error")

    async def action_delete(self) -> None:
        manager = self._manager()
        if manager is None:
            return
        try:
            result = await manager.delete_tool(self._tool_name)
            self.notify(result)
            self.app.pop_screen()
        except Exception:
            logger.exception("Failed to delete tool")
            self.notify("Failed to delete tool", severity="error")

    def action_back(self) -> None:
        self.app.pop_screen()

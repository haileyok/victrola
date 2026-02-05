import logging
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, ListItem, ListView

logger = logging.getLogger(__name__)


class ToolListScreen(Screen):
    """Screen showing all custom tools with approval controls."""

    BINDINGS = [
        Binding("a", "approve_tool", "Approve"),
        Binding("r", "revoke_tool", "Revoke"),
        Binding("enter", "view_tool", "View"),
        Binding("v", "view_tool", "View"),
        Binding("s", "manage_secrets", "Secrets"),
        Binding("escape", "back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(name="Custom Tools")
        yield ListView(id="tool-list")
        yield Footer()

    async def on_mount(self) -> None:
        await self._refresh_tools()

    async def on_screen_resume(self) -> None:
        # refresh when returning from the detail screen so state changes show
        await self._refresh_tools()

    def _get_manager(self) -> Any:
        executor = getattr(self.app, "executor", None)
        if executor is None:
            return None
        return getattr(executor, "_custom_tool_manager", None)

    async def _refresh_tools(self) -> None:
        list_view = self.query_one("#tool-list", ListView)
        await list_view.clear()

        manager = self._get_manager()
        if manager is None:
            await list_view.append(
                ListItem(Label("Custom tool manager not available."))
            )
            return

        # reload from PDS
        try:
            await manager.load_tools()
        except Exception:
            logger.exception("Failed to reload custom tools")

        tools = manager.list_tools()
        if not tools:
            await list_view.append(
                ListItem(Label("No custom tools found. Create one via the agent."))
            )
            return

        for tool in tools:
            status = "[green]approved[/green]" if tool.approved else "[yellow]pending[/yellow]"
            label = f"{tool.name}  {status}  {tool.description[:60]}"
            item = ListItem(Label(label, markup=True))
            item._tool_name = tool.name  # type: ignore[attr-defined]
            await list_view.append(item)

    async def action_approve_tool(self) -> None:
        manager = self._get_manager()
        if manager is None:
            self.notify("No custom tool manager", severity="error")
            return

        list_view = self.query_one("#tool-list", ListView)
        if list_view.highlighted_child is None:
            return
        name = getattr(list_view.highlighted_child, "_tool_name", None)
        if not name:
            return

        try:
            result = await manager.approve_tool(name)
            self.notify(result)
            await self._refresh_tools()
        except Exception:
            logger.exception("Failed to approve tool")
            self.notify("Failed to approve tool", severity="error")

    async def action_revoke_tool(self) -> None:
        manager = self._get_manager()
        if manager is None:
            self.notify("No custom tool manager", severity="error")
            return

        list_view = self.query_one("#tool-list", ListView)
        if list_view.highlighted_child is None:
            return
        name = getattr(list_view.highlighted_child, "_tool_name", None)
        if not name:
            return

        try:
            result = await manager.revoke_tool(name)
            self.notify(result)
            await self._refresh_tools()
        except Exception:
            logger.exception("Failed to revoke tool")
            self.notify("Failed to revoke tool", severity="error")

    def action_view_tool(self) -> None:
        list_view = self.query_one("#tool-list", ListView)
        if list_view.highlighted_child is None:
            return
        name = getattr(list_view.highlighted_child, "_tool_name", None)
        if not name:
            return

        from src.tui.screens.tool_detail import ToolDetailScreen

        self.app.push_screen(ToolDetailScreen(tool_name=name))

    def action_manage_secrets(self) -> None:
        from src.tui.screens.secrets import SecretListScreen

        self.app.push_screen(SecretListScreen())

    def action_back(self) -> None:
        self.app.pop_screen()

import logging
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, ListItem, ListView

logger = logging.getLogger(__name__)


class SessionListScreen(Screen):
    """Screen showing all chat sessions with options to create/delete/open."""

    BINDINGS = [
        Binding("n", "new_session", "New Session"),
        Binding("d", "delete_session", "Delete"),
        Binding("enter", "open_session", "Open"),
        Binding("t", "manage_tools", "Tools"),
        Binding("s", "manage_secrets", "Secrets"),
        Binding("c", "manage_schedules", "Schedules"),
        Binding("p", "view_prompt", "Prompt"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(name="Victrola")
        yield Label("", id="startup-banner")
        yield ListView(id="session-list")
        yield Footer()

    async def on_mount(self) -> None:
        await self._refresh_banner()
        await self._refresh_sessions()

    async def on_screen_resume(self) -> None:
        await self._refresh_banner()
        await self._refresh_sessions()

    async def _refresh_banner(self) -> None:
        banner = self.query_one("#startup-banner", Label)
        try:
            from src.config import CONFIG

            executor = self.app.executor  # type: ignore[attr-defined]
            agent = self.app.agent  # type: ignore[attr-defined]

            # model
            model_name = getattr(
                getattr(agent, "_client", None), "_model_name", CONFIG.model_name
            )

            # scheduler task count
            scheduler = getattr(executor, "_scheduler", None)
            task_count = len(scheduler.list_tasks()) if scheduler else 0

            # secret count
            sm = getattr(executor, "_secret_manager", None)
            secret_count = len(sm.list_secret_names()) if sm else 0

            # custom tools: pending / approved counts
            mgr = getattr(executor, "_custom_tool_manager", None)
            if mgr:
                all_tools = mgr.list_tools()
                approved = sum(1 for t in all_tools if t.approved)
                pending = len(all_tools) - approved
                tools_str = f"{approved} approved, {pending} pending"
            else:
                tools_str = "—"

            # discord on/off
            discord_on = bool(sm and sm.get_secret("DISCORD_BOT_TOKEN"))

            banner.update(
                f"[b]Model:[/b] {model_name}  ·  "
                f"[b]Discord:[/b] {'on' if discord_on else 'off'}  ·  "
                f"[b]Schedules:[/b] {task_count}  ·  "
                f"[b]Secrets:[/b] {secret_count}  ·  "
                f"[b]Custom tools:[/b] {tools_str}"
            )
        except Exception:
            logger.exception("banner render failed")
            banner.update("")

    def _chat_store(self) -> Any:
        executor = self.app.executor  # type: ignore[attr-defined]
        store = getattr(executor._ctx, "_store", None)
        if store is None:
            return None
        return store.chat

    async def _refresh_sessions(self) -> None:
        list_view = self.query_one("#session-list", ListView)
        await list_view.clear()

        chat = self._chat_store()
        if chat is None:
            await list_view.append(
                ListItem(Label("Store not initialized — cannot load sessions."))
            )
            return

        try:
            resp = await chat.list_sessions(limit=50)
            sessions: list[dict[str, Any]] = resp.get("sessions", [])
        except Exception:
            logger.exception("Failed to list sessions")
            await list_view.append(ListItem(Label("Failed to load sessions.")))
            return

        if not sessions:
            await list_view.append(
                ListItem(Label("No sessions yet. Press [bold]n[/bold] to create one."))
            )
            return

        for session in sessions:
            rkey = session.get("rkey", "")
            created = session.get("createdAt", "")[:19]
            title = session.get("title", "")
            preview = title if title else "(untitled)"
            label = f"{rkey}  {created}  {preview}"
            item = ListItem(Label(label))
            item._session_rkey = rkey  # type: ignore[attr-defined]
            await list_view.append(item)

    async def action_new_session(self) -> None:
        chat = self._chat_store()
        if chat is None:
            self.notify("Store not initialized", severity="error")
            return

        try:
            resp = await chat.create_session()
            logger.info("create_session response: %s", resp)
            rkey = resp.get("rkey", "")
            if rkey:
                from src.tui.screens.chat import ChatScreen

                self.app.push_screen(ChatScreen(session_id=rkey))
            else:
                self.notify("Failed to create session", severity="error")
        except Exception:
            logger.exception("Failed to create session")
            self.notify("Failed to create session", severity="error")

    async def action_delete_session(self) -> None:
        list_view = self.query_one("#session-list", ListView)
        if list_view.highlighted_child is None:
            return
        item = list_view.highlighted_child
        rkey = getattr(item, "_session_rkey", None)
        if not rkey:
            return

        chat = self._chat_store()
        if chat is None:
            return

        try:
            await chat.delete_session(rkey=rkey)
            self.notify(f"Deleted session {rkey}")
            await self._refresh_sessions()
        except Exception:
            logger.exception("Failed to delete session")
            self.notify("Failed to delete session", severity="error")

    def action_manage_tools(self) -> None:
        from src.tui.screens.tools import ToolListScreen

        self.app.push_screen(ToolListScreen())

    def action_view_prompt(self) -> None:
        from src.tui.screens.system_prompt import SystemPromptScreen

        self.app.push_screen(SystemPromptScreen())

    def action_manage_secrets(self) -> None:
        from src.tui.screens.secrets import SecretListScreen

        self.app.push_screen(SecretListScreen())

    def action_manage_schedules(self) -> None:
        from src.tui.screens.schedules import ScheduleListScreen

        self.app.push_screen(ScheduleListScreen())

    @on(ListView.Selected)
    async def on_list_selected(self, event: ListView.Selected) -> None:
        await self._open_selected()

    async def action_open_session(self) -> None:
        await self._open_selected()

    async def _open_selected(self) -> None:
        list_view = self.query_one("#session-list", ListView)
        if list_view.highlighted_child is None:
            return
        item = list_view.highlighted_child
        rkey = getattr(item, "_session_rkey", None)
        if not rkey:
            return

        from src.tui.screens.chat import ChatScreen

        self.app.push_screen(ChatScreen(session_id=rkey))

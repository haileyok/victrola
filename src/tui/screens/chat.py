import logging
import time

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Static

from src.agent.agent import AgentEvent
from src.config import CONFIG
from src.tui.widgets.message import MessageBubble, ToolActivity
from src.tui.widgets.status_bar import AgentStatusBar
from src.tui.widgets.token_stats import TokenStatsBar

logger = logging.getLogger(__name__)


class ChatScreen(Screen):
    """Full-screen chat with an agent session."""

    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("T", "manage_tools", "Tools"),
    ]

    def __init__(self, session_id: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.session_id = session_id
        self._current_tool_widget: ToolActivity | None = None
        self._llm_start_time: float | None = None

    def compose(self) -> ComposeResult:
        yield Header(name="Victrola Chat")
        yield VerticalScroll(id="chat-log")
        yield Static("", id="pending-banner", classes="hidden")
        # Yield order matters for dock:bottom stacking — earlier yields end up
        # closer to the edge. Footer first so it's the absolute bottom row.
        yield Footer()
        yield Input(placeholder="Type a message...", id="chat-input")
        yield AgentStatusBar(id="status-bar")
        yield TokenStatsBar(context_limit=CONFIG.context_limit, id="token-stats")

    async def on_mount(self) -> None:
        conv_manager = self.app.conversation_manager  # type: ignore[attr-defined]
        agent = self.app.agent  # type: ignore[attr-defined]

        if conv_manager:
            try:
                messages = await conv_manager.load_session(self.session_id)
                if messages:
                    # Acquire agent's chat lock before touching _conversation so
                    # we don't clobber a concurrent Discord/scheduler chat()'s
                    # in-flight override.
                    async with agent._chat_lock:
                        agent._conversation = messages
                    chat_log = self.query_one("#chat-log", VerticalScroll)
                    for msg in messages:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            text_parts = [
                                b.get("text", "") for b in content
                                if isinstance(b, dict) and b.get("type") == "text"
                            ]
                            content = "\n".join(text_parts)
                        if content:
                            await chat_log.mount(MessageBubble(role=role, content=content))
                    chat_log.scroll_end(animate=False)
            except Exception:
                logger.exception("Failed to load session history")
        else:
            async with agent._chat_lock:
                agent._conversation = []

        self.query_one("#chat-input", Input).focus()
        await self._refresh_pending_banner()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return

        input_widget = self.query_one("#chat-input", Input)
        input_widget.value = ""
        input_widget.disabled = True

        chat_log = self.query_one("#chat-log", VerticalScroll)
        await chat_log.mount(MessageBubble(role="user", content=user_text))
        chat_log.scroll_end(animate=False)

        # persist user message
        conv_manager = self.app.conversation_manager  # type: ignore[attr-defined]
        if conv_manager:
            try:
                await conv_manager.save_message(
                    self.session_id, {"role": "user", "content": user_text}
                )
            except Exception:
                logger.exception("Failed to persist user message")

        self._run_agent(user_text)

    def _run_agent(self, user_text: str) -> None:
        """Launch agent.chat() as a Textual worker."""
        self.run_worker(self._do_agent_work(user_text), exclusive=True)

    async def _on_agent_event(self, event: AgentEvent) -> None:
        """Callback for agent events — updates the status bar and token stats."""
        status_bar = self.query_one("#status-bar", AgentStatusBar)
        chat_log = self.query_one("#chat-log", VerticalScroll)
        token_stats = self.query_one("#token-stats", TokenStatsBar)

        match event.kind:
            case "llm_start":
                status_bar.show_thinking()
                self._llm_start_time = time.monotonic()
            case "llm_done":
                status_bar.hide()
                usage = event.data.get("usage")
                elapsed: float | None = None
                if self._llm_start_time is not None:
                    elapsed = time.monotonic() - self._llm_start_time
                    self._llm_start_time = None
                if usage:
                    token_stats.add_usage(usage, elapsed_s=elapsed)
            case "tool_start":
                tool_name = event.data.get("tool", "unknown")
                code = event.data.get("code", "")
                status_bar.show_tool(tool_name)
                widget = ToolActivity(tool_name=tool_name, code=code)
                self._current_tool_widget = widget
                await chat_log.mount(widget)
                chat_log.scroll_end(animate=False)
            case "tool_done":
                status_bar.hide()
                result = event.data.get("result")
                success = bool(event.data.get("success", True))
                if self._current_tool_widget is not None and result is not None:
                    self._current_tool_widget.set_result(result, success)
                self._current_tool_widget = None
                # refresh pending tools banner after any tool call —
                # the agent may have created a new custom tool
                await self._refresh_pending_banner()

    async def _do_agent_work(self, user_text: str) -> None:
        """Run agent.chat() and display the response."""
        agent = self.app.agent  # type: ignore[attr-defined]
        conv_manager = self.app.conversation_manager  # type: ignore[attr-defined]
        status_bar = self.query_one("#status-bar", AgentStatusBar)
        chat_log = self.query_one("#chat-log", VerticalScroll)
        input_widget = self.query_one("#chat-input", Input)

        try:
            response = await agent.chat(user_text, on_event=self._on_agent_event)
            status_bar.hide()

            if response:
                await chat_log.mount(MessageBubble(role="assistant", content=response))
                chat_log.scroll_end(animate=False)

                # persist assistant response
                if conv_manager:
                    try:
                        await conv_manager.save_message(
                            self.session_id,
                            {"role": "assistant", "content": response},
                        )
                    except Exception:
                        logger.exception("Failed to persist assistant message")

                # try to auto-generate a session title after the first real turn
                try:
                    from src.agent.conversation import maybe_generate_session_title

                    store = self.app.executor._ctx._store  # type: ignore[attr-defined]
                    if store is not None:
                        await maybe_generate_session_title(
                            store, self.session_id, conv_manager._llm if conv_manager else None
                        )
                except Exception:
                    logger.exception("Failed to auto-generate session title")
        except Exception as e:
            logger.exception("Agent chat failed")
            status_bar.hide()
            # escape markup brackets so error messages containing URLs or
            # rich-style markup don't crash the Markdown renderer
            err_msg = str(e).replace("[", r"\[").replace("]", r"\]")
            err_text = f"**{type(e).__name__}**: {err_msg}"
            if len(err_text) > 2000:
                err_text = err_text[:2000] + "\n... (truncated — see victrola.log for full traceback)"
            await chat_log.mount(
                MessageBubble(
                    role="system",
                    content=f"Agent chat failed:\n\n{err_text}\n\nFull traceback in `victrola.log`.",
                )
            )
            chat_log.scroll_end(animate=False)
        finally:
            input_widget.disabled = False
            input_widget.focus()

    async def action_go_back(self) -> None:
        agent = self.app.agent  # type: ignore[attr-defined]
        # Lock so we don't clear _conversation while another surface's
        # chat() is mid-flight using it.
        async with agent._chat_lock:
            agent._conversation = []
        self.app.pop_screen()

    def action_manage_tools(self) -> None:
        from src.tui.screens.tools import ToolListScreen

        self.app.push_screen(ToolListScreen())

    async def _refresh_pending_banner(self) -> None:
        """Show a banner at the top of the chat log if any custom tools are pending review."""
        executor = self.app.executor  # type: ignore[attr-defined]
        manager = getattr(executor, "_custom_tool_manager", None)
        banner = self.query_one("#pending-banner", Static)
        if manager is None:
            banner.add_class("hidden")
            return
        pending = [t for t in manager.list_tools() if not t.approved]
        if not pending:
            banner.add_class("hidden")
            banner.update("")
            return
        banner.remove_class("hidden")
        n = len(pending)
        names = ", ".join(t.name for t in pending[:3])
        if len(pending) > 3:
            names += f", +{len(pending) - 3} more"
        banner.update(
            f"⚠ {n} custom tool{'s' if n != 1 else ''} pending review "
            f"({names}) — press [bold]T[/bold] to review"
        )

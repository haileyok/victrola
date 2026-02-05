"""Read-only view of the agent's current (freshly-rebuilt) system prompt.

Useful for debugging "why doesn't the agent know X?" — shows exactly the text
the agent sees, including live state: secret names, approved custom tools,
skills, the self-note, and tool docs.
"""

import logging
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

logger = logging.getLogger(__name__)


class SystemPromptScreen(Screen):
    """Show the current rendered system prompt."""

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("escape", "back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(name="Current System Prompt")
        yield VerticalScroll(id="prompt-body")
        yield Footer()

    async def on_mount(self) -> None:
        await self._render_prompt()

    async def on_screen_resume(self) -> None:
        # refresh when returning from elsewhere
        await self._render_prompt()

    async def _render_prompt(self) -> None:
        body = self.query_one("#prompt-body", VerticalScroll)
        await body.remove_children()

        agent = getattr(self.app, "agent", None)
        if agent is None:
            await body.mount(Static("No agent available."))
            return

        # Prefer the live-rebuilt prompt so the operator sees current state,
        # not whatever was cached at boot.
        text: str
        try:
            provider = getattr(agent, "_system_prompt_provider", None)
            if provider is not None:
                text = await provider()
            else:
                text = getattr(agent, "_system_prompt", None) or "(no prompt set)"
        except Exception as e:
            logger.exception("Failed to fetch current system prompt")
            text = f"(error fetching prompt: {e})"

        header = (
            f"[dim]{len(text)} chars · ~{len(text)//4} tokens · "
            "press 'r' to refresh[/dim]\n\n"
        )
        # Render as plain text (markup=False so brackets / code / URLs pass through)
        await body.mount(Static(header, markup=True))
        await body.mount(Static(text, markup=False, classes="prompt-text"))

    def action_refresh(self) -> None:
        self.run_worker(self._render_prompt(), exclusive=True)

    def action_back(self) -> None:
        self.app.pop_screen()

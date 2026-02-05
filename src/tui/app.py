import logging
from typing import Any

from textual.app import App
from textual.binding import Binding

logger = logging.getLogger(__name__)


class VictrolaApp(App):
    """Textual TUI for Victrola chat sessions."""

    CSS_PATH = "app.tcss"
    TITLE = "Victrola"

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
    ]

    def __init__(
        self,
        agent: Any,
        executor: Any,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        self.conversation_manager: Any = None
        self.executor = executor

    async def on_mount(self) -> None:
        """Initialize all async services then show session list."""
        try:
            await self.executor.initialize()
            logger.info("Executor initialized")
        except Exception:
            logger.exception("Executor init failed")

        # build conversation manager now that the store is ready
        try:
            from src.agent.conversation import ConversationManager
            from src.agent.llm import SubAgentLLM
            from src.config import CONFIG

            sub_api_key = CONFIG.sub_model_api_key or CONFIG.model_api_key
            llm_client = None
            if sub_api_key:
                llm_client = SubAgentLLM(
                    api=CONFIG.sub_model_api,
                    model=CONFIG.sub_model_name,
                    api_key=sub_api_key,
                    endpoint=CONFIG.sub_model_endpoint or None,
                )
            self.conversation_manager = ConversationManager(
                ctx=self.executor._ctx,
                llm_client=llm_client,
            )
        except Exception:
            logger.exception("Failed to build conversation manager")

        # wire a prompt provider so the agent rebuilds the system prompt
        # at the start of every chat() turn — new secrets, approved tools,
        # self-note edits, and skill additions propagate without a restart.
        try:
            from main import _load_system_prompt

            async def _refresh_prompt() -> str:
                return await _load_system_prompt(self.executor._ctx, self.executor)

            self.agent._system_prompt_provider = _refresh_prompt
            self.agent._system_prompt = await _refresh_prompt()
            logger.info("System prompt loaded")
        except Exception:
            logger.exception("System prompt load failed")

        # wire scheduler callback and start in background
        if self.executor._scheduler:
            import asyncio

            self.executor._scheduler._on_fire = self._on_schedule_fire
            asyncio.create_task(self.executor._scheduler.run())
            logger.info("Scheduler started in background")

        # start Discord chat bot if DISCORD_BOT_TOKEN is configured
        try:
            from main import _build_discord_bot

            bot = _build_discord_bot(self.executor, self.agent)
            if bot is not None:
                import asyncio

                asyncio.create_task(bot.start())
                logger.info("Discord bot started in background")
        except Exception:
            logger.exception("Discord bot failed to start")

        from src.tui.screens.sessions import SessionListScreen

        self.push_screen(SessionListScreen())

    async def _on_schedule_fire(self, task_name: str, prompt: str) -> str:
        """Handle a scheduled task firing. Creates a dedicated chat session,
        runs the agent in an isolated conversation, and saves messages."""
        tagged_prompt = f"[Scheduled task: {task_name}] {prompt}"

        # create a dedicated session for this scheduled run
        session_id: str | None = None
        chat_store = getattr(self.executor._ctx, "_store", None)
        if chat_store is not None and chat_store.chat is not None:
            try:
                resp = await chat_store.chat.create_session(title=f"scheduled: {task_name}")
                session_id = resp.get("rkey") or None
                logger.info("Created session '%s' for schedule '%s'", session_id, task_name)
            except Exception:
                logger.exception("Failed to create session for schedule '%s'", task_name)

        # save the prompt as a user message
        if session_id and self.conversation_manager:
            try:
                await self.conversation_manager.save_message(
                    session_id, {"role": "user", "content": tagged_prompt}
                )
            except Exception:
                logger.exception("Failed to save scheduled prompt to session")

        # run the scheduled prompt in an isolated conversation. agent.chat()
        # swaps + restores _conversation internally under its lock, so no
        # manual snapshot is needed.
        try:
            response = await self.agent.chat(tagged_prompt, conversation_override=[])
            logger.info(
                "Schedule '%s' completed: %s",
                task_name,
                (response[:200] if response else "(empty)"),
            )

            if session_id and self.conversation_manager and response:
                try:
                    await self.conversation_manager.save_message(
                        session_id, {"role": "assistant", "content": response}
                    )
                except Exception:
                    logger.exception("Failed to save scheduled response to session")

            return response
        except Exception:
            logger.exception("Schedule '%s' agent call failed", task_name)
            return ""

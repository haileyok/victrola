import asyncio
import logging
from collections.abc import Callable
from typing import Any, Literal

import click
import httpx

from src.agent.agent import Agent
from src.agent.config_utils import (
    resolve_sub_agent_endpoint as _resolve_sub_agent_endpoint,
    resolve_sub_agent_key as _resolve_sub_agent_key,
)
from src.agent.prompt import build_system_prompt
from src.config import CONFIG
from src.tools.executor import ToolExecutor
from src.tools.registry import TOOL_REGISTRY, ToolContext

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("victrola.log"),
    ],
)

logger = logging.getLogger(__name__)

# disable httpx/httpcore verbose logging
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
httpcore_logger = logging.getLogger("httpcore")
httpcore_logger.setLevel(logging.WARNING)


SHARED_OPTIONS: list[Callable[..., Callable[..., object]]] = [
    click.option("--model-api"),
    click.option("--model-name"),
    click.option("--model-api-key"),
    click.option("--model-endpoint"),
]


def shared_options[F: Callable[..., object]](func: F) -> F:
    for option in reversed(SHARED_OPTIONS):
        func = option(func)  # type: ignore[assignment]
    return func


def build_services(
    model_api: Literal["anthropic", "openai", "openapi", "umans"] | None,
    model_name: str | None,
    model_api_key: str | None,
    model_endpoint: str | None,
) -> tuple[ToolExecutor, Agent]:
    exa_client = None
    if CONFIG.exa_api_key:
        from exa_py import Exa

        exa_client = Exa(api_key=CONFIG.exa_api_key)

    http_client = httpx.AsyncClient(timeout=60.0)

    from src.agent.llm import SubAgentLLM

    effective_model_api = model_api or CONFIG.model_api
    effective_model_api_key = model_api_key or CONFIG.model_api_key

    sub_api_key = _resolve_sub_agent_key(
        model_api=effective_model_api,
        model_api_key=effective_model_api_key,
        sub_model_api=CONFIG.sub_model_api,
        sub_model_api_key=CONFIG.sub_model_api_key,
    )
    llm_client = None
    if sub_api_key:
        sub_endpoint = _resolve_sub_agent_endpoint()
        llm_client = SubAgentLLM(
            api=CONFIG.sub_model_api,
            model=CONFIG.sub_model_name,
            api_key=sub_api_key,
            endpoint=sub_endpoint,
        )

    tool_context = ToolContext(
        exa_client=exa_client,
        llm_client=llm_client,
        http_client=http_client,
    )

    executor = ToolExecutor(
        registry=TOOL_REGISTRY,
        ctx=tool_context,
    )

    agent = Agent(
        model_api=effective_model_api,
        model_name=model_name or CONFIG.model_name,
        model_api_key=model_api_key or CONFIG.model_api_key,
        model_endpoint=model_endpoint or CONFIG.model_endpoint or None,
        tool_executor=executor,
        sub_llm_client=llm_client,
        compact_threshold_chars=CONFIG.compact_threshold_chars,
    )

    return executor, agent


def _wire_scheduler(executor: ToolExecutor, agent: Agent) -> None:
    """Wire the scheduler callback to the agent after initialization."""
    if executor.scheduler is not None:

        async def on_schedule_fire(task_name: str, prompt: str) -> str:
            try:
                response = await agent.chat(
                    f"[Scheduled task: {task_name}] {prompt}", conversation=[]
                )
            except Exception:
                logger.exception("Scheduled task '%s' raised an exception", task_name)
                await _send_default_notification(
                    executor,
                    f"Scheduled task '{task_name}' raised an exception. "
                    "Check victrola.log for details.",
                    title=f"Error: {task_name}",
                )
                raise

            if not response:
                await _send_default_notification(
                    executor,
                    f"Scheduled task '{task_name}' returned an empty response.",
                    title=f"Error: {task_name}",
                )

            return response

        executor.scheduler._on_fire = on_schedule_fire

        async def run_condition(
            code: str, requires_net: bool, secrets: list[str]
        ) -> dict[str, Any]:
            env: dict[str, str] = {}
            sm = executor.secret_manager
            if sm:
                for name in secrets:
                    val = sm.get_secret(name)
                    if val:
                        env[name.upper()] = val
            return await executor.execute_condition_code(
                code=code, env=env, allow_net=requires_net
            )

        executor.scheduler._condition_runner = run_condition


async def _send_default_notification(
    executor: ToolExecutor, content: str, title: str = ""
) -> None:
    """Send an error notification to the default channel (Signal if configured, else Discord).

    Only called as a safety net when a scheduled task raises an exception or
    returns an empty response. Normal scheduled results are the agent's
    responsibility — it decides whether to notify via `notify.send`.

    Uses executor.ctx.http_client for HTTP calls. Logs errors but does not
    raise — scheduled task results should not crash the scheduler.
    """
    from src.config import CONFIG

    message = f"{title}\n\n{content}" if title else content

    if (
        CONFIG.signal_service
        and CONFIG.signal_bot_phone
        and CONFIG.signal_operator_phone
    ):
        from src.utils.text import _chunk

        send_url = f"http://{CONFIG.signal_service}/v2/send"

        for chunk in _chunk(message):
            try:
                resp = await executor.ctx.http_client.post(
                    send_url,
                    json={
                        "message": chunk,
                        "number": CONFIG.signal_bot_phone,
                        "recipients": [CONFIG.signal_operator_phone],
                    },
                )
            except Exception:
                logger.exception("Failed to send scheduled notification via Signal")
                return
            if resp.status_code >= 400:
                logger.error(
                    "Scheduled Signal notification failed: HTTP %d — %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return
    else:
        # Discord fallback — read webhook from secrets, truncate at 2000 chars
        sm = executor.secret_manager
        webhook_url = sm.get_secret("DISCORD_WEBHOOK_URL") if sm else None
        if not webhook_url:
            logger.warning(
                "No default notification channel configured — scheduler result not delivered"
            )
            return
        try:
            payload = (
                {"embeds": [{"title": title[:256], "description": content[:2000]}]}
                if title
                else {"content": message[:2000]}
            )
            resp = await executor.ctx.http_client.post(webhook_url, json=payload)
            if resp.status_code >= 400:
                logger.error(
                    "Scheduled Discord notification failed: HTTP %d — %s",
                    resp.status_code,
                    resp.text[:200],
                )
        except Exception:
            logger.exception("Failed to send scheduled notification via Discord")


def _build_discord_bot(executor: ToolExecutor, agent: Agent):
    """Return a DiscordBot if DISCORD_BOT_TOKEN is configured, else None."""
    sm = executor.secret_manager
    if sm is None:
        return None
    from src.discord_bot.bot import DISCORD_TOKEN_SECRET, DiscordBot

    token = sm.get_secret(DISCORD_TOKEN_SECRET)
    if not token:
        logger.info(
            "No %s secret configured — Discord bot not starting.",
            DISCORD_TOKEN_SECRET,
        )
        return None
    return DiscordBot(
        token=token,
        channel_name=CONFIG.discord_sessions_channel,
        agent=agent,
        executor=executor,
    )


def _build_signal_bot(executor: ToolExecutor, agent: Agent):
    """Return a SignalBot if Signal is fully configured, else None."""
    if not CONFIG.signal_service or not CONFIG.signal_bot_phone:
        logger.info("Signal not configured — Signal bot not starting.")
        return None
    if not CONFIG.signal_operator_phone:
        logger.warning(
            "SIGNAL_OPERATOR_PHONE not set — Signal bot not starting. "
            "The bot would poll destructively but ignore all messages."
        )
        return None
    from src.signal_bot.bot import SignalBot

    return SignalBot(
        signal_service=CONFIG.signal_service,
        bot_phone=CONFIG.signal_bot_phone,
        operator_phone=CONFIG.signal_operator_phone,
        agent=agent,
        executor=executor,
    )


async def _load_system_prompt(
    tool_context: ToolContext, executor: ToolExecutor
) -> str:
    """Load self-note, operator-note, skills, and tool docs into the system prompt."""
    self_doc = ""
    operator_doc = ""
    skills = "No skills installed yet."

    memory = tool_context._store.memory if tool_context._store else None

    # load self entry
    if memory is not None:
        try:
            entries = await memory.get_by_scope("self", "self")
            if entries:
                self_doc = entries[0].get("content", "")
            if self_doc:
                logger.info("Loaded self memory entry (%d chars)", len(self_doc))
            else:
                logger.info("No self memory entry found, using defaults")
        except Exception:
            logger.warning("Failed to load self memory entry", exc_info=True)

    # load operator entries
    if memory is not None:
        try:
            entries = await memory.get_by_scope("operator", "operator")
            if entries:
                operator_doc = "\n".join(e.get("content", "") for e in entries)
            if operator_doc:
                logger.info("Loaded operator memory (%d entries, %d chars)", len(entries), len(operator_doc))
            else:
                logger.info("No operator memory entries found, using defaults")
        except Exception:
            logger.warning("Failed to load operator memory entries", exc_info=True)

    # load skills list
    if memory is not None:
        try:
            skill_entries = await memory.list_skills()
            if skill_entries:
                skill_lines = []
                for s in skill_entries:
                    name = s.get("name", "")
                    preview = s.get("preview", "")
                    skill_lines.append(f"- **{name}**: {preview}")
                skills = "\n".join(skill_lines)
                logger.info("Loaded %d skills", len(skill_lines))
        except Exception:
            logger.info("Failed to load skills, using defaults")

    # tool documentation — built-in tools get full docs; MCP tools get a
    # compact catalog (one line per tool). The agent fetches full MCP tool
    # params on demand via system.get_tool_docs.
    tool_docs = TOOL_REGISTRY.generate_builtin_tool_documentation()
    mcp_catalog = TOOL_REGISTRY.generate_mcp_tool_catalog()

    # collect available secret names for the system prompt
    secret_names: list[str] = []
    if executor.secret_manager:
        secret_names = executor.secret_manager.list_secret_names()

    # build compact custom tools list for the system prompt
    custom_tools_list = ""
    if executor.custom_tool_manager is not None:
        approved = executor.custom_tool_manager.get_approved_tools()
        if approved:
            lines = []
            for t in approved:
                lines.append(f"- **{t.name}**: {t.description}")
            custom_tools_list = "\n".join(lines)

    return build_system_prompt(
        self_doc=self_doc,
        operator_doc=operator_doc,
        skills=skills,
        tool_docs=tool_docs,
        secret_names=secret_names,
        custom_tools_list=custom_tools_list,
        mcp_tool_catalog=mcp_catalog,
    )


async def _init_memory(executor: ToolExecutor, agent: Agent) -> None:
    """Initialize memory services after executor.initialize().

    Runs migration, creates RecallService, and wires recall into the agent.
    Must be called before the first system prompt load (which reads from
    memory_entries).
    """
    # 1. Run migration (needs embedding client for backfill)
    try:
        from src.memory.migration import migrate_documents_to_memory, backfill_embeddings

        migrated = await migrate_documents_to_memory(
            executor.store, executor.ctx._embedding_client
        )
        if migrated > 0:
            logger.info("Migrated %d documents to memory_entries", migrated)

        # Backfill embeddings for entries that were migrated without Ollama
        backfilled = await backfill_embeddings(
            executor.store, executor.ctx._embedding_client
        )
        if backfilled > 0:
            logger.info("Backfilled %d embeddings", backfilled)
    except Exception:
        logger.warning("Memory migration failed", exc_info=True)

    # 2. Create recall service wrapping the existing SearchEngine
    if executor.ctx._search_engine is not None:
        from src.memory.recall import RecallService

        recall_service = RecallService(search_engine=executor.ctx.search_engine)
        agent.memory_recall = recall_service.recall


@click.group()
def cli():
    pass


@cli.command()
@shared_options
def main(
    model_api: Literal["anthropic", "openai", "openapi", "umans"] | None,
    model_name: str | None,
    model_api_key: str | None,
    model_endpoint: str | None,
):
    executor, agent = build_services(
        model_api=model_api,
        model_name=model_name,
        model_api_key=model_api_key,
        model_endpoint=model_endpoint,
    )

    async def run():
        try:
            await executor.initialize()
            _wire_scheduler(executor, agent)
            await _init_memory(executor, agent)

            async def _refresh_prompt() -> str:
                return await _load_system_prompt(executor.ctx, executor)

            agent.system_prompt_provider = _refresh_prompt
            agent.system_prompt = await _refresh_prompt()

            discord_bot = _build_discord_bot(executor, agent)
            signal_bot = _build_signal_bot(executor, agent)

            async with asyncio.TaskGroup() as tg:
                if executor.scheduler:
                    tg.create_task(executor.scheduler.run())
                if discord_bot is not None:
                    tg.create_task(discord_bot.start())
                if signal_bot is not None:
                    tg.create_task(signal_bot.start())
        finally:
            await agent.aclose()
            await executor.aclose()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("received keyboard interrupt")


@cli.command(name="chat")
@shared_options
def chat(
    model_api: Literal["anthropic", "openai", "openapi", "umans"] | None,
    model_name: str | None,
    model_api_key: str | None,
    model_endpoint: str | None,
):
    executor, agent = build_services(
        model_api=model_api,
        model_name=model_name,
        model_api_key=model_api_key,
        model_endpoint=model_endpoint,
    )

    async def run():
        try:
            await executor.initialize()
            await _init_memory(executor, agent)

            async def _refresh_prompt() -> str:
                return await _load_system_prompt(executor.ctx, executor)

            agent.system_prompt_provider = _refresh_prompt
            agent.system_prompt = await _refresh_prompt()
            logger.info("Services initialized. Starting interactive chat.")
            print("\nAgent ready. Type your message (Ctrl+C to exit).\n")

            conversation: list[dict[str, Any]] = []
            while True:
                try:
                    user_input = input("You: ")
                except EOFError:
                    break

                if not user_input.strip():
                    continue

                logger.info("User: %s", user_input)
                response = await agent.chat(user_input, conversation=conversation)
                print(f"\nAgent: {response}\n")
        finally:
            await agent.aclose()
            await executor.aclose()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nExiting.")


@cli.command(name="serve")
@shared_options
def serve(
    model_api: Literal["anthropic", "openai", "openapi", "umans"] | None,
    model_name: str | None,
    model_api_key: str | None,
    model_endpoint: str | None,
):
    executor, agent = build_services(
        model_api=model_api,
        model_name=model_name,
        model_api_key=model_api_key,
        model_endpoint=model_endpoint,
    )

    async def run():
        try:
            await executor.initialize()
            _wire_scheduler(executor, agent)
            await _init_memory(executor, agent)

            async def _refresh_prompt() -> str:
                return await _load_system_prompt(executor.ctx, executor)

            agent.system_prompt_provider = _refresh_prompt
            agent.system_prompt = await _refresh_prompt()

            from src.agent.conversation import ConversationManager

            conversation_manager = ConversationManager(
                ctx=executor.ctx, llm_client=executor.llm_client
            )

            discord_bot = _build_discord_bot(executor, agent)
            signal_bot = _build_signal_bot(executor, agent)

            from src.web.app import create_app

            import uvicorn

            async with asyncio.TaskGroup() as tg:
                if executor.scheduler:
                    tg.create_task(executor.scheduler.run())
                if discord_bot is not None:
                    tg.create_task(discord_bot.start())
                if signal_bot is not None:
                    tg.create_task(signal_bot.start())
                config = uvicorn.Config(
                    create_app(agent, executor, conversation_manager),
                    host=CONFIG.web_host,
                    port=CONFIG.web_port,
                    log_level="info",
                )
                server = uvicorn.Server(config)

                async def _serve_and_stop():
                    """Run uvicorn, then stop the scheduler/Discord bot on exit."""
                    try:
                        await server.serve()
                    finally:
                        if executor.scheduler:
                            executor.scheduler.stop()
                        if discord_bot is not None:
                            await discord_bot.close()
                        if signal_bot is not None:
                            await signal_bot.close()

                tg.create_task(_serve_and_stop())
        finally:
            await agent.aclose()
            await executor.aclose()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("received keyboard interrupt")


if __name__ == "__main__":
    cli()

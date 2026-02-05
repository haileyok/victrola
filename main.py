import asyncio
import logging
from collections.abc import Callable
from typing import Literal

import click
import httpx

from src.agent.agent import Agent
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
    model_api: Literal["anthropic", "openai", "openapi"] | None,
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

    sub_api_key = CONFIG.sub_model_api_key or CONFIG.model_api_key
    llm_client = None
    if sub_api_key:
        llm_client = SubAgentLLM(
            api=CONFIG.sub_model_api,
            model=CONFIG.sub_model_name,
            api_key=sub_api_key,
            endpoint=CONFIG.sub_model_endpoint or None,
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
        model_api=model_api or CONFIG.model_api,
        model_name=model_name or CONFIG.model_name,
        model_api_key=model_api_key or CONFIG.model_api_key,
        model_endpoint=model_endpoint or CONFIG.model_endpoint or None,
        tool_executor=executor,
        sub_llm_client=llm_client,
    )

    return executor, agent


def _wire_scheduler(executor: ToolExecutor, agent: Agent) -> None:
    """Wire the scheduler callback to the agent after initialization."""
    if executor._scheduler is not None:

        async def on_schedule_fire(task_name: str, prompt: str) -> str:
            return await agent.chat(f"[Scheduled task: {task_name}] {prompt}")

        executor._scheduler._on_fire = on_schedule_fire


def _build_discord_bot(executor: ToolExecutor, agent: Agent):
    """Return a DiscordBot if DISCORD_BOT_TOKEN is configured, else None."""
    sm = getattr(executor, "_secret_manager", None)
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


async def _load_system_prompt(tool_context: ToolContext, executor: ToolExecutor) -> str:
    """Load self-note, skills, and tool docs into the system prompt."""
    self_doc = ""
    skills = "No skills installed yet."

    docs = tool_context._store.documents if tool_context._store else None

    # load self-note
    if docs is not None:
        try:
            doc = await docs.get("self")
            self_doc = doc.get("content", "")
            if self_doc:
                logger.info("Loaded self-note (%d chars)", len(self_doc))
            else:
                logger.info("Self-note exists but is empty")
        except Exception as e:
            if "not found" in str(e).lower():
                logger.info("No self-note found, using defaults")
            else:
                logger.warning("Failed to load self-note: %s", e)

    # load skills list
    if docs is not None:
        try:
            resp = await docs.list(limit=100)
            skill_lines = []
            for doc in resp.get("documents", []):
                rkey = doc.get("rkey", "")
                if rkey.startswith("skill:"):
                    name = rkey[6:]
                    content = doc.get("content", "")
                    preview = content[:80].replace("\n", " ")
                    skill_lines.append(f"- **{name}**: {preview}")
            if skill_lines:
                skills = "\n".join(skill_lines)
                logger.info("Loaded %d skills", len(skill_lines))
        except Exception:
            logger.info("Failed to load skills, using defaults")

    # tool documentation
    tool_docs = TOOL_REGISTRY.generate_tool_documentation()

    # collect available secret names for the system prompt
    secret_names: list[str] = []
    if executor._secret_manager:
        secret_names = executor._secret_manager.list_secret_names()

    # build compact custom tools list for the system prompt
    custom_tools_list = ""
    if executor._custom_tool_manager is not None:
        approved = executor._custom_tool_manager.get_approved_tools()
        if approved:
            lines = []
            for t in approved:
                lines.append(f"- **{t.name}**: {t.description}")
            custom_tools_list = "\n".join(lines)

    return build_system_prompt(
        self_doc=self_doc,
        skills=skills,
        tool_docs=tool_docs,
        secret_names=secret_names,
        custom_tools_list=custom_tools_list,
    )


@click.group()
def cli():
    pass


@cli.command()
@shared_options
def main(
    model_api: Literal["anthropic", "openai", "openapi"] | None,
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
        await executor.initialize()
        _wire_scheduler(executor, agent)

        async def _refresh_prompt() -> str:
            return await _load_system_prompt(executor._ctx, executor)

        agent._system_prompt_provider = _refresh_prompt
        agent._system_prompt = await _refresh_prompt()

        discord_bot = _build_discord_bot(executor, agent)

        async with asyncio.TaskGroup() as tg:
            if executor._scheduler:
                tg.create_task(executor._scheduler.run())
            if discord_bot is not None:
                tg.create_task(discord_bot.start())

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("received keyboard interrupt")


@cli.command(name="chat")
@shared_options
def chat(
    model_api: Literal["anthropic", "openai", "openapi"] | None,
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
        await executor.initialize()

        async def _refresh_prompt() -> str:
            return await _load_system_prompt(executor._ctx, executor)

        agent._system_prompt_provider = _refresh_prompt
        agent._system_prompt = await _refresh_prompt()
        logger.info("Services initialized. Starting interactive chat.")
        print("\nAgent ready. Type your message (Ctrl+C to exit).\n")

        while True:
            try:
                user_input = input("You: ")
            except EOFError:
                break

            if not user_input.strip():
                continue

            logger.info("User: %s", user_input)
            response = await agent.chat(user_input)
            print(f"\nAgent: {response}\n")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nExiting.")


@cli.command(name="tui")
@shared_options
def tui(
    model_api: Literal["anthropic", "openai", "openapi"] | None,
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

    from src.tui import VictrolaApp

    app = VictrolaApp(
        agent=agent,
        executor=executor,
    )
    app.run()


if __name__ == "__main__":
    cli()

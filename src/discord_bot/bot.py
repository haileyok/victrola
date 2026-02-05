"""Discord bot giving the operator a chat surface alongside the TUI.

Each thread inside the configured `DISCORD_SESSIONS_CHANNEL` is a chat session.
- Operator posts at top level of the channel → bot creates a thread from that
  message; first message = start of the session.
- Operator creates a thread themselves → first message = start of the session.

Only the agent's final text response is posted to the thread. Tool activity
(code the agent writes, tool results) is intentionally hidden from Discord —
review those in the TUI if you need the full trace.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import TYPE_CHECKING, Any

import discord

from src.config import CONFIG

if TYPE_CHECKING:
    from src.agent.agent import Agent, AgentEvent
    from src.tools.executor import ToolExecutor


# Per-image cap. Discord allows up to 10MB on free; we match that. Very
# large images will be quietly skipped with a log line rather than crash
# the agent call.
MAX_IMAGE_BYTES = 10 * 1024 * 1024


async def _extract_images(
    message: discord.Message,
) -> list[dict[str, str]]:
    """Pull image attachments off a Discord message as base64 blobs.

    Non-image attachments and oversized images are ignored (logged).
    Returned dicts are shaped for Agent.chat(..., images=...).
    """
    images: list[dict[str, str]] = []
    for att in message.attachments:
        content_type = (att.content_type or "").lower()
        if not content_type.startswith("image/"):
            continue
        if att.size and att.size > MAX_IMAGE_BYTES:
            logger.warning(
                "Skipping oversized image %s (%d bytes, max %d)",
                att.filename,
                att.size,
                MAX_IMAGE_BYTES,
            )
            continue
        try:
            data = await att.read()
        except Exception:
            logger.exception("Failed to read attachment %s", att.filename)
            continue
        images.append(
            {
                "media_type": content_type,
                "data": base64.b64encode(data).decode("ascii"),
            }
        )
    return images

logger = logging.getLogger(__name__)

# Discord caps content at 2000 chars. Leave headroom for code fences / markup.
MAX_CHUNK = 1900
DISCORD_TOKEN_SECRET = "DISCORD_BOT_TOKEN"


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


class _UsageTracker:
    """Accumulates per-chat() usage from agent events so we can show a footer."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read = 0
        self.cache_creation = 0
        self.calls = 0
        self.last_ctx: int | None = None
        self.last_tps: float | None = None
        self._llm_start: float | None = None

    async def on_event(self, event: "AgentEvent") -> None:
        match event.kind:
            case "llm_start":
                self._llm_start = time.monotonic()
            case "llm_done":
                elapsed = (
                    time.monotonic() - self._llm_start
                    if self._llm_start is not None
                    else None
                )
                self._llm_start = None
                usage = event.data.get("usage") or {}
                inp = usage.get("input_tokens", 0)
                out = usage.get("output_tokens", 0)
                cr = usage.get("cache_read_input_tokens", 0)
                cc = usage.get("cache_creation_input_tokens", 0)
                self.input_tokens += inp
                self.output_tokens += out
                self.cache_read += cr
                self.cache_creation += cc
                self.calls += 1
                self.last_ctx = inp + cr + cc
                if elapsed and elapsed > 0 and out > 0:
                    self.last_tps = out / elapsed
            case _:
                pass

    def format_footer(self) -> str:
        limit = getattr(CONFIG, "context_limit", 200_000) or 200_000
        parts: list[str] = []
        if self.last_ctx is not None:
            pct = f" ({self.last_ctx / limit:.0%})" if limit else ""
            parts.append(
                f"ctx {_fmt_tokens(self.last_ctx)}/{_fmt_tokens(limit)}{pct}"
            )
        if self.last_tps is not None:
            parts.append(f"tps {self.last_tps:.1f}")
        if self.input_tokens:
            parts.append(f"in {_fmt_tokens(self.input_tokens)}")
        if self.output_tokens:
            parts.append(f"out {_fmt_tokens(self.output_tokens)}")
        parts.append(f"calls {self.calls}")
        return " · ".join(parts)


def _chunk(text: str, limit: int = MAX_CHUNK) -> list[str]:
    """Split text into <=limit-char chunks, breaking on newlines when possible."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:  # no good break point — hard split
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


async def _send_chunked(
    thread: discord.Thread, text: str, *, prefix: str = ""
) -> None:
    """Post `text` to `thread`, splitting across multiple messages if needed.

    On the first failed chunk, stop and post a visible failure marker so the
    operator knows the response is truncated — partial silent posts look like
    the agent just stopped mid-thought.
    """
    if not text:
        return
    chunks = _chunk(text)
    for i, chunk in enumerate(chunks):
        content = f"{prefix}{chunk}" if i == 0 and prefix else chunk
        try:
            await thread.send(content)
        except Exception:
            logger.exception(
                "Failed to send chunk %d/%d to thread %s", i + 1, len(chunks), thread.id
            )
            # try to tell the operator what happened; if even that fails,
            # give up silently (the log still has the full trace).
            try:
                await thread.send(
                    f"⚠️ failed to post the rest of this response "
                    f"({len(chunks) - i} chunk(s) dropped — see `victrola.log`)"
                )
            except Exception:
                pass
            return


class DiscordBot:
    """Long-running Discord bot wrapping discord.py.

    Start with `await bot.start()` as a background asyncio task.
    """

    def __init__(
        self,
        token: str,
        channel_name: str,
        agent: "Agent",
        executor: "ToolExecutor",
    ) -> None:
        self._token = token
        self._channel_name = channel_name
        self._agent = agent
        self._executor = executor
        # serialize all agent.chat() calls — agent._conversation is shared state
        self._chat_lock = asyncio.Lock()

        intents = discord.Intents.default()
        intents.message_content = True  # privileged; must also be enabled on Dev Portal
        self._client = discord.Client(intents=intents)
        self._client.event(self.on_ready)
        self._client.event(self.on_message)

    async def start(self) -> None:
        """Run the bot forever (or until .close() is called)."""
        try:
            await self._client.start(self._token)
        except Exception:
            logger.exception("Discord bot crashed")

    async def close(self) -> None:
        await self._client.close()

    async def on_ready(self) -> None:
        logger.info(
            "Discord bot logged in as %s (watching #%s for sessions)",
            self._client.user,
            self._channel_name,
        )

    async def on_message(self, message: discord.Message) -> None:
        # Ignore self and other bots
        if message.author == self._client.user or message.author.bot:
            return

        thread = await self._resolve_thread(message)
        if thread is None:
            return  # not in our sessions channel

        async with self._chat_lock:
            try:
                await self._handle_message(message, thread)
            except Exception as e:
                logger.exception("Discord message handling failed")
                try:
                    await thread.send(f"⚠️ Error: `{type(e).__name__}`: {e}")
                except Exception:
                    pass

    async def _resolve_thread(
        self, message: discord.Message
    ) -> discord.Thread | None:
        """Return the thread this message belongs to, creating one if the
        message is a top-level post in the sessions channel.

        Returns None if the message is unrelated (different channel, etc.).
        """
        channel = message.channel

        # Case A: message inside a thread whose parent is our sessions channel
        if isinstance(channel, discord.Thread):
            parent = channel.parent
            if parent is not None and parent.name == self._channel_name:
                return channel
            return None

        # Case B: top-level message in the sessions channel
        if (
            isinstance(channel, (discord.TextChannel, discord.abc.GuildChannel))
            and getattr(channel, "name", None) == self._channel_name
        ):
            thread_name = (message.content or "chat")[:80] or "chat"
            try:
                thread = await message.create_thread(name=thread_name)
                logger.info(
                    "Created thread %s (id=%s) from top-level message", thread_name, thread.id
                )
                return thread
            except Exception:
                logger.exception("Failed to create thread from message")
                return None

        return None

    async def _handle_message(
        self, message: discord.Message, thread: discord.Thread
    ) -> None:
        """Route an operator message into a session and run the agent."""
        store = self._executor._ctx._store
        if store is None or store.chat is None:
            await thread.send("⚠️ Local store not initialized.")
            return

        thread_id = str(thread.id)
        # Ensure session exists for this thread. Title starts empty so the
        # auto-title generator can fill it in after a few turns.
        await store.chat.ensure_session(rkey=thread_id, title="")

        user_text = message.content or ""

        # pull any image attachments off the message (ephemeral — not persisted)
        images = await _extract_images(message)

        # nothing to do if the operator sent neither text nor an image
        if not user_text and not images:
            return

        # save user message. only the text is persisted; images are ephemeral
        # to the current turn by design (see Agent.chat docstring).
        await store.chat.create_message(
            session_id=thread_id,
            sender="user",
            content=json.dumps(
                {"role": "user", "content": user_text}, default=str
            ),
        )

        # load the full session conversation for agent context. agent.chat()
        # swaps + restores `_conversation` atomically under its own lock, so
        # we don't need manual snapshot/restore logic here.
        loaded = await self._load_conversation(thread_id)

        tracker = _UsageTracker()
        async with thread.typing():
            response = await self._agent.chat(
                user_text,
                on_event=tracker.on_event,
                conversation_override=loaded,
                images=images or None,
            )

        # save assistant response
        if response:
            await store.chat.create_message(
                session_id=thread_id,
                sender="assistant",
                content=json.dumps(
                    {"role": "assistant", "content": response}, default=str
                ),
            )
            # response body + a small stats footer as a separate message
            await _send_chunked(thread, response)
            footer = tracker.format_footer()
            if footer:
                try:
                    await thread.send(f"-# {footer}")
                except Exception:
                    logger.exception("Failed to send stats footer")
        else:
            await thread.send("(empty response)")

        # auto-title the session (and rename the Discord thread to match)
        try:
            from src.agent.conversation import maybe_generate_session_title

            title = await maybe_generate_session_title(
                store, thread_id, self._executor._ctx._llm_client
            )
            if title:
                try:
                    await thread.edit(name=title[:100])
                except Exception:
                    logger.exception("Failed to rename Discord thread %s", thread.id)
        except Exception:
            logger.exception("Auto-title generation failed for %s", thread.id)

    async def _load_conversation(self, thread_id: str) -> list[dict[str, Any]]:
        """Load a session's full message history in agent.chat() format."""
        store = self._executor._ctx._store
        assert store is not None and store.chat is not None

        messages: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            data = await store.chat.list_messages(
                session_id=thread_id, limit=100, cursor=cursor
            )
            records = data.get("messages", [])
            if not records:
                break
            for record in records:
                content_str = record.get("content", "{}")
                try:
                    msg = json.loads(content_str)
                    messages.append(msg)
                except (json.JSONDecodeError, TypeError):
                    messages.append({"role": "user", "content": content_str})
            cursor = data.get("cursor")
            if not cursor:
                break
        # Drop the most recent user message — agent.chat() will re-append it.
        if messages and messages[-1].get("role") == "user":
            messages = messages[:-1]
        return messages


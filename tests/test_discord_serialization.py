"""Tests for Discord bot per-thread message serialization."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import CONFIG
from src.store.store import Store
from src.tools.registry import ToolContext


@pytest.mark.asyncio
async def test_discord_serializes_same_thread_messages(tmp_path: Path):
    """Concurrent messages in the same Discord thread must be serialized.

    Without a per-thread lock, two rapid messages interleave load/save/
    agent.chat and corrupt the conversation history.
    """
    from src.discord_bot.bot import DiscordBot

    store = Store(path=tmp_path / "test.db")
    await store.initialize()

    ctx = ToolContext(store=store)
    executor = MagicMock()
    executor.store = store
    executor.ctx = ctx
    executor.llm_client = None

    call_log: list[tuple[str, str]] = []

    class SlowAgent:
        async def chat(self, user_message, conversation=None, on_event=None,
                       on_compact=None, images=None):
            call_log.append(("start", user_message))
            await asyncio.sleep(0.2)
            call_log.append(("end", user_message))
            return f"reply to {user_message}"

    with patch.object(CONFIG, "discord_allowed_user_ids", ""):
        bot = DiscordBot(
            token="fake", channel_name="test",
            agent=SlowAgent(), executor=executor,
        )

    # Mock Discord thread
    thread = MagicMock()
    thread.id = 12345
    thread.send = AsyncMock()
    thread.edit = AsyncMock()

    class _FakeTyping:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    thread.typing = MagicMock(return_value=_FakeTyping())

    # Bypass thread resolution
    bot._resolve_thread = AsyncMock(return_value=thread)
    bot._client = MagicMock()
    bot._client.user = MagicMock()
    bot._client.user.id = 0

    def make_msg(content):
        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.bot = False
        msg.author.id = 999
        msg.content = content
        msg.attachments = []
        return msg

    msg1 = make_msg("first")
    msg2 = make_msg("second")

    # Fire both concurrently
    await asyncio.gather(
        bot.on_message(msg1),
        bot.on_message(msg2),
    )

    # Assert serialization: the first call must finish before the second starts.
    # On current code (no lock), both start immediately and interleave.
    starts = [i for i, (event, _) in enumerate(call_log) if event == "start"]
    ends = [i for i, (event, _) in enumerate(call_log) if event == "end"]
    assert len(starts) == 2 and len(ends) == 2
    assert ends[0] < starts[1], (
        "second message started before first completed — messages not serialized"
    )

    await store.close()

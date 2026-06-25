"""Tests for PR 11: Add timeout to Discord agent.chat() calls."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import CONFIG


def test_config_has_chat_timeout():
    """Config should have discord_chat_timeout_seconds field."""
    assert hasattr(CONFIG, "discord_chat_timeout_seconds")
    assert CONFIG.discord_chat_timeout_seconds == 300


@pytest.mark.asyncio
async def test_timeout_wraps_agent_chat():
    """_handle_message should wrap agent.chat() in asyncio.wait_for."""
    from src.discord_bot.bot import DiscordBot

    with patch.object(CONFIG, "discord_allowed_user_ids", ""):
        bot = DiscordBot(
            token="fake",
            channel_name="test",
            agent=MagicMock(),
            executor=MagicMock(),
        )

    # Mock the executor store and conversation loading
    store = MagicMock()
    store.chat = MagicMock()
    store.chat.ensure_session = AsyncMock()
    store.chat.create_message = AsyncMock()
    bot._executor._ctx._store = store
    bot._load_conversation = AsyncMock(return_value=[])

    # Mock agent.chat to return quickly
    bot._agent.chat = AsyncMock(return_value="test response")

    # Mock the message and thread
    message = MagicMock()
    message.content = "hello"
    message.attachments = []
    thread = MagicMock()
    thread.id = 123
    thread.typing = MagicMock(return_value=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()))
    thread.send = AsyncMock()

    with patch.object(CONFIG, "discord_chat_timeout_seconds", 1), \
         patch("src.discord_bot.bot._extract_images", new_callable=AsyncMock, return_value=[]):
        await bot._handle_message(message, thread)

    bot._agent.chat.assert_called_once()


@pytest.mark.asyncio
async def test_timeout_posts_error_on_timeout():
    """On timeout, an error message should be posted to the thread."""
    from src.discord_bot.bot import DiscordBot

    with patch.object(CONFIG, "discord_allowed_user_ids", ""):
        bot = DiscordBot(
            token="fake",
            channel_name="test",
            agent=MagicMock(),
            executor=MagicMock(),
        )

    store = MagicMock()
    store.chat = MagicMock()
    store.chat.ensure_session = AsyncMock()
    store.chat.create_message = AsyncMock()
    bot._executor._ctx._store = store
    bot._load_conversation = AsyncMock(return_value=[])

    # Mock _extract_images to return empty list
    with patch("src.discord_bot.bot._extract_images", new_callable=AsyncMock, return_value=[]):
        # Mock agent.chat to sleep forever
        async def slow_chat(*args, **kwargs):
            await asyncio.sleep(100)

        bot._agent.chat = slow_chat

        message = MagicMock()
        message.content = "hello"
        message.attachments = []
        thread = MagicMock()
        thread.id = 123
        thread.typing = MagicMock(return_value=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()))
        thread.send = AsyncMock()

        with patch.object(CONFIG, "discord_chat_timeout_seconds", 0.1):
            await bot._handle_message(message, thread)

    # Verify timeout error was posted
    thread.send.assert_called()
    send_arg = thread.send.call_args[0][0]
    assert "timed out" in send_arg.lower()

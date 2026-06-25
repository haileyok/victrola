"""Tests for PR 10: Add Discord operator allowlist."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import CONFIG


def test_config_has_discord_allowed_user_ids():
    """Config should have discord_allowed_user_ids field defaulting to empty string."""
    assert hasattr(CONFIG, "discord_allowed_user_ids")
    assert CONFIG.discord_allowed_user_ids == ""


def test_config_has_discord_chat_timeout():
    """Config should have discord_chat_timeout_seconds field."""
    assert hasattr(CONFIG, "discord_chat_timeout_seconds")
    assert CONFIG.discord_chat_timeout_seconds == 300


def test_bot_parses_allowlist():
    """DiscordBot should parse comma-separated user IDs into a set."""
    from src.discord_bot.bot import DiscordBot

    with patch.object(CONFIG, "discord_allowed_user_ids", "123,456,789"):
        bot = DiscordBot(
            token="fake",
            channel_name="test",
            agent=MagicMock(),
            executor=MagicMock(),
        )
    assert bot._allowed_user_ids == {123, 456, 789}


def test_bot_allowlist_none_when_empty():
    """DiscordBot should set _allowed_user_ids to None when config is empty."""
    from src.discord_bot.bot import DiscordBot

    with patch.object(CONFIG, "discord_allowed_user_ids", ""):
        bot = DiscordBot(
            token="fake",
            channel_name="test",
            agent=MagicMock(),
            executor=MagicMock(),
        )
    assert bot._allowed_user_ids is None


def test_bot_has_no_chat_lock():
    """DiscordBot should not have _chat_lock attribute."""
    from src.discord_bot.bot import DiscordBot

    bot = DiscordBot(
        token="fake",
        channel_name="test",
        agent=MagicMock(),
        executor=MagicMock(),
    )
    assert not hasattr(bot, "_chat_lock")

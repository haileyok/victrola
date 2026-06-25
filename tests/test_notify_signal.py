"""Tests for notify.signal and notify.send tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.tools.definitions.notify import discord, signal, send
from src.tools.registry import ToolContext


def _mock_ctx(http_client=None, secret_manager=None):
    """Build a ToolContext with a mock http_client and optional secret_manager."""
    ctx = MagicMock(spec=ToolContext)
    if http_client is None:
        http_client = MagicMock()
        http_client.post = AsyncMock()
        http_client.get = AsyncMock()
    ctx.http_client = http_client
    ctx.secret_manager = secret_manager
    return ctx


def _make_response(status_code=200, text="OK", json_data=None):
    """Build a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


async def test_notify_signal_sends_message():
    """notify.signal sends to the correct Signal REST endpoint."""
    with patch("src.config.CONFIG") as mock_config:
        mock_config.signal_service = "127.0.0.1:8080"
        mock_config.signal_bot_phone = "+1111111111"
        mock_config.signal_operator_phone = "+2222222222"

        ctx = _mock_ctx()
        ctx.http_client.post.return_value = _make_response(200)

        result = await signal(ctx, "hello world")

        assert result["success"] is True
        assert result["chunks_sent"] == 1

        # Verify the URL and payload
        call_args = ctx.http_client.post.call_args
        assert "v2/send/+1111111111" in call_args.args[0]
        payload = call_args.kwargs["json"]
        assert payload["message"] == "hello world"
        assert payload["recipients"] == ["+2222222222"]


async def test_notify_signal_with_title():
    """notify.signal prepends title to the message."""
    with patch("src.config.CONFIG") as mock_config:
        mock_config.signal_service = "127.0.0.1:8080"
        mock_config.signal_bot_phone = "+1111111111"
        mock_config.signal_operator_phone = "+2222222222"

        ctx = _mock_ctx()
        ctx.http_client.post.return_value = _make_response(200)

        result = await signal(ctx, "body text", title="Alert")

        assert result["success"] is True
        call_args = ctx.http_client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["message"] == "Alert\n\nbody text"


async def test_notify_signal_not_configured():
    """notify.signal returns error when not configured."""
    with patch("src.config.CONFIG") as mock_config:
        mock_config.signal_service = ""
        mock_config.signal_bot_phone = ""
        mock_config.signal_operator_phone = ""

        ctx = _mock_ctx()
        result = await signal(ctx, "hello")

        assert "error" in result
        assert "not configured" in result["error"].lower()
        ctx.http_client.post.assert_not_called()


async def test_notify_signal_http_error():
    """notify.signal returns error on HTTP failure."""
    with patch("src.config.CONFIG") as mock_config:
        mock_config.signal_service = "127.0.0.1:8080"
        mock_config.signal_bot_phone = "+1111111111"
        mock_config.signal_operator_phone = "+2222222222"

        ctx = _mock_ctx()
        ctx.http_client.post.return_value = _make_response(500, text="Internal Error")

        result = await signal(ctx, "hello")

        assert "error" in result
        assert "500" in result["error"]


async def test_notify_signal_empty_content():
    """notify.signal rejects empty content."""
    with patch("src.config.CONFIG") as mock_config:
        mock_config.signal_service = "127.0.0.1:8080"
        mock_config.signal_bot_phone = "+1111111111"
        mock_config.signal_operator_phone = "+2222222222"

        ctx = _mock_ctx()
        result = await signal(ctx, "")

        assert "error" in result
        assert "content" in result["error"].lower()


async def test_notify_send_routes_to_signal():
    """notify.send routes to Signal when configured."""
    with patch("src.config.CONFIG") as mock_config:
        mock_config.signal_service = "127.0.0.1:8080"
        mock_config.signal_bot_phone = "+1111111111"
        mock_config.signal_operator_phone = "+2222222222"

        ctx = _mock_ctx()
        ctx.http_client.post.return_value = _make_response(200)

        result = await send(ctx, "test message")

        assert result["success"] is True
        assert "v2/send" in ctx.http_client.post.call_args.args[0]


async def test_notify_send_falls_back_to_discord():
    """notify.send routes to Discord when Signal is not configured."""
    with patch("src.config.CONFIG") as mock_config:
        mock_config.signal_service = ""
        mock_config.signal_bot_phone = ""
        mock_config.signal_operator_phone = ""

        sm = MagicMock()
        sm.get_secret.return_value = "https://discord.com/api/webhooks/123/abc"

        ctx = _mock_ctx(secret_manager=sm)
        ctx.http_client.post.return_value = _make_response(204)

        result = await send(ctx, "test message")

        assert result["success"] is True
        # Should have posted to the Discord webhook URL
        call_url = ctx.http_client.post.call_args.args[0]
        assert "discord.com" in call_url


async def test_notify_signal_chunks_long_messages():
    """Long messages are split into multiple chunks."""
    with patch("src.config.CONFIG") as mock_config:
        mock_config.signal_service = "127.0.0.1:8080"
        mock_config.signal_bot_phone = "+1111111111"
        mock_config.signal_operator_phone = "+2222222222"

        ctx = _mock_ctx()
        ctx.http_client.post.return_value = _make_response(200)

        # Create a message longer than the 1900 char chunk limit
        long_text = "x" * 5000
        result = await signal(ctx, long_text)

        assert result["success"] is True
        assert result["chunks_sent"] > 1

        # Verify each chunk was sent
        assert ctx.http_client.post.call_count == result["chunks_sent"]

"""Tests for PR 6: Add LLM API call retry with exponential backoff."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from anthropic import APIStatusError, APIConnectionError, APITimeoutError

from src.agent.agent import (
    _retry_with_backoff,
    _is_retryable_status,
    _is_anthropic_retryable,
    _is_httpx_retryable,
    AnthropicClient,
    OpenAICompatibleClient,
)


# --- _is_retryable_status tests ---

def test_is_retryable_429():
    assert _is_retryable_status(429) is True

def test_is_retryable_500():
    assert _is_retryable_status(500) is True

def test_is_retryable_503():
    assert _is_retryable_status(503) is True

def test_not_retryable_400():
    assert _is_retryable_status(400) is False

def test_not_retryable_404():
    assert _is_retryable_status(404) is False

def test_not_retryable_200():
    assert _is_retryable_status(200) is False


# --- _retry_with_backoff tests ---

@pytest.mark.asyncio
async def test_retry_succeeds_after_failures():
    """Should retry and succeed when transient failures occur."""
    call_count = 0

    async def coro_factory():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("transient")
        return "success"

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await _retry_with_backoff(
            coro_factory,
            is_retryable_exc=lambda e: isinstance(e, ConnectionError),
            base_delay=0.01,
        )
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_propagates_after_max_retries():
    """Should propagate the exception after max retries."""
    call_count = 0

    async def coro_factory():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("persistent")

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(ConnectionError, match="persistent"):
            await _retry_with_backoff(
                coro_factory,
                is_retryable_exc=lambda e: isinstance(e, ConnectionError),
                max_retries=3,
                base_delay=0.01,
            )
    assert call_count == 4  # initial + 3 retries


@pytest.mark.asyncio
async def test_no_retry_on_non_retryable():
    """Should not retry non-retryable exceptions."""
    call_count = 0

    async def coro_factory():
        nonlocal call_count
        call_count += 1
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        await _retry_with_backoff(
            coro_factory,
            is_retryable_exc=lambda e: isinstance(e, ConnectionError),
            base_delay=0.01,
        )
    assert call_count == 1


# --- Anthropic client retry tests ---

@pytest.mark.asyncio
async def test_anthropic_retries_on_overloaded():
    """AnthropicClient.complete should retry on overloaded errors."""
    client = AnthropicClient(api_key="fake", model_name="test")

    call_count = 0
    original_msg = MagicMock()
    original_msg.content = []
    original_msg.usage = MagicMock(input_tokens=1, output_tokens=1)
    original_msg.stop_reason = "end_turn"

    async def fake_stream():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            # Simulate a 529 Overloaded error
            raise APIStatusError(
                message="Overloaded",
                response=MagicMock(status_code=529),
                body=None,
            )
        return original_msg

    with patch.object(client, "_client") as mock_client:
        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.get_final_message = fake_stream
        mock_client.messages.stream = MagicMock(return_value=mock_stream)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client.complete(messages=[], system="test")

    assert call_count == 3
    assert result.stop_reason == "end_turn"


# --- httpx retryable tests ---

def test_httpx_transport_error_retryable():
    exc = httpx.TransportError("connection lost")
    assert _is_httpx_retryable(exc) is True

def test_httpx_timeout_retryable():
    exc = httpx.TimeoutException("timed out")
    assert _is_httpx_retryable(exc) is True

def test_httpx_429_retryable():
    response = MagicMock()
    response.status_code = 429
    exc = httpx.HTTPStatusError("rate limited", request=MagicMock(), response=response)
    assert _is_httpx_retryable(exc) is True

def test_httpx_400_not_retryable():
    response = MagicMock()
    response.status_code = 400
    exc = httpx.HTTPStatusError("bad request", request=MagicMock(), response=response)
    assert _is_httpx_retryable(exc) is False

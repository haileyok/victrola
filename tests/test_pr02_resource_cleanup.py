"""Tests for PR 2: Resource cleanup (aclose on shutdown)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agent.agent import Agent, AnthropicClient, OpenAICompatibleClient


def test_anthropic_client_has_aclose():
    """AnthropicClient should have an async aclose method."""
    client = AnthropicClient(api_key="fake", model_name="test")
    assert hasattr(client, "aclose")
    import inspect
    assert inspect.iscoroutinefunction(client.aclose)


def test_openai_compatible_client_has_aclose():
    """OpenAICompatibleClient should have an async aclose method."""
    client = OpenAICompatibleClient(
        api_key="fake", model_name="test", endpoint="http://localhost:8080/v1"
    )
    assert hasattr(client, "aclose")
    import inspect
    assert inspect.iscoroutinefunction(client.aclose)


def test_agent_has_aclose():
    """Agent should have an async aclose method."""
    agent = Agent(model_api="anthropic", model_name="x", model_api_key="key")
    assert hasattr(agent, "aclose")
    import inspect
    assert inspect.iscoroutinefunction(agent.aclose)


@pytest.mark.asyncio
async def test_agent_aclose_delegates_to_client():
    """Agent.aclose() should call aclose() on its underlying client."""
    agent = Agent(model_api="anthropic", model_name="x", model_api_key="key")

    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()
    agent._client = mock_client

    await agent.aclose()

    mock_client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_anthropic_client_aclose_closes_underlying():
    """AnthropicClient.aclose() should close the AsyncAnthropic client."""
    client = AnthropicClient(api_key="fake", model_name="test")

    client._client = MagicMock()
    client._client.close = AsyncMock()

    await client.aclose()

    client._client.close.assert_called_once()


@pytest.mark.asyncio
async def test_openai_client_aclose_closes_http():
    """OpenAICompatibleClient.aclose() should close the httpx AsyncClient."""
    client = OpenAICompatibleClient(
        api_key="fake", model_name="test", endpoint="http://localhost:8080/v1"
    )

    client._http = MagicMock()
    client._http.aclose = AsyncMock()

    await client.aclose()

    client._http.aclose.assert_called_once()

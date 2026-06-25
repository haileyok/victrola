"""Tests for PR 1: Per-session conversation state."""

import pytest

from src.agent.agent import Agent, AgentResponse, AgentTextBlock
from unittest.mock import MagicMock


@pytest.mark.asyncio
async def test_conversations_are_independent():
    """Two conversation lists passed to chat() must remain independent."""
    agent = Agent(model_api="anthropic", model_name="x", model_api_key="key")

    # Stub the client to return a simple text response
    async def fake_complete(**kwargs):
        return AgentResponse(
            content=[AgentTextBlock(text="response")],
            stop_reason="end_turn",
            usage={"input_tokens": 1, "output_tokens": 1},
        )

    agent._client = MagicMock()
    agent._client.complete = fake_complete

    conv1: list = []
    conv2: list = []

    await agent.chat("hello from conv1", conversation=conv1)
    await agent.chat("hello from conv2", conversation=conv2)

    # Each conversation should have exactly 2 messages (user + assistant)
    assert len(conv1) == 2
    assert len(conv2) == 2

    # Verify they didn't cross-contaminate
    user_msg_1 = conv1[0]["content"]
    user_msg_2 = conv2[0]["content"]
    assert "conv1" in user_msg_1
    assert "conv2" in user_msg_2
    assert "conv2" not in user_msg_1
    assert "conv1" not in user_msg_2


@pytest.mark.asyncio
async def test_conversation_mutated_in_place():
    """chat() must mutate the caller's list in place (not rebind)."""
    agent = Agent(model_api="anthropic", model_name="x", model_api_key="key")

    async def fake_complete(**kwargs):
        return AgentResponse(
            content=[AgentTextBlock(text="ok")],
            stop_reason="end_turn",
            usage={"input_tokens": 1, "output_tokens": 1},
        )

    agent._client = MagicMock()
    agent._client.complete = fake_complete

    conv: list = []
    original_id = id(conv)
    await agent.chat("test", conversation=conv)

    # The same list object must have been mutated (not replaced)
    assert id(conv) == original_id
    assert len(conv) == 2  # user + assistant


def test_agent_has_no_conversation_attr():
    """Agent must not store _conversation or _chat_lock."""
    agent = Agent(model_api="anthropic", model_name="x", model_api_key="key")
    assert not hasattr(agent, "_conversation")
    assert not hasattr(agent, "_chat_lock")


def test_chat_requires_conversation_param():
    """chat() must require a conversation parameter."""
    import inspect
    sig = inspect.signature(Agent.chat)
    assert "conversation" in sig.parameters
    # conversation should NOT have a default (it's required)
    assert sig.parameters["conversation"].default is inspect.Parameter.empty

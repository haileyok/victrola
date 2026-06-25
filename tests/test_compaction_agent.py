"""Tests for agent on_compact callback and ConversationManager checkpoint loading."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.conversation import ConversationManager
from src.store.store import Store
from src.tools.registry import ToolContext


class _StubSubLLM:
    """Stub sub-LLM for compaction — returns a fixed summary."""

    def __init__(self, summary: str = "Summary of older messages."):
        self._summary = summary
        self.calls: list = []

    async def complete(self, prompt, system=None, max_tokens=None):
        self.calls.append({"prompt": prompt, "system": system})
        return self._summary


async def _make_agent_with_stub_client(tmp_path, sub_llm=None, compact_threshold=1000):
    """Build an Agent with a mock client that returns immediately."""
    from src.agent.agent import Agent, AgentTextBlock, AgentResponse

    store = Store(path=tmp_path / "test.db")
    await store.initialize()

    from src.tools.registry import ToolRegistry
    from src.tools.executor import ToolExecutor

    ctx = ToolContext(store=store)
    registry = ToolRegistry()
    executor = ToolExecutor(registry=registry, ctx=ctx)
    executor._tool_definition = None

    # Create a mock client that returns a simple text response
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(
        return_value=AgentResponse(
            content=[AgentTextBlock(text="stub response")],
            stop_reason="end_turn",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
    )
    mock_client.aclose = AsyncMock()

    # Build agent without going through __init__ (which creates a real client)
    agent = Agent.__new__(Agent)
    agent._client = mock_client
    agent._tool_executor = executor
    agent._max_iterations = 30
    agent._system_prompt = "test system prompt"
    agent._system_prompt_provider = None
    agent._sub_llm_client = sub_llm
    agent._compact_threshold_chars = compact_threshold
    agent._memory_recall = None

    return agent, store, mock_client


async def test_on_compact_callback_called(tmp_path):
    """When conversation exceeds threshold, on_compact is called with summary and split_idx."""
    sub_llm = _StubSubLLM("Compacted summary text.")
    agent, store, mock_client = await _make_agent_with_stub_client(
        tmp_path, sub_llm=sub_llm, compact_threshold=500
    )

    # Build a conversation that exceeds the threshold
    conversation = []
    for i in range(10):
        conversation.append(
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i} " * 50}
        )

    callback_calls = []

    async def on_compact(summary, split_idx):
        callback_calls.append((summary, split_idx))

    await agent.chat("final message", conversation=conversation, on_compact=on_compact)

    assert len(callback_calls) == 1
    summary, split_idx = callback_calls[0]
    assert summary == "Compacted summary text."
    assert split_idx > 0
    # split_idx is the count of messages summarized, from the original conversation
    assert split_idx < 10  # original conversation had 10 messages

    # Conversation should be compacted (fewer messages than original + new user msg)
    assert len(conversation) < 10 + 1

    await store.close()


async def test_on_compact_not_called_below_threshold(tmp_path):
    """Short conversation should not trigger compaction."""
    sub_llm = _StubSubLLM("Should not be called.")
    agent, store, mock_client = await _make_agent_with_stub_client(
        tmp_path, sub_llm=sub_llm, compact_threshold=10000
    )

    conversation = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]

    callback_calls = []

    async def on_compact(summary, split_idx):
        callback_calls.append((summary, split_idx))

    await agent.chat("test", conversation=conversation, on_compact=on_compact)

    assert len(callback_calls) == 0
    assert len(sub_llm.calls) == 0

    await store.close()


async def test_on_compact_optional(tmp_path):
    """chat() without on_compact should work fine."""
    sub_llm = _StubSubLLM("Summary.")
    agent, store, mock_client = await _make_agent_with_stub_client(
        tmp_path, sub_llm=sub_llm, compact_threshold=500
    )

    conversation = []
    for i in range(10):
        conversation.append(
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i} " * 50}
        )

    # Should not raise
    response = await agent.chat("test", conversation=conversation)
    assert response is not None

    await store.close()


async def test_on_compact_callback_failure_does_not_crash(tmp_path):
    """If the on_compact callback raises, compaction should still proceed."""
    sub_llm = _StubSubLLM("Summary despite callback failure.")
    agent, store, mock_client = await _make_agent_with_stub_client(
        tmp_path, sub_llm=sub_llm, compact_threshold=500
    )

    conversation = []
    for i in range(10):
        conversation.append(
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i} " * 50}
        )

    async def failing_callback(summary, split_idx):
        raise RuntimeError("Callback failed!")

    # Should not raise — the error is caught and logged
    response = await agent.chat("test", conversation=conversation, on_compact=failing_callback)
    assert response is not None

    # Conversation should still be compacted in memory
    assert len(conversation) < 10 + 1  # +1 for the appended user message

    await store.close()


async def test_load_session_with_ids_no_checkpoint(tmp_path):
    """load_session_with_ids returns messages and IDs with no compaction."""
    store = Store(path=tmp_path / "test.db")
    await store.initialize()

    session_id = "test-sess"
    await store.chat.ensure_session(rkey=session_id, title="test")

    # Seed messages
    ids = []
    for i in range(5):
        result = await store.chat.create_message(
            session_id=session_id,
            sender="user" if i % 2 == 0 else "assistant",
            content=json.dumps(
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            ),
        )
        ids.append(result["id"])

    ctx = ToolContext(store=store)
    conv_manager = ConversationManager(ctx=ctx, llm_client=None)
    messages, msg_ids = await conv_manager.load_session_with_ids(session_id)

    # Last message is user (index 4 is even → user), so it's dropped
    assert len(messages) == 4
    assert len(msg_ids) == 4
    assert msg_ids == ids[:4]

    await store.close()


async def test_load_session_with_ids_with_checkpoint(tmp_path):
    """load_session_with_ids prepends summary and skips compacted messages."""
    store = Store(path=tmp_path / "test.db")
    await store.initialize()

    session_id = "test-sess"
    await store.chat.ensure_session(rkey=session_id, title="test")

    # Seed 5 messages
    ids = []
    for i in range(5):
        result = await store.chat.create_message(
            session_id=session_id,
            sender="user" if i % 2 == 0 else "assistant",
            content=json.dumps(
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            ),
        )
        ids.append(result["id"])

    # Set checkpoint at message 3
    await store.chat.set_compaction_checkpoint(session_id, ids[2], "Summary of first 3")

    ctx = ToolContext(store=store)
    conv_manager = ConversationManager(ctx=ctx, llm_client=None)
    messages, msg_ids = await conv_manager.load_session_with_ids(session_id)

    # Should be: [summary_msg, msg3, msg4] → drop tail if user
    # ids[3] is assistant (index 3 is odd), ids[4] is user (index 4 is even)
    # So msg_ids[4] is user → dropped. Result: [summary, msg3]
    assert len(messages) == 2
    assert msg_ids[0] == -1  # synthetic summary
    assert msg_ids[1] == ids[3]  # first post-checkpoint message
    assert "Summary of first 3" in messages[0]["content"]

    await store.close()


async def test_load_session_delegates_to_with_ids(tmp_path):
    """load_session returns the same messages as load_session_with_ids."""
    store = Store(path=tmp_path / "test.db")
    await store.initialize()

    session_id = "test-sess"
    await store.chat.ensure_session(rkey=session_id, title="test")
    await store.chat.create_message(
        session_id=session_id,
        sender="user",
        content=json.dumps({"role": "user", "content": "hello"}),
    )
    await store.chat.create_message(
        session_id=session_id,
        sender="assistant",
        content=json.dumps({"role": "assistant", "content": "hi"}),
    )

    ctx = ToolContext(store=store)
    conv_manager = ConversationManager(ctx=ctx, llm_client=None)

    msgs_old = await conv_manager.load_session(session_id)
    msgs_new, _ = await conv_manager.load_session_with_ids(session_id)

    assert msgs_old == msgs_new

    await store.close()

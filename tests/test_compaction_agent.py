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


async def test_on_message_persists_assistant_turn(tmp_path):
    """on_message fires once with the assistant message on a no-tool turn."""
    agent, store, mock_client = await _make_agent_with_stub_client(tmp_path)

    received = []

    async def on_message(message):
        received.append(message)

    await agent.chat("hi", conversation=[], on_message=on_message)

    assert len(received) == 1
    assert received[0]["role"] == "assistant"
    assert received[0]["content"][0]["type"] == "text"
    assert received[0]["content"][0]["text"] == "stub response"

    await store.close()


async def test_on_message_persists_full_tool_turn(tmp_path):
    """on_message fires for the assistant tool_use turn, the tool_result turn,
    and the final assistant text turn — in order."""
    from src.agent.agent import AgentResponse, AgentTextBlock, AgentToolUseBlock

    agent, store, mock_client = await _make_agent_with_stub_client(tmp_path)

    mock_client.complete = AsyncMock(
        side_effect=[
            AgentResponse(
                content=[
                    AgentToolUseBlock(id="t1", name="execute_code", input={"code": "1"})
                ],
                stop_reason="tool_use",
                usage={},
            ),
            AgentResponse(
                content=[AgentTextBlock(text="done")],
                stop_reason="end_turn",
                usage={},
            ),
        ]
    )
    # Bypass the real Deno executor.
    agent._handle_tool_call = AsyncMock(return_value={"output": "ok"})

    received = []

    async def on_message(message):
        received.append(message)

    await agent.chat("run it", conversation=[], on_message=on_message)

    shape = [(m["role"], m["content"][0]["type"]) for m in received]
    assert shape == [
        ("assistant", "tool_use"),
        ("user", "tool_result"),
        ("assistant", "text"),
    ]

    await store.close()


async def test_on_message_optional(tmp_path):
    """chat() without on_message should work fine."""
    agent, store, mock_client = await _make_agent_with_stub_client(tmp_path)
    response = await agent.chat("test", conversation=[])
    assert response == "stub response"
    await store.close()


async def test_on_message_skips_empty_assistant_turn(tmp_path):
    """An empty assistant turn (no text/tool_use blocks) is not persisted —
    reloading one would send an empty content list and 400 the API."""
    from src.agent.agent import AgentResponse

    agent, store, mock_client = await _make_agent_with_stub_client(tmp_path)
    mock_client.complete = AsyncMock(
        return_value=AgentResponse(content=[], stop_reason="end_turn", usage={})
    )

    received = []

    async def on_message(message):
        received.append(message)

    response = await agent.chat("hi", conversation=[], on_message=on_message)
    assert response == ""
    assert received == []

    await store.close()


def test_persistable_message_strips_tool_result_images():
    """Image blocks in tool results are replaced with a placeholder for storage,
    without mutating the original (in-memory) message."""
    from src.agent.agent import _persistable_message

    msg = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": [
                    {"type": "text", "text": "here"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "A" * 5000,
                        },
                    },
                ],
            }
        ],
    }
    out = _persistable_message(msg)
    inner = out["content"][0]["content"]
    # Text is preserved; only the image block is replaced.
    assert isinstance(inner, list)
    assert {"type": "text", "text": "here"} in inner
    assert {"type": "text", "text": "[image omitted from history]"} in inner
    assert "A" * 5000 not in json.dumps(out)
    assert out["content"][0]["tool_use_id"] == "t1"
    # Original message is not mutated.
    assert any(b.get("type") == "image" for b in msg["content"][0]["content"])

    # String tool results pass through unchanged (same object).
    str_msg = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "ok"}],
    }
    assert _persistable_message(str_msg) is str_msg


async def test_on_message_persists_image_tool_result_without_base64(tmp_path):
    """A tool result carrying an image is persisted without the base64 blob."""
    from src.agent.agent import AgentResponse, AgentTextBlock, AgentToolUseBlock

    agent, store, mock_client = await _make_agent_with_stub_client(tmp_path)
    mock_client.complete = AsyncMock(
        side_effect=[
            AgentResponse(
                content=[
                    AgentToolUseBlock(id="t1", name="execute_code", input={"code": "1"})
                ],
                stop_reason="tool_use",
                usage={},
            ),
            AgentResponse(
                content=[AgentTextBlock(text="done")], stop_reason="end_turn", usage={}
            ),
        ]
    )
    agent._handle_tool_call = AsyncMock(
        return_value={
            "output": {
                "type": "image_result",
                "image": {"type": "base64", "media_type": "image/png", "data": "B" * 5000},
            }
        }
    )

    received = []

    async def on_message(message):
        received.append(message)

    await agent.chat("show", conversation=[], on_message=on_message)

    tool_result_turn = next(m for m in received if m["role"] == "user")
    blob = json.dumps(tool_result_turn)
    assert "B" * 5000 not in blob
    assert "image omitted" in blob

    await store.close()


async def test_empty_text_block_not_persisted_alongside_tool_use(tmp_path):
    """An empty text block returned next to a tool_use is filtered out so the
    persisted/reloaded assistant turn never carries a zero-length text block."""
    from src.agent.agent import AgentResponse, AgentTextBlock, AgentToolUseBlock

    agent, store, mock_client = await _make_agent_with_stub_client(tmp_path)
    mock_client.complete = AsyncMock(
        side_effect=[
            AgentResponse(
                content=[
                    AgentTextBlock(text=""),
                    AgentToolUseBlock(id="t1", name="execute_code", input={"code": "1"}),
                ],
                stop_reason="tool_use",
                usage={},
            ),
            AgentResponse(
                content=[AgentTextBlock(text="done")], stop_reason="end_turn", usage={}
            ),
        ]
    )
    agent._handle_tool_call = AsyncMock(return_value={"output": "ok"})

    received = []

    async def on_message(message):
        received.append(message)

    await agent.chat("go", conversation=[], on_message=on_message)

    # First persisted turn is the assistant tool_use turn — tool_use only.
    assert [b["type"] for b in received[0]["content"]] == ["tool_use"]
    # No empty text block anywhere in the persisted turns.
    for m in received:
        for b in m["content"]:
            if b.get("type") == "text":
                assert b["text"] != ""

    await store.close()


async def test_title_generation_ignores_tool_result_user_rows(tmp_path):
    """tool_result rows (sender=user, no text) must not satisfy the auto-title
    user-message threshold."""
    from src.agent.conversation import maybe_generate_session_title

    store = Store(path=tmp_path / "test.db")
    await store.initialize()
    session_id = "sess"
    await store.chat.ensure_session(rkey=session_id, title="")

    # One real human user turn + a structured tool turn whose tool_result row is
    # also stored with sender="user".
    await store.chat.create_message(
        session_id=session_id,
        sender="user",
        content=json.dumps({"role": "user", "content": "do a thing"}),
    )
    await store.chat.create_message(
        session_id=session_id,
        sender="assistant",
        content=json.dumps(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "execute_code", "input": {"code": "x"}}
                ],
            }
        ),
    )
    await store.chat.create_message(
        session_id=session_id,
        sender="user",
        content=json.dumps(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
            }
        ),
    )

    class _CountingLLM:
        def __init__(self):
            self.called = False

        async def complete(self, *args, **kwargs):
            self.called = True
            return "Some Title"

    llm = _CountingLLM()
    # Only one real human user turn (< TITLE_GEN_MIN_USER_MESSAGES) -> no title.
    result = await maybe_generate_session_title(store, session_id, llm)
    assert result is None
    assert llm.called is False

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


async def test_load_session_with_ids_no_drop_tail(tmp_path):
    """drop_current_user_tail=False preserves a trailing user message."""
    store = Store(path=tmp_path / "test.db")
    await store.initialize()

    session_id = "test-sess"
    await store.chat.ensure_session(rkey=session_id, title="test")

    # End on a user message (simulating an unanswered turn)
    await store.chat.create_message(
        session_id=session_id, sender="user",
        content=json.dumps({"role": "user", "content": "unanswered question"}),
    )

    ctx = ToolContext(store=store)
    conv_manager = ConversationManager(ctx=ctx, llm_client=None)

    # With drop_current_user_tail=True (default), the user message is dropped
    msgs_drop, _ = await conv_manager.load_session_with_ids(session_id)
    assert len(msgs_drop) == 0

    # With drop_current_user_tail=False, the user message is preserved
    msgs_keep, _ = await conv_manager.load_session_with_ids(
        session_id, drop_current_user_tail=False
    )
    assert len(msgs_keep) == 1
    assert msgs_keep[0]["content"] == "unanswered question"

    await store.close()


async def test_compaction_does_not_orphan_tool_result(tmp_path):
    """After compaction + repair, no tool_result lacks a matching tool_use.

    Compaction splits purely by char budget, so it can land between an
    assistant tool_use message and its user tool_result message — orphaning
    the tool_result. _repair_conversation must neutralize such orphans.
    """
    sub_llm = _StubSubLLM("Compacted summary.")
    agent, store, _ = await _make_agent_with_stub_client(
        tmp_path, sub_llm=sub_llm, compact_threshold=500
    )

    conversation = []
    # padding messages to push us over the threshold
    for i in range(6):
        conversation.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Padding message {i} " * 20,
        })
    # assistant tool_use message
    conversation.append({
        "role": "assistant",
        "content": [{
            "type": "tool_use", "id": "tool_1",
            "name": "execute_code", "input": {"code": "1+1"},
        }],
    })
    # user tool_result — large enough that the char-budget split keeps it
    # in `recent` while its matching tool_use falls into `older`.
    conversation.append({
        "role": "user",
        "content": [{
            "type": "tool_result", "tool_use_id": "tool_1",
            "content": "R" * 400,
        }],
    })

    await agent._maybe_compact(conversation)
    agent._repair_conversation(conversation)

    # Collect all tool_use ids present in the conversation
    tool_use_ids: set[str] = set()
    for msg in conversation:
        content = msg.get("content", [])
        if msg["role"] == "assistant" and isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tool_use_ids.add(b["id"])

    # Assert: every tool_result has a matching tool_use
    for msg in conversation:
        content = msg.get("content", [])
        if msg["role"] == "user" and isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    assert b["tool_use_id"] in tool_use_ids, (
                        f"Orphaned tool_result for tool_use_id={b['tool_use_id']} "
                        f"with no matching tool_use in conversation"
                    )

    # The pair is kept together in `recent`, so the tool's actual output
    # survives compaction rather than being neutralized to a placeholder.
    joined = json.dumps(conversation)
    assert "tool_1" in joined
    assert "R" * 400 in joined
    assert "[orphaned tool result removed]" not in joined

    await store.close()

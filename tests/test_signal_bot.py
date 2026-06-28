"""Tests for SignalBot message handling, filtering, and lifecycle."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.agent.conversation import ConversationManager
from src.signal_bot.bot import SignalBot
from src.store.store import Store
from src.tools.registry import ToolContext, ToolRegistry
from src.tools.executor import ToolExecutor


class MockResponse:
    """Simple mock HTTP response."""

    def __init__(self, status_code=200, json_data=None, text="OK", content=b""):
        self.status_code = status_code
        self.text = text
        self.content = content
        self._json_data = json_data

    def json(self):
        if self._json_data is not None:
            return self._json_data
        return []


def _make_signal_message(source, text="", attachments=None):
    """Build a signal-cli-rest-api envelope structure."""
    data_msg = {}
    if text:
        data_msg["message"] = text
    if attachments:
        data_msg["attachments"] = attachments
    return {"envelope": {"source": source, "dataMessage": data_msg}}


@pytest.fixture
async def signal_bot(tmp_path):
    """Yield (bot, agent, store, mock_http) with automatic cleanup."""
    store = Store(path=tmp_path / "test.db")
    await store.initialize()

    mock_http = MagicMock()
    mock_http.get = AsyncMock(return_value=MockResponse(json_data=[]))
    mock_http.post = AsyncMock(return_value=MockResponse(200))
    mock_http.request = AsyncMock(return_value=MockResponse(204))

    ctx = ToolContext(store=store, http_client=mock_http)
    registry = ToolRegistry()
    executor = ToolExecutor(registry=registry, ctx=ctx)

    agent = MagicMock()
    agent.chat = AsyncMock(return_value="agent response")

    bot = SignalBot(
        signal_service="127.0.0.1:8080",
        bot_phone="+1111111111",
        operator_phone="+2222222222",
        agent=agent,
        executor=executor,
    )

    yield bot, agent, store, mock_http
    await store.close()


async def test_signal_message_handling(signal_bot):
    """Operator message is routed to agent.chat and response sent via Signal."""
    bot, agent, store, mock_http = signal_bot

    msg = _make_signal_message("+2222222222", "hello agent")
    await bot._handle_message(msg)

    # Agent should have been called
    agent.chat.assert_called_once()
    call_args = agent.chat.call_args
    # user_text is passed as the first positional arg
    assert call_args.args[0] == "hello agent"

    # Response should have been sent via Signal
    mock_http.post.assert_called()
    send_call = mock_http.post.call_args
    assert send_call.args[0] == "http://127.0.0.1:8080/v2/send"
    assert send_call.kwargs["json"]["message"] == "agent response"
    assert send_call.kwargs["json"]["number"] == "+1111111111"
    assert send_call.kwargs["json"]["recipients"] == ["+2222222222"]


async def test_signal_filters_non_operator(signal_bot):
    """Messages from unknown phone numbers are ignored."""
    bot, agent, store, mock_http = signal_bot

    msg = _make_signal_message("+9999999999", "hello from stranger")
    await bot._handle_message(msg)

    agent.chat.assert_not_called()
    mock_http.post.assert_not_called()


async def test_signal_matches_operator_by_uuid(signal_bot):
    """Operator with username-only account is matched via sourceUuid."""
    bot, agent, store, mock_http = signal_bot

    # Replace the bot's operator_phone with the UUID it should match on
    bot._operator_phone = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    msg = {
        "envelope": {
            "source": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "sourceNumber": None,
            "sourceUuid": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "sourceName": "TestOperator",
            "dataMessage": {"message": "hi from username"},
        }
    }
    await bot._handle_message(msg)

    agent.chat.assert_called_once()
    assert agent.chat.call_args.args[0] == "hi from username"


async def test_signal_empty_message_ignored(signal_bot):
    """Messages with no text and no images are ignored."""
    bot, agent, store, mock_http = signal_bot

    msg = _make_signal_message("+2222222222", "")
    await bot._handle_message(msg)

    agent.chat.assert_not_called()


async def test_signal_persists_structured_turn(signal_bot):
    """User message is saved up front; the agent's full structured turn
    (assistant tool_use, tool_result, final text) is persisted via on_message."""
    bot, agent, store, mock_http = signal_bot

    async def mock_chat(
        user_text, conversation, on_compact=None, on_message=None, images=None
    ):
        assert on_message is not None
        await on_message(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "execute_code", "input": {"code": "x"}}
                ],
            }
        )
        await on_message(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
                ],
            }
        )
        await on_message(
            {"role": "assistant", "content": [{"type": "text", "text": "agent response"}]}
        )
        return "agent response"

    agent.chat = mock_chat

    msg = _make_signal_message("+2222222222", "test message")
    await bot._handle_message(msg)

    data = await store.chat.list_messages(session_id="signal-persistent", limit=100)
    messages = data["messages"]
    # user + assistant(tool_use) + tool_result(role=user) + assistant(text)
    assert len(messages) == 4
    assert [m["sender"] for m in messages] == ["user", "assistant", "user", "assistant"]

    # The tool_use turn round-trips as structured content.
    tool_use_turn = json.loads(messages[1]["content"])
    assert tool_use_turn["content"][0]["type"] == "tool_use"

    # And the agent would see the full structured history on the next turn.
    conv_mgr = ConversationManager(ctx=bot._executor.ctx)
    reloaded, _ = await conv_mgr.load_session_with_ids("signal-persistent")
    types = [
        (m["role"], m["content"][0]["type"] if isinstance(m["content"], list) else "text")
        for m in reloaded
    ]
    assert ("assistant", "tool_use") in types
    assert ("user", "tool_result") in types


async def test_signal_compaction_persisted(signal_bot):
    """When compaction fires, set_compaction_checkpoint is called with the correct ID."""
    bot, agent, store, mock_http = signal_bot

    session_id = "signal-persistent"
    await store.chat.ensure_session(rkey=session_id, title="Signal")

    # Seed enough messages so msg_ids has real IDs after load_session_with_ids
    for i in range(5):
        await store.chat.create_message(
            session_id=session_id,
            sender="user" if i % 2 == 0 else "assistant",
            content=json.dumps({"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}),
        )

    async def mock_chat(
        user_text, conversation, on_compact=None, on_message=None, images=None
    ):
        if on_compact:
            # Simulate compacting 3 messages from the start
            await on_compact("test summary", 3)
        return "response"

    agent.chat = mock_chat

    msg = _make_signal_message("+2222222222", "trigger compaction")
    await bot._handle_message(msg)

    # Check that the checkpoint was persisted
    checkpoint = await store.chat.get_compaction_checkpoint(session_id)
    assert checkpoint is not None
    assert checkpoint["summary"] == "test summary"
    # 5 seeded rows (ids 1..5) then the new user row (id 6, dropped on load);
    # split_idx=3 maps to msg_ids[2] == id 3, so the checkpoint must land there.
    assert checkpoint["compacted_up_to_msg_id"] == 3


async def test_signal_close_stops_polling(signal_bot):
    """close() sets the stopped event so start() exits cleanly."""
    bot, agent, store, mock_http = signal_bot

    # Start the bot in a background task, then close it
    task = asyncio.create_task(bot.start())
    await asyncio.sleep(0.1)  # let it start polling

    await bot.close()

    # The task should complete within a short time
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except asyncio.TimeoutError:
        task.cancel()
        pytest.fail("start() did not exit after close()")


async def test_signal_poll_handles_http_error(signal_bot):
    """_poll_once logs and returns gracefully on HTTP errors."""
    bot, agent, store, mock_http = signal_bot
    mock_http.get.return_value = MockResponse(500)

    # Should not raise
    await bot._poll_once()

    # No messages should have been processed
    agent.chat.assert_not_called()


async def test_signal_poll_handles_connection_error(signal_bot):
    """_poll_once logs and returns gracefully when signal-cli-rest-api is down."""
    bot, agent, store, mock_http = signal_bot
    mock_http.get.side_effect = httpx.ConnectError("Connection refused")

    # Should not raise
    await bot._poll_once()
    agent.chat.assert_not_called()


async def test_signal_send_response_chunks_long_text(signal_bot):
    """_send_response chunks long text into multiple messages."""
    bot, agent, store, mock_http = signal_bot

    long_text = "x" * 5000
    await bot._send_response(long_text)

    # Should have been split into multiple chunks
    assert mock_http.post.call_count > 1


async def test_signal_send_response_handles_error(signal_bot):
    """_send_response logs and stops on HTTP error."""
    bot, agent, store, mock_http = signal_bot
    mock_http.post.return_value = MockResponse(500, text="Server Error")

    # Should not raise
    await bot._send_response("test message")

    # Only one attempt (stops on first error)
    assert mock_http.post.call_count == 1

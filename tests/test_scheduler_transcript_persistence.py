"""Scheduled-run transcripts must be persisted to per-schedule sessions.

Previously `on_schedule_fire` ran `agent.chat(..., conversation=[])` and
discarded the transcript — tool payloads (e.g. delve notification digests)
left no audit trail, so incidents could only be reconstructed from the
agent's after-the-fact memory. The callback now persists the prompt as a
user message and every assistant/tool message via `on_message`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.store.store import Store

import main


@pytest.fixture
async def store(tmp_path):
    store = Store(path=tmp_path / "test.db")
    await store.initialize()
    yield store
    await store.close()


def _executor(store: Store) -> MagicMock:
    from src.tools.registry import ToolContext

    ctx = ToolContext(store=store)
    ex = MagicMock()
    ex.scheduler = MagicMock()  # non-None so _wire_scheduler wires the callback
    ex.store = store
    ex.ctx = ctx
    ex.llm_client = None
    return ex


def _agent() -> MagicMock:
    agent = MagicMock()

    async def fake_chat(
        user_message: str,
        conversation: list[dict[str, Any]],
        on_message: Any = None,
        **kwargs: Any,
    ) -> str:
        assert conversation == []  # each fire still runs fresh
        # Simulate a structured turn: assistant message with tool_use, then
        # a tool-results message, both handed to on_message as they occur.
        if on_message is not None:
            await on_message(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "checking notifications"},
                        {"type": "tool_use", "id": "t1", "name": "execute_code", "input": {}},
                    ],
                }
            )
            await on_message(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "{'count': 1}"}
                    ],
                }
            )
        return "did the thing"

    agent.chat = fake_chat
    return agent


async def test_scheduled_run_transcript_is_persisted(store: Store):
    ex = _executor(store)
    agent = _agent()
    main._wire_scheduler(ex, agent)
    on_fire = ex.scheduler._on_fire
    assert on_fire is not None

    response = await on_fire("check_delve_notifications", "call delve.get_notifications")
    assert response == "did the thing"

    # The agent got an on_message callback and was called with it.
    session_id = "schedule-check_delve_notifications"
    messages = await store.chat.list_messages(session_id=session_id, limit=50)
    msgs = messages.get("messages", [])
    senders = [m.get("sender") for m in msgs]
    # prompt (user) + assistant + tool-result (user) = 3 messages
    assert senders == ["user", "assistant", "user"]
    import json as _json

    first = _json.loads(msgs[0]["content"])
    assert first == {"role": "user", "content": "call delve.get_notifications"}
    # the tool payload survives verbatim — the audit trail reason for this
    tool_result = _json.loads(msgs[2]["content"])
    assert tool_result["content"][0]["content"] == "{'count': 1}"


async def test_session_is_reused_across_fires(store: Store):
    ex = _executor(store)
    agent = _agent()
    main._wire_scheduler(ex, agent)

    await ex.scheduler._on_fire("daily_task", "prompt one")
    await ex.scheduler._on_fire("daily_task", "prompt two")

    sessions = await store.chat.list_sessions(limit=50)
    rkeys = [s["rkey"] for s in sessions["sessions"]]
    assert rkeys == ["schedule-daily_task"]  # one session, two runs appended

    messages = await store.chat.list_messages(session_id="schedule-daily_task", limit=50)
    # two full turns appended into the same session
    assert len(messages["messages"]) == 6


async def test_persistence_failure_does_not_break_the_run(store: Store):
    ex = _executor(store)
    # Break persistence: no chat store on the executor.
    broken_store = MagicMock()
    broken_store.chat = None
    ex.store = broken_store
    agent = _agent()
    main._wire_scheduler(ex, agent)

    response = await ex.scheduler._on_fire("some_task", "do things")
    assert response == "did the thing"  # run still succeeds, just unpersisted


async def test_on_message_persistence_failure_does_not_break_the_run(store: Store):
    """If per-message persistence fails mid-turn, the agent turn continues."""
    from src.tools.registry import ToolContext

    ctx = MagicMock(spec=ToolContext)
    ctx.store = store
    from src.agent.conversation import ConversationManager

    real_save = ConversationManager.save_message

    async def flaky_save(self: ConversationManager, session_id: str, message: dict[str, Any]) -> None:
        if message["role"] == "user":  # tool-result messages "fail"
            raise RuntimeError("disk on fire")
        await real_save(self, session_id, message)

    ConversationManager.save_message = flaky_save
    try:
        ex = _executor(store)
        agent = _agent()
        main._wire_scheduler(ex, agent)
        response = await ex.scheduler._on_fire("task", "prompt")
        assert response == "did the thing"
    finally:
        ConversationManager.save_message = real_save

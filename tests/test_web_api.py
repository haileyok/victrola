"""Tests for the web API (FastAPI backend)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.agent.agent import Agent, AgentEvent
from src.agent.conversation import ConversationManager
from src.store.store import Store
from src.tools.custom import CustomTool, CustomToolManager
from src.tools.executor import ToolExecutor
from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolRegistry
from src.tools.secrets import SecretManager
from src.scheduler.scheduler import Scheduler, ScheduledTask
from src.web.app import create_app


async def _build_full_executor(tmp_path: Path) -> ToolExecutor:
    """Build an executor with real store + secrets + scheduler + custom tools."""
    store = Store(path=tmp_path / "test.db")
    await store.initialize()

    ctx = ToolContext(store=store)
    executor = ToolExecutor(registry=TOOL_REGISTRY, ctx=ctx)

    # attach secret manager
    sm = SecretManager(path=tmp_path / "secrets.json")
    await sm.load_secrets()
    executor._secret_manager = sm
    ctx._secret_manager = sm

    # attach scheduler
    scheduler = Scheduler(store=store.documents)
    await scheduler.load_tasks()
    executor._scheduler = scheduler
    ctx._scheduler = scheduler

    # attach custom tool manager
    ctm = CustomToolManager(store=store, executor=executor, secret_manager=sm)
    await ctm.load_tools()
    executor._custom_tool_manager = ctm
    ctx._custom_tool_manager = ctm

    # attach search engine (keyword-only, no embedding client)
    from src.memory.search import SearchEngine
    search_engine = SearchEngine(store=store.memory, embedding_client=None)
    ctx._search_engine = search_engine

    return executor


def _make_stub_agent() -> Agent:
    """Create an Agent with a stub client that returns a fixed string."""
    from src.agent.agent import AgentClient, AgentResponse, AgentTextBlock

    class _StubClient(AgentClient):
        def __init__(self) -> None:
            self.calls: list = []

        async def complete(self, messages, system=None, tools=None):
            self.calls.append({"messages": messages})
            return AgentResponse(
                content=[AgentTextBlock(text="stub response")],
                stop_reason="end_turn",
                usage={"input_tokens": 10, "output_tokens": 5},
            )

    stub = _StubClient()
    # Build agent with openapi to avoid needing a real API key, then swap client
    with patch.object(Agent, "__init__", lambda self, *a, **kw: None):
        agent = Agent.__new__(Agent)
    agent._client = stub
    agent._system_prompt = "test prompt"
    agent._system_prompt_provider = None
    agent._tool_executor = None
    agent._max_iterations = 30
    agent._sub_llm_client = None
    agent._compact_threshold_chars = 240_000
    agent._memory_recall = None
    return agent


@pytest.fixture
def stub_agent():
    return _make_stub_agent()


@pytest.fixture
def app_client(tmp_path):
    """Synchronous fixture using TestClient — sets up executor + app."""
    import asyncio as _asyncio

    loop = _asyncio.new_event_loop()
    executor = loop.run_until_complete(_build_full_executor(Path(tmp_path)))
    agent = _make_stub_agent()
    conv_manager = ConversationManager(ctx=executor.ctx, llm_client=None)
    app = create_app(agent, executor, conv_manager)
    client = TestClient(app)
    client._executor = executor  # type: ignore[attr-defined]
    client._loop = loop  # type: ignore[attr-defined]
    yield client
    loop.run_until_complete(executor.store.close())
    loop.close()


class TestStatus:
    def test_status(self, app_client):
        resp = app_client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "model" in data
        assert "discord" in data
        assert "schedules" in data
        assert "secrets" in data
        assert "custom_tools_approved" in data
        assert "custom_tools_pending" in data


class TestSessions:
    def test_list_empty(self, app_client):
        resp = app_client.get("/api/sessions")
        assert resp.status_code == 200
        assert resp.json()["sessions"] == []

    def test_create_and_get(self, app_client):
        resp = app_client.post("/api/sessions", json={"title": "test session"})
        assert resp.status_code == 201
        rkey = resp.json()["rkey"]
        assert rkey

        # get session
        resp = app_client.get(f"/api/sessions/{rkey}")
        assert resp.status_code == 200
        assert resp.json()["title"] == "test session"

    def test_delete(self, app_client):
        # create
        resp = app_client.post("/api/sessions", json={"title": "to delete"})
        rkey = resp.json()["rkey"]

        # delete
        resp = app_client.delete(f"/api/sessions/{rkey}")
        assert resp.status_code == 204

        # verify gone
        resp = app_client.get(f"/api/sessions/{rkey}")
        assert resp.status_code == 404

    def test_list_messages_empty(self, app_client):
        resp = app_client.post("/api/sessions", json={"title": "msg test"})
        rkey = resp.json()["rkey"]

        resp = app_client.get(f"/api/sessions/{rkey}/messages")
        assert resp.status_code == 200
        assert resp.json()["messages"] == []

    def test_list_messages_missing_session(self, app_client):
        resp = app_client.get("/api/sessions/nonexistent/messages")
        assert resp.status_code == 404


class TestSecrets:
    def test_list_empty(self, app_client):
        resp = app_client.get("/api/secrets")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_set_and_list(self, app_client):
        resp = app_client.post(
            "/api/secrets", json={"name": "MY_SECRET", "value": "secretvalue123"}
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "MY_SECRET"
        assert "masked_value" in data
        assert "*" in data["masked_value"]

        resp = app_client.get("/api/secrets")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["name"] == "MY_SECRET"

    def test_delete(self, app_client):
        app_client.post("/api/secrets", json={"name": "TO_DELETE", "value": "val"})

        resp = app_client.delete("/api/secrets/TO_DELETE")
        assert resp.status_code == 204

        resp = app_client.get("/api/secrets")
        assert resp.json() == []

    def test_delete_not_found(self, app_client):
        resp = app_client.delete("/api/secrets/NONEXISTENT")
        assert resp.status_code == 404


class TestSchedules:
    def test_list_empty(self, app_client):
        resp = app_client.get("/api/schedules")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_and_list(self, app_client):
        resp = app_client.post(
            "/api/schedules",
            json={"name": "test_task", "schedule": "30m", "prompt": "do stuff"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "test_task"
        assert data["schedule"] == "30m"
        assert data["enabled"] is True

        resp = app_client.get("/api/schedules")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_invalid_schedule(self, app_client):
        resp = app_client.post(
            "/api/schedules",
            json={"name": "bad", "schedule": "notaschedule", "prompt": "test"},
        )
        assert resp.status_code == 400

    def test_enable_disable(self, app_client):
        app_client.post(
            "/api/schedules",
            json={"name": "toggle", "schedule": "1h", "prompt": "test"},
        )

        # disable
        resp = app_client.post("/api/schedules/toggle/disable")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

        # enable
        resp = app_client.post("/api/schedules/toggle/enable")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is True

    def test_delete(self, app_client):
        app_client.post(
            "/api/schedules",
            json={"name": "del", "schedule": "1h", "prompt": "test"},
        )
        resp = app_client.delete("/api/schedules/del")
        assert resp.status_code == 204

        resp = app_client.get("/api/schedules")
        assert resp.json() == []


class TestTools:
    def test_list_empty(self, app_client):
        resp = app_client.get("/api/tools")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_and_detail(self, app_client):
        # create a custom tool directly via manager
        executor = app_client._executor  # type: ignore[attr-defined]
        mgr = executor.custom_tool_manager
        tool = CustomTool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            code="// test code",
            approved=False,
            secrets=[],
            requires_net=False,
        )
        app_client._loop.run_until_complete(mgr.create_tool(tool))  # type: ignore[attr-defined]

        resp = app_client.get("/api/tools")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["name"] == "test_tool"

        resp = app_client.get("/api/tools/test_tool")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == "// test code"
        assert data["approved"] is False

    def test_approve_no_secrets(self, app_client):
        executor = app_client._executor  # type: ignore[attr-defined]
        mgr = executor.custom_tool_manager
        tool = CustomTool(
            name="approve_me",
            description="test",
            parameters={"type": "object", "properties": {}},
            code="// code",
        )
        app_client._loop.run_until_complete(mgr.create_tool(tool))  # type: ignore[attr-defined]

        resp = app_client.post("/api/tools/approve_me/approve")
        assert resp.status_code == 200

    def test_approve_missing_secrets(self, app_client):
        executor = app_client._executor  # type: ignore[attr-defined]
        mgr = executor.custom_tool_manager
        tool = CustomTool(
            name="needs_secret",
            description="test",
            parameters={"type": "object", "properties": {}},
            code="// code",
            secrets=["MISSING_KEY"],
        )
        app_client._loop.run_until_complete(mgr.create_tool(tool))  # type: ignore[attr-defined]

        resp = app_client.post("/api/tools/needs_secret/approve")
        assert resp.status_code == 400
        data = resp.json()
        assert "missing_secrets" in str(data) or "MISSING_KEY" in str(data)

    def test_delete(self, app_client):
        executor = app_client._executor  # type: ignore[attr-defined]
        mgr = executor.custom_tool_manager
        tool = CustomTool(
            name="del_tool",
            description="test",
            parameters={"type": "object", "properties": {}},
            code="// code",
        )
        app_client._loop.run_until_complete(mgr.create_tool(tool))  # type: ignore[attr-defined]

        resp = app_client.delete("/api/tools/del_tool")
        assert resp.status_code == 204


class TestSystemPrompt:
    def test_get(self, app_client):
        resp = app_client.get("/api/system-prompt")
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "test prompt"
        assert data["char_count"] == len("test prompt")
        assert data["token_estimate"] > 0


class TestCsrf:
    def test_cross_origin_post_blocked(self, app_client):
        """POSTs with a non-localhost Origin header should be rejected."""
        resp = app_client.post(
            "/api/sessions",
            json={"title": "csrf test"},
            headers={"origin": "https://evil.com"},
        )
        assert resp.status_code == 403

    def test_localhost_origin_allowed(self, app_client):
        """POSTs from localhost (any port) should be allowed."""
        resp = app_client.post(
            "/api/sessions",
            json={"title": "ok"},
            headers={"origin": "http://localhost:5173"},
        )
        assert resp.status_code == 201

    def test_no_origin_allowed(self, app_client):
        """POSTs without an Origin header (non-browser clients) are allowed."""
        resp = app_client.post("/api/sessions", json={"title": "no origin"})
        assert resp.status_code == 201

    def test_localhost_subdomain_blocked(self, app_client):
        """Origins like http://localhost.evil.com must not bypass the check."""
        resp = app_client.post(
            "/api/sessions",
            json={"title": "csrf"},
            headers={"origin": "http://localhost.evil.com"},
        )
        assert resp.status_code == 403

    def test_127_subdomain_blocked(self, app_client):
        """Origins like http://127.0.0.1.evil.com must not bypass the check."""
        resp = app_client.post(
            "/api/sessions",
            json={"title": "csrf"},
            headers={"origin": "http://127.0.0.1.evil.com"},
        )
        assert resp.status_code == 403


class TestChatSSE:
    def test_chat_streams_events(self, app_client):
        """Verify the SSE event stream format including synthesized response/done."""
        # create a session
        resp = app_client.post("/api/sessions", json={"title": "chat test"})
        rkey = resp.json()["rkey"]

        # send a chat message and collect SSE events
        resp = app_client.post(
            f"/api/sessions/{rkey}/chat",
            json={"message": "hello"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # parse SSE events from the response body
        body = resp.text
        events: list[tuple[str, dict]] = []
        for part in body.split("\n\n"):
            if not part.strip():
                continue
            lines = part.split("\n")
            event_name = "message"
            data_str = "{}"
            for line in lines:
                if line.startswith("event: "):
                    event_name = line[7:]
                elif line.startswith("data: "):
                    data_str = line[6:]
            import json

            events.append((event_name, json.loads(data_str)))

        event_names = [e[0] for e in events]
        # the stub agent returns immediately with "stub response" — we should
        # get at least a response event and a done event
        assert "response" in event_names
        assert "done" in event_names

        # find the response event
        response_data = next(d for n, d in events if n == "response")
        assert response_data["text"] == "stub response"

    def test_chat_persists_messages(self, app_client):
        """After chat, both user and assistant messages should be in the store."""
        resp = app_client.post("/api/sessions", json={"title": "persist test"})
        rkey = resp.json()["rkey"]

        app_client.post(f"/api/sessions/{rkey}/chat", json={"message": "hello"})

        # check messages were saved
        resp = app_client.get(f"/api/sessions/{rkey}/messages")
        assert resp.status_code == 200
        messages = resp.json()["messages"]
        # should have user + assistant messages
        assert len(messages) >= 2

    def test_chat_empty_message(self, app_client):
        resp = app_client.post("/api/sessions", json={"title": "err test"})
        rkey = resp.json()["rkey"]

        resp = app_client.post(f"/api/sessions/{rkey}/chat", json={"message": ""})
        assert resp.status_code == 400

    def test_chat_session_not_found(self, app_client):
        resp = app_client.post(
            "/api/sessions/nonexistent/chat", json={"message": "hello"}
        )
        assert resp.status_code == 404

    def test_chat_concurrent_returns_409(self, app_client):
        """A second chat on the same session while one is in progress returns 409."""
        # create a session
        resp = app_client.post("/api/sessions", json={"title": "concurrent"})
        rkey = resp.json()["rkey"]

        # acquire the lock manually to simulate a chat in progress
        from src.web.routers.chat import _session_locks

        lock = asyncio.Lock()
        app_client._loop.run_until_complete(lock.acquire())
        _session_locks[rkey] = lock

        try:
            resp = app_client.post(
                f"/api/sessions/{rkey}/chat", json={"message": "hello"}
            )
            assert resp.status_code == 409
        finally:
            lock.release()


class TestMemory:
    def test_list_empty(self, app_client):
        resp = app_client.get("/api/memory")
        assert resp.status_code == 200
        assert resp.json()["entries"] == []

    def test_create_and_get(self, app_client):
        resp = app_client.post("/api/memory", json={
            "type": "episodic", "scope": "test-session",
            "content": "User deployed to prod", "tags": ["deploy"],
        })
        assert resp.status_code == 201
        entry_id = resp.json()["id"]

        resp = app_client.get(f"/api/memory/{entry_id}")
        assert resp.status_code == 200
        assert resp.json()["content"] == "User deployed to prod"

    def test_create_invalid_type(self, app_client):
        resp = app_client.post("/api/memory", json={
            "type": "invalid", "scope": "x", "content": "test",
        })
        assert resp.status_code == 422

    def test_create_empty_content_rejected(self, app_client):
        """Empty or whitespace-only content is rejected (mirrors memory.add tool)."""
        resp = app_client.post("/api/memory", json={
            "type": "factual", "scope": "topic", "content": "",
        })
        assert resp.status_code == 422
        resp = app_client.post("/api/memory", json={
            "type": "factual", "scope": "topic", "content": "   ",
        })
        assert resp.status_code == 422

    def test_create_self_singleton_conflict(self, app_client):
        # create first self entry
        app_client.post("/api/memory", json={
            "type": "self", "scope": "self", "content": "I am the agent",
        })
        # second should fail
        resp = app_client.post("/api/memory", json={
            "type": "self", "scope": "self", "content": "duplicate",
        })
        assert resp.status_code == 409

    def test_update(self, app_client):
        create = app_client.post("/api/memory", json={
            "type": "factual", "scope": "topic", "content": "original",
        })
        entry_id = create.json()["id"]
        resp = app_client.put(f"/api/memory/{entry_id}", json={
            "content": "updated content", "tags": ["new-tag"],
        })
        assert resp.status_code == 200
        assert resp.json()["content"] == "updated content"
        assert resp.json()["metadata"]["tags"] == ["new-tag"]

    def test_update_content_preserves_existing_tags(self, app_client):
        """Content-only update must not clobber existing tags."""
        create = app_client.post("/api/memory", json={
            "type": "factual", "scope": "topic",
            "content": "original", "tags": ["keep-me"],
        })
        entry_id = create.json()["id"]
        # Update only content — tags should be preserved
        resp = app_client.put(f"/api/memory/{entry_id}", json={
            "content": "new content",
        })
        assert resp.status_code == 200
        assert resp.json()["content"] == "new content"
        assert resp.json()["metadata"]["tags"] == ["keep-me"]

    def test_update_empty_content_rejected(self, app_client):
        """Empty/whitespace content on update is rejected."""
        create = app_client.post("/api/memory", json={
            "type": "factual", "scope": "topic", "content": "original",
        })
        entry_id = create.json()["id"]
        resp = app_client.put(f"/api/memory/{entry_id}", json={
            "content": "",
        })
        assert resp.status_code == 422
        resp = app_client.put(f"/api/memory/{entry_id}", json={
            "content": "   ",
        })
        assert resp.status_code == 422

    def test_delete(self, app_client):
        create = app_client.post("/api/memory", json={
            "type": "factual", "scope": "topic", "content": "to delete",
        })
        entry_id = create.json()["id"]
        resp = app_client.delete(f"/api/memory/{entry_id}")
        assert resp.status_code == 204
        resp = app_client.get(f"/api/memory/{entry_id}")
        assert resp.status_code == 404

    def test_delete_not_found(self, app_client):
        resp = app_client.delete("/api/memory/99999")
        assert resp.status_code == 404

    def test_search(self, app_client):
        app_client.post("/api/memory", json={
            "type": "factual", "scope": "topic",
            "content": "The deploy process uses blue-green strategy",
        })
        app_client.post("/api/memory", json={
            "type": "factual", "scope": "topic",
            "content": "Pizza is best with pineapple",
        })
        resp = app_client.post("/api/memory/search", json={
            "query": "deploy",
        })
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) >= 1
        assert "deploy" in results[0]["content"].lower()

    def test_search_empty_query(self, app_client):
        resp = app_client.post("/api/memory/search", json={"query": ""})
        assert resp.status_code == 422

    def test_search_engine_unavailable_returns_503(self, app_client):
        """When SearchEngine is not configured, search returns 503 but list still works."""
        original_se = app_client._executor.ctx._search_engine
        app_client._executor.ctx._search_engine = None
        try:
            resp = app_client.post("/api/memory/search", json={"query": "anything"})
            assert resp.status_code == 503
            # Other endpoints still work
            resp = app_client.get("/api/memory")
            assert resp.status_code == 200
        finally:
            app_client._executor.ctx._search_engine = original_se

    def test_filter_by_type(self, app_client):
        app_client.post("/api/memory", json={
            "type": "factual", "scope": "topic", "content": "fact",
        })
        app_client.post("/api/memory", json={
            "type": "episodic", "scope": "topic", "content": "event",
        })
        resp = app_client.get("/api/memory?type=factual")
        assert resp.status_code == 200
        entries = resp.json()["entries"]
        assert all(e["type"] == "factual" for e in entries)

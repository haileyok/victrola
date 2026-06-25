"""Shared test fixtures for Victrola."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.store.store import Store
from src.tools.registry import ToolContext, ToolRegistry
from src.tools.executor import ToolExecutor


@pytest.fixture
async def temp_store(tmp_path: Path) -> Store:
    """Create a Store against a temp SQLite DB, initialize, yield, close."""
    store = Store(path=tmp_path / "test.db")
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def mock_agent_client():
    """A no-op AgentClient stub that returns canned responses."""
    from src.agent.agent import AgentClient, AgentResponse, AgentTextBlock

    class _StubClient(AgentClient):
        def __init__(self) -> None:
            self.calls: list = []
            self._responses: list[AgentResponse] = []

        def set_response(self, text: str) -> None:
            self._responses.append(
                AgentResponse(
                    content=[AgentTextBlock(text=text)],
                    stop_reason="end_turn",
                    usage={"input_tokens": 10, "output_tokens": 5},
                )
            )

        async def complete(self, messages, system=None, tools=None):
            self.calls.append({"messages": messages, "system": system, "tools": tools})
            if self._responses:
                return self._responses.pop(0)
            return AgentResponse(
                content=[AgentTextBlock(text="stub response")],
                stop_reason="end_turn",
                usage={"input_tokens": 10, "output_tokens": 5},
            )

    return _StubClient()


@pytest.fixture
async def isolated_executor(tmp_path: Path) -> ToolExecutor:
    """A ToolExecutor with a temp store and mock context."""
    store = Store(path=tmp_path / "test.db")
    await store.initialize()

    ctx = ToolContext(store=store)
    registry = ToolRegistry()
    executor = ToolExecutor(registry=registry, ctx=ctx)
    executor._tool_definition = None

    yield executor

    await store.close()


@pytest.fixture
async def isolated_custom_tool_manager(tmp_path: Path):
    """A CustomToolManager backed by a temp store."""
    from src.tools.custom import CustomToolManager

    store = Store(path=tmp_path / "test.db")
    await store.initialize()

    ctx = ToolContext(store=store)
    registry = ToolRegistry()
    executor = ToolExecutor(registry=registry, ctx=ctx)

    manager = CustomToolManager(store=store, executor=executor, secret_manager=None)

    yield manager

    await store.close()

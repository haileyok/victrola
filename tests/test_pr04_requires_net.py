"""Tests for PR 4: Gate custom tool network access behind requires_net flag."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.tools.custom import CustomTool, CustomToolManager


def test_custom_tool_defaults_requires_net_false():
    """CustomTool should default requires_net to False."""
    tool = CustomTool(
        name="test",
        description="test tool",
        parameters={},
        code="output('hello')",
    )
    assert tool.requires_net is False


def test_custom_tool_requires_net_true():
    """CustomTool should accept requires_net=True."""
    tool = CustomTool(
        name="test",
        description="test tool",
        parameters={},
        code="output('hello')",
        requires_net=True,
    )
    assert tool.requires_net is True


def test_to_dict_includes_requires_net():
    """to_dict should include requiresNet."""
    tool = CustomTool(
        name="test",
        description="test tool",
        parameters={},
        code="output('hello')",
        requires_net=True,
    )
    d = tool.to_dict()
    assert d["requiresNet"] is True


def test_from_dict_reads_requires_net():
    """from_dict should read requiresNet."""
    data = {
        "name": "test",
        "description": "test tool",
        "parameters": {},
        "code": "output('hello')",
        "approved": False,
        "requiresNet": True,
    }
    tool = CustomTool.from_dict(data)
    assert tool.requires_net is True


def test_from_dict_defaults_requires_net_false():
    """from_dict should default requires_net to False when missing."""
    data = {
        "name": "test",
        "description": "test tool",
        "parameters": {},
        "code": "output('hello')",
        "approved": False,
    }
    tool = CustomTool.from_dict(data)
    assert tool.requires_net is False


@pytest.mark.asyncio
async def test_execute_tool_passes_requires_net_to_executor():
    """execute_tool should pass allow_net=tool.requires_net to execute_custom_tool_code."""
    store = MagicMock()
    executor = MagicMock()
    executor.execute_custom_tool_code = AsyncMock(return_value={"success": True})

    manager = CustomToolManager(store=store, executor=executor, secret_manager=None)

    # Tool with requires_net=False
    tool_false = CustomTool(
        name="no_net",
        description="no net",
        parameters={},
        code="output('hello')",
        approved=True,
        requires_net=False,
    )
    manager._tools["no_net"] = tool_false
    await manager.execute_tool("no_net", {})
    call_args = executor.execute_custom_tool_code.call_args
    assert call_args.kwargs["allow_net"] is False

    # Tool with requires_net=True
    executor.execute_custom_tool_code.reset_mock()
    tool_true = CustomTool(
        name="with_net",
        description="with net",
        parameters={},
        code="output('hello')",
        approved=True,
        requires_net=True,
    )
    manager._tools["with_net"] = tool_true
    await manager.execute_tool("with_net", {})
    call_args = executor.execute_custom_tool_code.call_args
    assert call_args.kwargs["allow_net"] is True

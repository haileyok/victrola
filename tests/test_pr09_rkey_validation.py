"""Tests for PR 9: Validate names in custom tools.

The rkey validation tests for `notes.py` have been removed — the notes
system was replaced by the entry-based memory system (`memory.*` tools)
which has its own inline validation. Custom tool name validation is
still tested here.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.tools.custom import CustomTool, CustomToolManager


# --- CustomToolManager.create_tool tests ---

@pytest.mark.asyncio
async def test_create_tool_rejects_invalid_name():
    """create_tool should reject names with path separators."""
    store = MagicMock()
    executor = MagicMock()
    manager = CustomToolManager(store=store, executor=executor, secret_manager=None)

    tool = CustomTool(
        name="../../evil",
        description="bad",
        parameters={},
        code="output('hi')",
    )

    result = await manager.create_tool(tool)
    assert "Error" in result
    assert "invalid" in result.lower()


@pytest.mark.asyncio
async def test_create_tool_rejects_name_with_colon():
    """create_tool should reject names with colons (would collide with rkey prefix)."""
    store = MagicMock()
    executor = MagicMock()
    manager = CustomToolManager(store=store, executor=executor, secret_manager=None)

    tool = CustomTool(
        name="customtool:fake",
        description="bad",
        parameters={},
        code="output('hi')",
    )

    result = await manager.create_tool(tool)
    assert "Error" in result
    assert "invalid" in result.lower()


@pytest.mark.asyncio
async def test_create_tool_accepts_valid_name():
    """create_tool should accept valid names."""
    store = MagicMock()
    store.documents = MagicMock()
    store.documents.create = AsyncMock()
    executor = MagicMock()
    manager = CustomToolManager(store=store, executor=executor, secret_manager=None)

    tool = CustomTool(
        name="my_tool.v2",
        description="good",
        parameters={},
        code="output('hi')",
    )

    result = await manager.create_tool(tool)
    assert "created" in result.lower()

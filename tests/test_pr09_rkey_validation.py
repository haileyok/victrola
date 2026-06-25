"""Tests for PR 9: Validate rkey in notes and custom tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.tools.definitions.notes import _validate_rkey, note_upsert
from src.tools.custom import CustomTool, CustomToolManager


# --- _validate_rkey tests ---

def test_validate_rkey_valid():
    assert _validate_rkey("self") is None
    assert _validate_rkey("operator") is None
    assert _validate_rkey("skill:my-skill") is None
    assert _validate_rkey("task_1.0") is None
    assert _validate_rkey("note~with~tilde") is None


def test_validate_rkey_rejects_path_traversal():
    result = _validate_rkey("../../etc/passwd")
    assert result is not None
    assert "invalid" in result.lower()


def test_validate_rkey_rejects_customtool_prefix():
    result = _validate_rkey("customtool:evil")
    assert result is not None
    assert "customtool" in result.lower()
    assert "reserved" in result.lower()


def test_validate_rkey_rejects_empty():
    result = _validate_rkey("")
    assert result is not None
    assert "required" in result.lower()


def test_validate_rkey_rejects_slashes():
    result = _validate_rkey("foo/bar")
    assert result is not None
    assert "invalid" in result.lower()


def test_validate_rkey_rejects_newlines():
    result = _validate_rkey("foo\nbar")
    assert result is not None
    assert "invalid" in result.lower()


def test_validate_rkey_rejects_too_long():
    result = _validate_rkey("a" * 513)
    assert result is not None
    assert "invalid" in result.lower()


# --- note_upsert integration test ---

@pytest.mark.asyncio
async def test_note_upsert_rejects_invalid_rkey():
    """note_upsert should reject path-traversal rkeys without calling the store."""
    ctx = MagicMock()
    ctx.store.documents = MagicMock()
    ctx.store.documents.get = AsyncMock()

    result = await note_upsert(ctx, rkey="../../etc/passwd", content="test")

    assert "Error" in result
    assert "invalid" in result.lower()
    ctx.store.documents.get.assert_not_called()


@pytest.mark.asyncio
async def test_note_upsert_rejects_customtool_prefix():
    """note_upsert should reject customtool: prefix."""
    ctx = MagicMock()
    ctx.store.documents = MagicMock()
    ctx.store.documents.get = AsyncMock()

    result = await note_upsert(ctx, rkey="customtool:evil", content="test")

    assert "Error" in result
    assert "reserved" in result.lower()


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

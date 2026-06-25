"""Tests for PR 16: Fix StoreNotFound misuse for conflicts."""

import pytest
from pathlib import Path

from src.store.store import Store, StoreConflict, StoreNotFound


@pytest.fixture
async def store(tmp_path):
    s = Store(path=tmp_path / "test.db")
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_store_conflict_exists():
    """StoreConflict exception class should exist."""
    assert issubclass(StoreConflict, Exception)


@pytest.mark.asyncio
async def test_document_create_duplicate_raises_conflict(store):
    """DocumentStore.create should raise StoreConflict on duplicate, not StoreNotFound."""
    await store.documents.create("dup_key", "first")
    with pytest.raises(StoreConflict):
        await store.documents.create("dup_key", "second")


@pytest.mark.asyncio
async def test_record_create_duplicate_raises_conflict(store):
    """RecordStore.create should raise StoreConflict on duplicate."""
    await store.records.create("col", "key", {"v": 1})
    with pytest.raises(StoreConflict):
        await store.records.create("col", "key", {"v": 2})


@pytest.mark.asyncio
async def test_store_not_found_still_raised_for_missing(store):
    """StoreNotFound should still be raised for actual not-found conditions."""
    with pytest.raises(StoreNotFound):
        await store.documents.get("nonexistent_key")


@pytest.mark.asyncio
async def test_custom_tool_create_catches_conflict(tmp_path):
    """CustomToolManager.create_tool should catch StoreConflict explicitly."""
    from src.tools.custom import CustomTool, CustomToolManager
    from unittest.mock import MagicMock

    store = Store(path=tmp_path / "test.db")
    await store.initialize()
    executor = MagicMock()
    manager = CustomToolManager(store=store, executor=executor, secret_manager=None)

    tool = CustomTool(
        name="test_tool",
        description="test",
        parameters={},
        code="output('hi')",
    )

    # First create should succeed
    result = await manager.create_tool(tool)
    assert "created" in result.lower()

    # Second create with same name should not raise — should fall back to update
    result2 = await manager.create_tool(tool)
    assert "created" in result2.lower()

    await store.close()

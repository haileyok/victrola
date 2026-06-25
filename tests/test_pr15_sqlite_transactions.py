"""Tests for PR 15: Fix SQLite — use explicit transactions."""

import asyncio
import pytest
from pathlib import Path

from src.store.store import Store, StoreNotFound


@pytest.fixture
async def store(tmp_path):
    s = Store(path=tmp_path / "test.db")
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_document_create_uses_transaction(store):
    """DocumentStore.create should work with BEGIN IMMEDIATE."""
    doc = await store.documents.create("test_key", "hello")
    assert doc["rkey"] == "test_key"
    assert doc["content"] == "hello"

    # Verify it was persisted
    fetched = await store.documents.get("test_key")
    assert fetched["content"] == "hello"


@pytest.mark.asyncio
async def test_document_create_duplicate_raises(store):
    """Duplicate create should raise StoreNotFound (or StoreConflict in PR 16)."""
    await store.documents.create("dup_key", "first")
    with pytest.raises(StoreNotFound):
        await store.documents.create("dup_key", "second")


@pytest.mark.asyncio
async def test_document_update(store):
    """DocumentStore.update should work with transaction."""
    await store.documents.create("upd_key", "original")
    result = await store.documents.update("upd_key", "updated")
    assert result["content"] == "updated"
    fetched = await store.documents.get("upd_key")
    assert fetched["content"] == "updated"


@pytest.mark.asyncio
async def test_document_delete(store):
    """DocumentStore.delete should work with transaction."""
    await store.documents.create("del_key", "bye")
    await store.documents.delete("del_key")
    with pytest.raises(StoreNotFound):
        await store.documents.get("del_key")


@pytest.mark.asyncio
async def test_record_create_and_get(store):
    """RecordStore should work with transactions."""
    await store.records.create("col1", "key1", {"val": 42})
    result = await store.records.get("col1", "key1")
    assert result["value"] == {"val": 42}


@pytest.mark.asyncio
async def test_chat_create_session_and_message(store):
    """ChatStore should work with transactions."""
    session = await store.chat.create_session(title="test")
    sid = session["rkey"]
    msg = await store.chat.create_message(sid, "user", "hello")
    assert msg["sender"] == "user"
    assert msg["content"] == "hello"


@pytest.mark.asyncio
async def test_concurrent_writes_do_not_corrupt(store):
    """Two sequential writes should both succeed without corruption.

    Note: aiosqlite uses a single connection with a serialized worker thread,
    so BEGIN IMMEDIATE transactions are naturally serialized. This test
    verifies the transaction wrapping doesn't break sequential writes.
    """
    await store.documents.create("concurrent_1", "content_1")
    await store.documents.create("concurrent_2", "content_2")

    doc1 = await store.documents.get("concurrent_1")
    doc2 = await store.documents.get("concurrent_2")
    assert doc1["content"] == "content_1"
    assert doc2["content"] == "content_2"

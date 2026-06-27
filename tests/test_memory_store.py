"""Tests for MemoryStore write-lock discipline."""

import asyncio

import pytest

from src.store.store import Store


@pytest.mark.asyncio
async def test_update_entry_does_not_hold_lock_during_embed(tmp_path):
    """update_entry must release the shared write lock during the embed call.

    Holding the global lock across an Ollama embed (which can take seconds)
    blocks every other store write (documents, records, chat, memory).
    add_entry already does its embed outside the lock; update_entry should
    match.
    """
    store = Store(path=tmp_path / "test.db")
    await store.initialize()

    embed_started = asyncio.Event()
    embed_can_finish = asyncio.Event()

    class FakeEmbeddingClient:
        async def embed(self, content):
            embed_started.set()
            await embed_can_finish.wait()
            return b"\x00" * (768 * 4)

    store.memory.set_embedding_client(FakeEmbeddingClient(), dimensions=768)

    # Add an entry to update later (embed outside lock, so fast)
    entry = await store.memory.add_entry(
        type="factual",
        scope="test",
        content="original content",
        embedding=b"\x00" * (768 * 4),
    )
    entry_id = entry["id"]

    # Start update_entry in the background — it will call embed and block
    update_task = asyncio.create_task(
        store.memory.update_entry(entry_id, content="new content")
    )

    # Wait until embed is in-flight
    await asyncio.wait_for(embed_started.wait(), timeout=5.0)

    # While embed is still running, another store write that needs the same
    # shared write lock should complete promptly. On current code this blocks
    # because update_entry holds the lock during embed.
    await asyncio.wait_for(
        store.documents.create("test:concurrent", "data"),
        timeout=3.0,
    )

    # Let embed finish and clean up
    embed_can_finish.set()
    await update_task
    await store.close()

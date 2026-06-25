"""Tests for compaction persistence: schema, checkpoint roundtrip, after_id loading."""

from src.store.store import Store


async def _seed_messages(store: Store, session_id: str, count: int) -> list[int]:
    """Create a session and insert `count` messages. Returns the list of row IDs."""
    await store.chat.ensure_session(rkey=session_id, title="test")
    ids = []
    for i in range(count):
        result = await store.chat.create_message(
            session_id=session_id,
            sender="user" if i % 2 == 0 else "assistant",
            content=f'{{"role": "user", "content": "msg {i}"}}',
        )
        ids.append(result["id"])
    return ids


async def test_schema_migration_adds_compaction_column(tmp_path):
    """Existing DB without compacted_up_to_msg_id should get it on reinit."""
    db_path = tmp_path / "test.db"
    store = Store(path=db_path)
    await store.initialize()
    await store.close()

    # Reopen — the column should be added by the migration
    store2 = Store(path=db_path)
    await store2.initialize()

    # Verify the column exists
    cur = await store2._db.execute("PRAGMA table_info(chat_sessions)")
    columns = [row[1] for row in await cur.fetchall()]
    assert "compacted_up_to_msg_id" in columns

    # Verify the compaction summaries table exists
    cur = await store2._db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chat_compaction_summaries'"
    )
    row = await cur.fetchone()
    assert row is not None

    await store2.close()


async def test_compaction_checkpoint_roundtrip(tmp_path):
    """set checkpoint → get checkpoint → verify summary and ID match."""
    store = Store(path=tmp_path / "test.db")
    await store.initialize()

    session_id = "test-session"
    ids = await _seed_messages(store, session_id, 5)

    # Initially no checkpoint
    assert await store.chat.get_compaction_checkpoint(session_id) is None

    # Set checkpoint
    await store.chat.set_compaction_checkpoint(
        session_id, ids[2], "Summary of first 3 messages"
    )

    # Get checkpoint
    checkpoint = await store.chat.get_compaction_checkpoint(session_id)
    assert checkpoint is not None
    assert checkpoint["summary"] == "Summary of first 3 messages"
    assert checkpoint["compacted_up_to_msg_id"] == ids[2]

    await store.close()


async def test_compaction_checkpoint_overwrite(tmp_path):
    """Setting a new checkpoint advances the ID (monotonic)."""
    store = Store(path=tmp_path / "test.db")
    await store.initialize()

    session_id = "test-session"
    ids = await _seed_messages(store, session_id, 10)

    # First checkpoint at message 3
    await store.chat.set_compaction_checkpoint(
        session_id, ids[2], "First summary"
    )
    cp1 = await store.chat.get_compaction_checkpoint(session_id)
    assert cp1["compacted_up_to_msg_id"] == ids[2]

    # Second checkpoint at message 7
    await store.chat.set_compaction_checkpoint(
        session_id, ids[6], "Second summary (includes first)"
    )
    cp2 = await store.chat.get_compaction_checkpoint(session_id)
    assert cp2["compacted_up_to_msg_id"] == ids[6]
    assert cp2["compacted_up_to_msg_id"] > cp1["compacted_up_to_msg_id"]
    assert cp2["summary"] == "Second summary (includes first)"

    # Both summaries should exist in the table (non-destructive)
    cur = await store._db.execute(
        "SELECT COUNT(*) FROM chat_compaction_summaries WHERE session_id = ?",
        (session_id,),
    )
    row = await cur.fetchone()
    assert row[0] == 2

    await store.close()


async def test_list_messages_after_id(tmp_path):
    """list_messages with after_id should skip messages below the threshold."""
    store = Store(path=tmp_path / "test.db")
    await store.initialize()

    session_id = "test-session"
    ids = await _seed_messages(store, session_id, 5)

    # Without after_id — all 5 messages
    data = await store.chat.list_messages(session_id=session_id, limit=100)
    assert len(data["messages"]) == 5

    # With after_id=3 — only messages 4 and 5
    data = await store.chat.list_messages(
        session_id=session_id, limit=100, after_id=ids[2]
    )
    assert len(data["messages"]) == 2
    assert data["messages"][0]["id"] == ids[3]
    assert data["messages"][1]["id"] == ids[4]

    await store.close()


async def test_non_destructive_compaction(tmp_path):
    """Raw messages remain in chat_messages after compaction."""
    store = Store(path=tmp_path / "test.db")
    await store.initialize()

    session_id = "test-session"
    ids = await _seed_messages(store, session_id, 5)

    # Set checkpoint
    await store.chat.set_compaction_checkpoint(
        session_id, ids[2], "Summary"
    )

    # All 5 raw messages should still exist
    data = await store.chat.list_messages(session_id=session_id, limit=100)
    assert len(data["messages"]) == 5

    await store.close()


async def test_compaction_checkpoint_none_for_no_checkpoint(tmp_path):
    """get_compaction_checkpoint returns None when no checkpoint exists."""
    store = Store(path=tmp_path / "test.db")
    await store.initialize()

    await store.chat.ensure_session(rkey="empty-session", title="test")
    assert await store.chat.get_compaction_checkpoint("empty-session") is None

    await store.close()


async def test_existing_db_data_preserved_on_migration(tmp_path):
    """Existing chat sessions and messages survive the schema migration."""
    db_path = tmp_path / "test.db"

    # Create a store with the old schema (no compaction column)
    store = Store(path=db_path)
    await store.initialize()

    # Seed data
    session_id = "preserve-test"
    await store.chat.ensure_session(rkey=session_id, title="preserve")
    await store.chat.create_message(
        session_id=session_id, sender="user", content="hello"
    )
    await store.close()

    # Reopen — migration runs
    store2 = Store(path=db_path)
    await store2.initialize()

    # Data should be preserved
    session = await store2.chat.get_session(session_id)
    assert session is not None
    assert session["title"] == "preserve"

    data = await store2.chat.list_messages(session_id=session_id)
    assert len(data["messages"]) == 1
    assert data["messages"][0]["content"] == "hello"

    # Existing row should have NULL compacted_up_to_msg_id
    cur = await store2._db.execute(
        "SELECT compacted_up_to_msg_id FROM chat_sessions WHERE rkey = ?",
        (session_id,),
    )
    row = await cur.fetchone()
    assert row[0] is None

    await store2.close()


async def test_stale_checkpoint_write_ignored(tmp_path):
    """A regressed checkpoint write must not create a stale summary row."""
    store = Store(path=tmp_path / "test.db")
    await store.initialize()

    session_id = "stale-test"
    ids = await _seed_messages(store, session_id, 10)

    # First checkpoint at message 7
    await store.chat.set_compaction_checkpoint(
        session_id, ids[6], "Summary up to 7"
    )
    cp1 = await store.chat.get_compaction_checkpoint(session_id)
    assert cp1["compacted_up_to_msg_id"] == ids[6]

    # Stale/regressed write at message 3 — should be ignored
    await store.chat.set_compaction_checkpoint(
        session_id, ids[2], "Stale summary up to 3"
    )

    # Checkpoint should still point to message 7
    cp2 = await store.chat.get_compaction_checkpoint(session_id)
    assert cp2["compacted_up_to_msg_id"] == ids[6]
    assert cp2["summary"] == "Summary up to 7"

    # Only one summary row should exist (the stale one was not inserted)
    cur = await store._db.execute(
        "SELECT COUNT(*) FROM chat_compaction_summaries WHERE session_id = ?",
        (session_id,),
    )
    row = await cur.fetchone()
    assert row[0] == 1

    await store.close()

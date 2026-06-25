"""One-time migration: agent_documents → memory_entries.

Non-destructive — original rows are left in `agent_documents` as inert
orphans. Only `customtool:*` entries are skipped (they stay in
`agent_documents` for `CustomToolManager`).
"""

import json
import logging
from typing import Any

from src.store.store import Store

logger = logging.getLogger(__name__)


async def migrate_documents_to_memory(
    store: Store, embedding_client: Any | None
) -> int:
    """Migrate agent_documents → memory_entries.

    Returns the number of entries migrated. Safe to run every startup —
    tracks migrated rkeys in `memory_config` so edited/deleted entries
    are not re-imported.
    """
    if store.memory is None:
        logger.warning("MemoryStore not initialized — skipping migration")
        return 0

    # Load already-migrated rkeys from memory_config
    cur = await store._db.execute(
        "SELECT value FROM memory_config WHERE key = 'migrated_rkeys'"
    )
    row = await cur.fetchone()
    if row:
        migrated_rkeys: set[str] = set(json.loads(row[0]))
    else:
        migrated_rkeys = set()

    # Load all documents
    cur = await store._db.execute(
        "SELECT rkey, content FROM agent_documents ORDER BY rkey"
    )
    docs = await cur.fetchall()

    if not docs:
        logger.debug("agent_documents is empty — nothing to migrate")
        return 0

    migrated = 0
    for rkey, content in docs:
        # Skip custom tool entries — they stay in agent_documents
        if rkey.startswith("customtool:"):
            continue

        # Skip already-migrated rkeys (prevents re-importing stale content
        # after the user edits or deletes a migrated entry)
        if rkey in migrated_rkeys:
            continue

        entry_type, scope = _classify_document(rkey)

        # Generate embedding if client is available
        embedding = None
        if embedding_client is not None:
            try:
                embedding = await embedding_client.embed(content)
            except Exception:
                logger.warning(
                    "Failed to generate embedding for '%s' during migration",
                    rkey,
                    exc_info=True,
                )

        # Insert memory entry + rkey marker atomically in one transaction.
        # We bypass MemoryStore.add_entry to keep both writes in the same
        # transaction — if startup crashes, either both are committed or neither.
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        meta = json.dumps({})
        try:
            async with store.memory._write_lock:
                await store._db.execute("BEGIN IMMEDIATE")
                await store._db.execute(
                    "INSERT INTO memory_entries "
                    "(type, scope, content, metadata, embedding, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (entry_type, scope, content, meta, embedding, now, now),
                )
                migrated_rkeys.add(rkey)
                await store._db.execute(
                    "INSERT INTO memory_config (key, value) VALUES ('migrated_rkeys', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (json.dumps(sorted(migrated_rkeys)),),
                )
                await store._db.commit()
            migrated += 1
            logger.info("Migrated '%s' → memory_entries (type=%s, scope=%s)", rkey, entry_type, scope)
        except Exception:
            try:
                await store._db.rollback()
            except Exception:
                pass
            logger.warning("Failed to migrate '%s'", rkey, exc_info=True)
            migrated_rkeys.discard(rkey)

    logger.info("Migration complete: %d documents migrated to memory_entries", migrated)
    return migrated


def _classify_document(rkey: str) -> tuple[str, str]:
    """Classify an agent_documents rkey into (type, scope) for memory_entries.

    - `self` → ('self', 'self')
    - `operator` → ('operator', 'operator')
    - `skill:<name>` → ('skill', 'skill:<name>')
    - `task:<name>` → ('episodic', 'task:<name>')
    - any other rkey → ('factual', rkey)
    """
    if rkey == "self":
        return ("self", "self")
    if rkey == "operator":
        return ("operator", "operator")
    if rkey.startswith("skill:"):
        return ("skill", rkey)
    if rkey.startswith("task:"):
        return ("episodic", rkey)
    return ("factual", rkey)


async def backfill_embeddings(store: Store, embedding_client: Any | None) -> int:
    """Backfill NULL embeddings for entries that don't have them.

    Called at startup when the embedding client is available. Returns
    the number of embeddings backfilled.
    """
    if store.memory is None or embedding_client is None:
        return 0

    entries = await store.memory.get_entries_without_embeddings()
    if not entries:
        return 0

    backfilled = 0
    for entry in entries:
        content = entry.get("content", "")
        entry_id = entry.get("id")
        if not content or entry_id is None:
            continue
        try:
            embedding = await embedding_client.embed(content)
            await store.memory.update_embedding(entry_id, embedding)
            backfilled += 1
        except Exception:
            logger.warning(
                "Failed to backfill embedding for entry %d", entry_id, exc_info=True
            )

    if backfilled > 0:
        logger.info("Backfilled %d embeddings", backfilled)
    return backfilled

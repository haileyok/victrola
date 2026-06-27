"""SQLite CRUD for the entry-based memory system.

Each entry is a discrete, independently editable row in `memory_entries`.
FTS5 triggers (created by Store._create_schema) keep the full-text index
in sync automatically — no manual FTS inserts needed here.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

# Valid memory types
_VALID_TYPES = {"self", "operator", "skill", "episodic", "factual"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    """CRUD for `memory_entries` + FTS5 full-text index.

    Takes the shared aiosqlite.Connection (same pattern as DocumentStore).
    The FTS5 triggers handle index sync on INSERT/UPDATE/DELETE — no
    manual FTS manipulation needed in Python.

    All write methods are serialized via an asyncio.Lock to prevent
    "cannot start a transaction within a transaction" errors on the
    shared connection.
    """

    def __init__(
        self,
        db: aiosqlite.Connection,
        fts5_available: bool = True,
        write_lock: asyncio.Lock | None = None,
    ) -> None:
        self._db = db
        self._fts5_available = fts5_available
        self._embedding_client: Any | None = None
        self._embedding_dimensions: int = 768
        self._write_lock = write_lock or asyncio.Lock()

    def set_embedding_client(self, client: Any | None, dimensions: int = 768) -> None:
        """Set the embedding client used for auto-generating embeddings on write."""
        self._embedding_client = client
        self._embedding_dimensions = dimensions

    def _validate_embedding(self, embedding: bytes | None) -> bytes | None:
        """Validate embedding BLOB size matches configured dimensions.

        Returns the embedding if valid, None if invalid (with a warning).
        """
        if embedding is None:
            return None
        expected_size = self._embedding_dimensions * 4
        if len(embedding) != expected_size:
            logger.warning(
                "Embedding BLOB size %d != expected %d (dims=%d * 4) — storing NULL",
                len(embedding),
                expected_size,
                self._embedding_dimensions,
            )
            return None
        return embedding

    # -- write methods --

    async def add_entry(
        self,
        type: str,
        scope: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedding: bytes | None = None,
    ) -> dict[str, Any]:
        """Insert a new memory entry. Returns the entry with its assigned ID.

        If *embedding* is None and an embedding client is configured, an
        embedding is auto-generated from *content*. If the client is
        unavailable or raises, the entry is stored with a NULL embedding.
        """
        if type not in _VALID_TYPES:
            raise ValueError(f"Invalid memory type '{type}'. Must be one of {_VALID_TYPES}")

        now = _now()
        meta = json.dumps(metadata or {})

        # Auto-generate embedding if client is available and none was passed
        if embedding is None and self._embedding_client is not None:
            try:
                embedding = await self._embedding_client.embed(content)
            except Exception:
                logger.warning("Failed to generate embedding on add", exc_info=True)
                embedding = None

        # Validate embedding size before storing
        embedding = self._validate_embedding(embedding)

        async with self._write_lock:
            try:
                await self._db.execute("BEGIN IMMEDIATE")
                cur = await self._db.execute(
                    "INSERT INTO memory_entries "
                    "(type, scope, content, metadata, embedding, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (type, scope, content, meta, embedding, now, now),
                )
                # Capture lastrowid before commit while still in the transaction
                row_id = cur.lastrowid
                await self._db.commit()
            except Exception:
                await self._db.rollback()
                raise

        # Fetch the inserted row by its captured ID
        cur = await self._db.execute(
            "SELECT id, type, scope, content, metadata, embedding, created_at, updated_at "
            "FROM memory_entries WHERE id = ?",
            (row_id,),
        )
        row = await cur.fetchone()
        return self._row_to_dict(row)

    async def update_entry(
        self,
        id: int,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Update specific fields of an entry by ID.

        If *content* is changed, the embedding is regenerated (when an
        embedding client is available). Only provided fields are updated.
        """
        now = _now()

        # Step 1: short locked read of current values
        async with self._write_lock:
            cur = await self._db.execute(
                "SELECT content, metadata FROM memory_entries WHERE id = ?",
                (id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None

            current_content = row[0]
            current_meta = row[1]

        # Step 2: compute new values and regenerate embedding OUTSIDE the
        # lock so a slow embed doesn't block all other store writes.
        new_content = content if content is not None else current_content
        if metadata is not None:
            new_meta = json.dumps(metadata)
        else:
            new_meta = current_meta

        new_embedding: bytes | None = None
        regenerate_embedding = content is not None and self._embedding_client is not None
        if regenerate_embedding:
            try:
                new_embedding = await self._embedding_client.embed(new_content)
            except Exception:
                logger.warning("Failed to regenerate embedding on update", exc_info=True)
                new_embedding = None

        # Validate embedding size before storing
        new_embedding = self._validate_embedding(new_embedding)

        # Step 3: locked UPDATE with precomputed values
        async with self._write_lock:
            try:
                await self._db.execute("BEGIN IMMEDIATE")
                if regenerate_embedding and new_embedding is not None:
                    await self._db.execute(
                        "UPDATE memory_entries SET content = ?, metadata = ?, "
                        "embedding = ?, updated_at = ? WHERE id = ?",
                        (new_content, new_meta, new_embedding, now, id),
                    )
                elif regenerate_embedding:
                    # Content changed but embedding generation failed — null it out
                    await self._db.execute(
                        "UPDATE memory_entries SET content = ?, metadata = ?, "
                        "embedding = NULL, updated_at = ? WHERE id = ?",
                        (new_content, new_meta, now, id),
                    )
                else:
                    await self._db.execute(
                        "UPDATE memory_entries SET content = ?, metadata = ?, "
                        "updated_at = ? WHERE id = ?",
                        (new_content, new_meta, now, id),
                    )
                await self._db.commit()
            except Exception:
                await self._db.rollback()
                raise

        return await self.get_entry(id)

    async def delete_entry(self, id: int) -> bool:
        """Delete a single entry by ID. Returns True if deleted, False if not found."""
        async with self._write_lock:
            try:
                await self._db.execute("BEGIN IMMEDIATE")
                cur = await self._db.execute(
                    "DELETE FROM memory_entries WHERE id = ?", (id,)
                )
                if cur.rowcount == 0:
                    await self._db.rollback()
                    return False
                await self._db.commit()
            except Exception:
                await self._db.rollback()
                raise
        return True

    # -- read methods --

    async def get_entry(self, id: int) -> dict[str, Any] | None:
        """Fetch a single entry by ID."""
        cur = await self._db.execute(
            "SELECT id, type, scope, content, metadata, embedding, created_at, updated_at "
            "FROM memory_entries WHERE id = ?",
            (id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    async def get_by_scope(self, type: str, scope: str) -> list[dict[str, Any]]:
        """Return all entries for a given type + scope (e.g. all operator facts)."""
        cur = await self._db.execute(
            "SELECT id, type, scope, content, metadata, embedding, created_at, updated_at "
            "FROM memory_entries WHERE type = ? AND scope = ? ORDER BY id",
            (type, scope),
        )
        rows = await cur.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def get_by_scope_any_type(self, scope: str) -> list[dict[str, Any]]:
        """Return all entries for a given scope across all types."""
        cur = await self._db.execute(
            "SELECT id, type, scope, content, metadata, embedding, created_at, updated_at "
            "FROM memory_entries WHERE scope = ? ORDER BY id",
            (scope,),
        )
        rows = await cur.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def get_by_type(
        self, type: str, limit: int = 20, cursor: int | None = None
    ) -> dict[str, Any]:
        """Paginated list of entries by type, newest-first."""
        limit = max(1, min(limit, 500))
        if cursor:
            cur = await self._db.execute(
                "SELECT id, type, scope, content, metadata, embedding, created_at, updated_at "
                "FROM memory_entries WHERE type = ? AND id < ? ORDER BY id DESC LIMIT ?",
                (type, cursor, limit + 1),
            )
        else:
            cur = await self._db.execute(
                "SELECT id, type, scope, content, metadata, embedding, created_at, updated_at "
                "FROM memory_entries WHERE type = ? ORDER BY id DESC LIMIT ?",
                (type, limit + 1),
            )
        rows = await cur.fetchall()
        next_cursor: int | None = None
        if len(rows) > limit:
            rows = rows[:limit]
            next_cursor = rows[-1][0]
        entries = [self._row_to_dict(r) for r in rows]
        out: dict[str, Any] = {"entries": entries}
        if next_cursor:
            out["cursor"] = next_cursor
        return out

    async def list_entries(
        self, type: str | None = None, limit: int = 50, cursor: int | None = None
    ) -> dict[str, Any]:
        """Paginated list of entries, optionally filtered by type.

        Ordered newest-first (id DESC). Cursor is the last ID in the
        previous page; the next page fetches entries with id < cursor.
        """
        limit = max(1, min(limit, 500))
        if type is not None and type not in _VALID_TYPES:
            raise ValueError(f"Invalid memory type '{type}'")

        base_select = (
            "SELECT id, type, scope, content, metadata, embedding, created_at, updated_at "
            "FROM memory_entries"
        )
        if type:
            if cursor:
                sql = f"{base_select} WHERE type = ? AND id < ? ORDER BY id DESC LIMIT ?"
                params: tuple = (type, cursor, limit + 1)
            else:
                sql = f"{base_select} WHERE type = ? ORDER BY id DESC LIMIT ?"
                params = (type, limit + 1)
        else:
            if cursor:
                sql = f"{base_select} WHERE id < ? ORDER BY id DESC LIMIT ?"
                params = (cursor, limit + 1)
            else:
                sql = f"{base_select} ORDER BY id DESC LIMIT ?"
                params = (limit + 1,)

        cur = await self._db.execute(sql, params)
        rows = await cur.fetchall()
        next_cursor: int | None = None
        if len(rows) > limit:
            rows = rows[:limit]
            next_cursor = rows[-1][0]
        entries = [self._row_to_dict(r) for r in rows]
        out: dict[str, Any] = {"entries": entries}
        if next_cursor:
            out["cursor"] = next_cursor
        return out

    async def list_skills(self) -> list[dict[str, Any]]:
        """Return all skill entries with name + 80-char preview."""
        cur = await self._db.execute(
            "SELECT id, scope, content FROM memory_entries "
            "WHERE type = 'skill' ORDER BY scope",
        )
        rows = await cur.fetchall()
        skills = []
        for r in rows:
            name = r[1].replace("skill:", "", 1) if r[1].startswith("skill:") else r[1]
            content = r[2] or ""
            preview = content[:80].replace("\n", " ")
            skills.append({"id": r[0], "name": name, "scope": r[1], "preview": preview})
        return skills

    async def get_skill(self, name: str) -> dict[str, Any] | None:
        """Return the full content of a skill by name (scope = skill:<name>)."""
        scope = f"skill:{name}" if not name.startswith("skill:") else name
        cur = await self._db.execute(
            "SELECT id, type, scope, content, metadata, embedding, created_at, updated_at "
            "FROM memory_entries WHERE type = 'skill' AND scope = ?",
            (scope,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    async def update_embedding(self, id: int, embedding_bytes: bytes) -> None:
        """Update just the embedding BLOB for an entry."""
        embedding_bytes = self._validate_embedding(embedding_bytes)
        async with self._write_lock:
            await self._db.execute("BEGIN IMMEDIATE")
            try:
                await self._db.execute(
                    "UPDATE memory_entries SET embedding = ? WHERE id = ?",
                    (embedding_bytes, id),
                )
                await self._db.commit()
            except Exception:
                await self._db.rollback()
                raise

    async def get_all_embeddings(self) -> list[tuple[int, bytes]]:
        """Load all (id, embedding) pairs with valid embeddings.

        Filters by expected float32 BLOB size (dims * 4) to skip NULL
        embeddings and entries with mismatched dimensions (model changed).
        Returns [] if no valid embeddings exist.
        """
        expected_size = self._embedding_dimensions * 4
        cur = await self._db.execute(
            "SELECT id, embedding FROM memory_entries "
            "WHERE embedding IS NOT NULL AND length(embedding) = ?",
            (expected_size,),
        )
        rows = await cur.fetchall()
        return [(r[0], r[1]) for r in rows]

    async def get_entries_without_embeddings(self) -> list[dict[str, Any]]:
        """Return entries needing (re-)embedding (for backfill).

        Includes entries with NULL embeddings and entries whose embedding
        BLOB size doesn't match the current model dimensions (model changed).
        """
        expected_size = self._embedding_dimensions * 4
        cur = await self._db.execute(
            "SELECT id, type, scope, content, metadata, embedding, created_at, updated_at "
            "FROM memory_entries "
            "WHERE embedding IS NULL OR length(embedding) != ? ORDER BY id",
            (expected_size,),
        )
        rows = await cur.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def has_self_entry(self) -> bool:
        """Check if a 'self' entry already exists."""
        cur = await self._db.execute(
            "SELECT 1 FROM memory_entries WHERE type = 'self' LIMIT 1"
        )
        return await cur.fetchone() is not None

    @staticmethod
    def _row_to_dict(row: tuple) -> dict[str, Any]:
        """Convert a DB row to a dict. Embedding BLOB is excluded from
        the public dict to avoid leaking large binary data."""
        return {
            "id": row[0],
            "type": row[1],
            "scope": row[2],
            "content": row[3],
            "metadata": json.loads(row[4]) if row[4] else {},
            "createdAt": row[6],
            "updatedAt": row[7],
        }

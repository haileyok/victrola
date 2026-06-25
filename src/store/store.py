"""Local SQLite-backed persistence for the agent.

Three focused stores sharing one SQLite connection:
- DocumentStore: flat rkey -> content (notes, skills, custom tool defs)
- RecordStore: (collection, rkey) -> JSON (general-purpose namespaced records)
- ChatStore: sessions + append-only messages for chat transcripts
"""

import asyncio
import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


class StoreNotFound(Exception):
    """Raised when a get_* call finds no matching row.

    Message contains the substring "not found" so existing callers that
    check `"not found" in str(e).lower()` keep working unchanged.
    """


class StoreConflict(Exception):
    """Raised when a create call collides with an existing row
    (duplicate primary key / IntegrityError)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_rkey() -> str:
    # 13-char base32-ish token; collision-safe for our scale
    return secrets.token_hex(8)


class Store:
    """Owns the SQLite connection and exposes the three concept stores.

    A single asyncio.Lock serializes all write transactions across every
    sub-store to prevent "cannot start a transaction within a transaction"
    on the shared aiosqlite.Connection.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._db: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()
        self.documents: DocumentStore | None = None
        self.records: RecordStore | None = None
        self.chat: ChatStore | None = None
        self.memory: MemoryStore | None = None

    async def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._create_schema()
        await self._db.commit()
        self.documents = DocumentStore(self._db, self._write_lock)
        self.records = RecordStore(self._db, self._write_lock)
        self.chat = ChatStore(self._db, self._write_lock)
        from src.memory.store import MemoryStore

        self.memory = MemoryStore(
            self._db, fts5_available=self._fts5_available, write_lock=self._write_lock
        )
        logger.info("Store initialized at %s", self._path)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def _create_schema(self) -> None:
        if self._db is None:
            raise RuntimeError("Database connection not initialized")
        await self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS agent_documents (
                rkey TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS private_records (
                collection TEXT NOT NULL,
                rkey TEXT NOT NULL,
                record TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (collection, rkey)
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                rkey TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(rkey) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS chat_messages_by_session
                ON chat_messages(session_id, id);

            CREATE TABLE IF NOT EXISTS chat_compaction_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                compacted_up_to_msg_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(rkey) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                scope TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                embedding BLOB,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS memory_entries_by_type
                ON memory_entries(type);
            CREATE INDEX IF NOT EXISTS memory_entries_by_scope
                ON memory_entries(scope);
            CREATE INDEX IF NOT EXISTS memory_entries_by_type_scope
                ON memory_entries(type, scope);

            CREATE UNIQUE INDEX IF NOT EXISTS memory_self_singleton
                ON memory_entries(scope) WHERE type = 'self';

            CREATE TABLE IF NOT EXISTS memory_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

        # FTS5 is optional — runtime check with LIKE fallback
        self._fts5_available = False
        try:
            await self._db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memory_entries_fts "
                "USING fts5(content, content='memory_entries', content_rowid='id')"
            )
            await self._db.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS memory_fts_insert
                AFTER INSERT ON memory_entries BEGIN
                    INSERT INTO memory_entries_fts(rowid, content)
                    VALUES (new.id, new.content);
                END;

                CREATE TRIGGER IF NOT EXISTS memory_fts_delete
                AFTER DELETE ON memory_entries BEGIN
                    INSERT INTO memory_entries_fts(memory_entries_fts, rowid, content)
                    VALUES('delete', old.id, old.content);
                END;

                CREATE TRIGGER IF NOT EXISTS memory_fts_update
                AFTER UPDATE ON memory_entries BEGIN
                    INSERT INTO memory_entries_fts(memory_entries_fts, rowid, content)
                    VALUES('delete', old.id, old.content);
                    INSERT INTO memory_entries_fts(rowid, content)
                    VALUES (new.id, new.content);
                END;
                """
            )
            self._fts5_available = True
        except Exception as e:
            logger.warning(
                "FTS5 not available — falling back to LIKE keyword search: %s", e
            )
        await self._db.execute("PRAGMA foreign_keys=ON")

        # Add compacted_up_to_msg_id column to chat_sessions if it doesn't
        # exist (SQLite doesn't support ADD COLUMN IF NOT EXISTS). Existing
        # rows get NULL, meaning "no compaction yet."
        cur = await self._db.execute("PRAGMA table_info(chat_sessions)")
        columns = [row[1] for row in await cur.fetchall()]
        if "compacted_up_to_msg_id" not in columns:
            await self._db.execute(
                "ALTER TABLE chat_sessions ADD COLUMN compacted_up_to_msg_id INTEGER DEFAULT NULL"
            )

class DocumentStore:
    """Agent documents: flat rkey -> content.

    Used for notes (`self`, `operator`, `skill:*`, `task:*`, free-form) and
    custom tool definitions (`customtool:*` entries).
    """

    def __init__(self, db: aiosqlite.Connection, write_lock: asyncio.Lock | None = None) -> None:
        self._db = db
        self._write_lock = write_lock or asyncio.Lock()

    async def create(self, rkey: str, content: str) -> dict[str, Any]:
        now = _now()
        async with self._write_lock:
            try:
                await self._db.execute("BEGIN IMMEDIATE")
                await self._db.execute(
                    "INSERT INTO agent_documents (rkey, content, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (rkey, content, now, now),
                )
                await self._db.commit()
            except aiosqlite.IntegrityError as e:
                await self._db.rollback()
                raise StoreConflict(f"agent_document '{rkey}' already exists") from e
            except Exception:
                await self._db.rollback()
                raise
        return {"rkey": rkey, "content": content, "createdAt": now, "updatedAt": now}

    async def update(self, rkey: str, content: str) -> dict[str, Any]:
        now = _now()
        async with self._write_lock:
            try:
                await self._db.execute("BEGIN IMMEDIATE")
                cur = await self._db.execute(
                    "UPDATE agent_documents SET content = ?, updated_at = ? WHERE rkey = ?",
                    (content, now, rkey),
                )
                if cur.rowcount == 0:
                    await self._db.rollback()
                    raise StoreNotFound(f"agent_document '{rkey}' not found")
                await self._db.commit()
            except StoreNotFound:
                raise
            except Exception:
                await self._db.rollback()
                raise
        return {"rkey": rkey, "content": content, "updatedAt": now}

    async def get(self, rkey: str) -> dict[str, Any]:
        cur = await self._db.execute(
            "SELECT rkey, content, created_at, updated_at FROM agent_documents WHERE rkey = ?",
            (rkey,),
        )
        row = await cur.fetchone()
        if row is None:
            raise StoreNotFound(f"agent_document '{rkey}' not found")
        return {
            "rkey": row[0],
            "content": row[1],
            "createdAt": row[2],
            "updatedAt": row[3],
        }

    async def list(
        self, limit: int = 20, cursor: str | None = None
    ) -> dict[str, Any]:
        limit = max(1, min(limit, 500))
        if cursor:
            cur = await self._db.execute(
                "SELECT rkey, content, created_at, updated_at FROM agent_documents "
                "WHERE rkey > ? ORDER BY rkey LIMIT ?",
                (cursor, limit + 1),
            )
        else:
            cur = await self._db.execute(
                "SELECT rkey, content, created_at, updated_at FROM agent_documents "
                "ORDER BY rkey LIMIT ?",
                (limit + 1,),
            )
        rows = await cur.fetchall()
        next_cursor: str | None = None
        if len(rows) > limit:
            rows = rows[:limit]
            next_cursor = rows[-1][0]
        docs = [
            {
                "rkey": r[0],
                "content": r[1],
                "createdAt": r[2],
                "updatedAt": r[3],
            }
            for r in rows
        ]
        out: dict[str, Any] = {"documents": docs}
        if next_cursor:
            out["cursor"] = next_cursor
        return out

    async def delete(self, rkey: str) -> None:
        async with self._write_lock:
            try:
                await self._db.execute("BEGIN IMMEDIATE")
                cur = await self._db.execute(
                    "DELETE FROM agent_documents WHERE rkey = ?", (rkey,)
                )
                if cur.rowcount == 0:
                    await self._db.rollback()
                    raise StoreNotFound(f"agent_document '{rkey}' not found")
                await self._db.commit()
            except StoreNotFound:
                raise
            except Exception:
                await self._db.rollback()
                raise


class RecordStore:
    """Private records: (collection, rkey) -> JSON record.

    Used for user tracking data.
    """

    def __init__(self, db: aiosqlite.Connection, write_lock: asyncio.Lock | None = None) -> None:
        self._db = db
        self._write_lock = write_lock or asyncio.Lock()

    async def create(
        self, collection: str, rkey: str, record: dict[str, Any]
    ) -> dict[str, Any]:
        now = _now()
        payload = json.dumps(record)
        async with self._write_lock:
            try:
                await self._db.execute("BEGIN IMMEDIATE")
                await self._db.execute(
                    "INSERT INTO private_records "
                    "(collection, rkey, record, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (collection, rkey, payload, now, now),
                )
                await self._db.commit()
            except aiosqlite.IntegrityError as e:
                await self._db.rollback()
                raise StoreConflict(
                    f"private_record '{collection}/{rkey}' already exists"
                ) from e
            except Exception:
                await self._db.rollback()
                raise
        return {"value": record}

    async def update(
        self, collection: str, rkey: str, record: dict[str, Any]
    ) -> dict[str, Any]:
        now = _now()
        payload = json.dumps(record)
        async with self._write_lock:
            try:
                await self._db.execute("BEGIN IMMEDIATE")
                cur = await self._db.execute(
                    "UPDATE private_records SET record = ?, updated_at = ? "
                    "WHERE collection = ? AND rkey = ?",
                    (payload, now, collection, rkey),
                )
                if cur.rowcount == 0:
                    await self._db.rollback()
                    raise StoreNotFound(f"private_record '{collection}/{rkey}' not found")
                await self._db.commit()
            except StoreNotFound:
                raise
            except Exception:
                await self._db.rollback()
                raise
        return {"value": record}

    async def get(self, collection: str, rkey: str) -> dict[str, Any]:
        cur = await self._db.execute(
            "SELECT record FROM private_records WHERE collection = ? AND rkey = ?",
            (collection, rkey),
        )
        row = await cur.fetchone()
        if row is None:
            raise StoreNotFound(f"private_record '{collection}/{rkey}' not found")
        return {"value": json.loads(row[0])}

    async def list(
        self, collection: str, limit: int = 20, cursor: str | None = None
    ) -> dict[str, Any]:
        limit = max(1, min(limit, 500))
        if cursor:
            cur = await self._db.execute(
                "SELECT rkey, record, created_at, updated_at FROM private_records "
                "WHERE collection = ? AND rkey > ? ORDER BY rkey LIMIT ?",
                (collection, cursor, limit + 1),
            )
        else:
            cur = await self._db.execute(
                "SELECT rkey, record, created_at, updated_at FROM private_records "
                "WHERE collection = ? ORDER BY rkey LIMIT ?",
                (collection, limit + 1),
            )
        rows = await cur.fetchall()
        next_cursor: str | None = None
        if len(rows) > limit:
            rows = rows[:limit]
            next_cursor = rows[-1][0]
        records = [
            {
                "rkey": r[0],
                "value": json.loads(r[1]),
                "createdAt": r[2],
                "updatedAt": r[3],
            }
            for r in rows
        ]
        out: dict[str, Any] = {"records": records}
        if next_cursor:
            out["cursor"] = next_cursor
        return out

    async def delete(self, collection: str, rkey: str) -> None:
        async with self._write_lock:
            try:
                await self._db.execute("BEGIN IMMEDIATE")
                cur = await self._db.execute(
                    "DELETE FROM private_records WHERE collection = ? AND rkey = ?",
                    (collection, rkey),
                )
                if cur.rowcount == 0:
                    await self._db.rollback()
                    raise StoreNotFound(f"private_record '{collection}/{rkey}' not found")
                await self._db.commit()
            except StoreNotFound:
                raise
            except Exception:
                await self._db.rollback()
                raise


class ChatStore:
    """Agent chat: sessions and their append-only messages.

    Used for persisted chat transcripts across conversations.
    """

    def __init__(self, db: aiosqlite.Connection, write_lock: asyncio.Lock | None = None) -> None:
        self._db = db
        self._write_lock = write_lock or asyncio.Lock()

    async def create_session(
        self, title: str = "", rkey: str | None = None
    ) -> dict[str, Any]:
        if rkey is None:
            rkey = _new_rkey()
        now = _now()
        async with self._write_lock:
            try:
                await self._db.execute("BEGIN IMMEDIATE")
                await self._db.execute(
                    "INSERT INTO chat_sessions (rkey, title, created_at) VALUES (?, ?, ?)",
                    (rkey, title, now),
                )
                await self._db.commit()
            except Exception:
                await self._db.rollback()
                raise
        return {"rkey": rkey, "title": title, "createdAt": now}

    async def get_session(self, rkey: str) -> dict[str, Any] | None:
        cur = await self._db.execute(
            "SELECT rkey, title, created_at FROM chat_sessions WHERE rkey = ?",
            (rkey,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return {"rkey": row[0], "title": row[1], "createdAt": row[2]}

    async def set_title(self, rkey: str, title: str) -> None:
        """Update a session's title."""
        async with self._write_lock:
            try:
                await self._db.execute("BEGIN IMMEDIATE")
                await self._db.execute(
                    "UPDATE chat_sessions SET title = ? WHERE rkey = ?",
                    (title, rkey),
                )
                await self._db.commit()
            except Exception:
                await self._db.rollback()
                raise

    async def ensure_session(
        self, rkey: str, title: str = ""
    ) -> dict[str, Any]:
        """Create a session with the given rkey if it doesn't exist; idempotent."""
        now = _now()
        async with self._write_lock:
            try:
                await self._db.execute("BEGIN IMMEDIATE")
                await self._db.execute(
                    "INSERT OR IGNORE INTO chat_sessions (rkey, title, created_at) "
                    "VALUES (?, ?, ?)",
                    (rkey, title, now),
                )
                await self._db.commit()
            except Exception:
                await self._db.rollback()
                raise
        existing = await self.get_session(rkey)
        if existing is None:
            raise RuntimeError("ensure_session failed: session not found after create")
        return existing

    async def list_sessions(
        self, limit: int = 10, cursor: str | None = None
    ) -> dict[str, Any]:
        limit = max(1, min(limit, 500))
        if cursor:
            cur = await self._db.execute(
                "SELECT rkey, title, created_at FROM chat_sessions "
                "WHERE created_at < ? ORDER BY created_at DESC LIMIT ?",
                (cursor, limit + 1),
            )
        else:
            cur = await self._db.execute(
                "SELECT rkey, title, created_at FROM chat_sessions "
                "ORDER BY created_at DESC LIMIT ?",
                (limit + 1,),
            )
        rows = await cur.fetchall()
        next_cursor: str | None = None
        if len(rows) > limit:
            rows = rows[:limit]
            next_cursor = rows[-1][2]
        sessions = [
            {
                "rkey": r[0],
                "title": r[1],
                "createdAt": r[2],
            }
            for r in rows
        ]
        out: dict[str, Any] = {"sessions": sessions}
        if next_cursor:
            out["cursor"] = next_cursor
        return out

    async def delete_session(self, rkey: str) -> None:
        # cascade-deletes messages via FK
        async with self._write_lock:
            try:
                await self._db.execute("BEGIN IMMEDIATE")
                cur = await self._db.execute(
                    "DELETE FROM chat_sessions WHERE rkey = ?", (rkey,)
                )
                if cur.rowcount == 0:
                    await self._db.rollback()
                    raise StoreNotFound(f"chat_session '{rkey}' not found")
                await self._db.commit()
            except StoreNotFound:
                raise
            except Exception:
                await self._db.rollback()
                raise

    async def create_message(
        self, session_id: str, sender: str, content: str
    ) -> dict[str, Any]:
        now = _now()
        async with self._write_lock:
            try:
                await self._db.execute("BEGIN IMMEDIATE")
                cur = await self._db.execute(
                    "INSERT INTO chat_messages (session_id, sender, content, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (session_id, sender, content, now),
                )
                await self._db.commit()
            except Exception:
                await self._db.rollback()
                raise
        return {
            "id": cur.lastrowid,
            "sessionId": session_id,
            "sender": sender,
            "content": content,
            "createdAt": now,
        }

    async def list_messages(
        self, session_id: str, limit: int = 10, cursor: str | None = None, after_id: int = 0
    ) -> dict[str, Any]:
        limit = max(1, min(limit, 500))
        cursor_id = int(cursor) if cursor else 0
        min_id = max(cursor_id, after_id)
        cur = await self._db.execute(
            "SELECT id, sender, content, created_at FROM chat_messages "
            "WHERE session_id = ? AND id > ? ORDER BY id LIMIT ?",
            (session_id, min_id, limit + 1),
        )
        rows = await cur.fetchall()
        next_cursor: str | None = None
        if len(rows) > limit:
            rows = rows[:limit]
            next_cursor = str(rows[-1][0])
        messages = [
            {
                "id": r[0],
                "sender": r[1],
                "content": r[2],
                "createdAt": r[3],
            }
            for r in rows
        ]
        out: dict[str, Any] = {"messages": messages}
        if next_cursor:
            out["cursor"] = next_cursor
        return out

    async def get_compaction_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        """Return the latest compaction summary and checkpoint ID for a session.

        Returns None when no compaction has been recorded yet. Selects the
        summary row whose ``compacted_up_to_msg_id`` matches the session's
        checkpoint column, rather than blindly trusting ``ORDER BY id DESC``,
        to avoid pairing the checkpoint with a stale/wrong summary row.
        """
        cur = await self._db.execute(
            "SELECT compacted_up_to_msg_id FROM chat_sessions WHERE rkey = ?",
            (session_id,),
        )
        row = await cur.fetchone()
        if not row or row[0] is None:
            return None
        checkpoint_id = row[0]
        cur = await self._db.execute(
            "SELECT summary, compacted_up_to_msg_id, created_at "
            "FROM chat_compaction_summaries "
            "WHERE session_id = ? AND compacted_up_to_msg_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (session_id, checkpoint_id),
        )
        row = await cur.fetchone()
        if not row:
            return None
        return {
            "summary": row[0],
            "compacted_up_to_msg_id": row[1],
            "created_at": row[2],
        }

    async def set_compaction_checkpoint(
        self, session_id: str, compacted_up_to_msg_id: int, summary: str
    ) -> None:
        """Persist a compaction summary and advance the session's checkpoint.

        Inserts a new row in chat_compaction_summaries and updates the
        compacted_up_to_msg_id column on chat_sessions in a single
        transaction. Raw messages are not deleted — they remain in
        chat_messages for auditability but are skipped on load.

        The checkpoint only advances forward: the UPDATE has a monotonic
        guard so a regressed write is ignored. The summary row is only
        inserted when the UPDATE actually advances the checkpoint,
        preventing stale summaries from becoming the effective checkpoint
        via ``ORDER BY id DESC`` in get_compaction_checkpoint.
        """
        now = _now()
        async with self._write_lock:
            try:
                await self._db.execute("BEGIN IMMEDIATE")
                cur = await self._db.execute(
                    "UPDATE chat_sessions SET compacted_up_to_msg_id = ? "
                    "WHERE rkey = ? "
                    "AND (compacted_up_to_msg_id IS NULL OR compacted_up_to_msg_id < ?)",
                    (compacted_up_to_msg_id, session_id, compacted_up_to_msg_id),
                )
                if cur.rowcount > 0:
                    await self._db.execute(
                        "INSERT INTO chat_compaction_summaries "
                        "(session_id, summary, compacted_up_to_msg_id, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (session_id, summary, compacted_up_to_msg_id, now),
                    )
                await self._db.commit()
            except Exception:
                await self._db.rollback()
                raise

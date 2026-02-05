"""Local SQLite-backed persistence for the agent.

Three focused stores sharing one SQLite connection:
- DocumentStore: flat rkey -> content (notes, skills, custom tool defs)
- RecordStore: (collection, rkey) -> JSON (general-purpose namespaced records)
- ChatStore: sessions + append-only messages for TUI transcripts
"""

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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_rkey() -> str:
    # 13-char base32-ish token; collision-safe for our scale
    return secrets.token_hex(8)


class Store:
    """Owns the SQLite connection and exposes the three concept stores."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._db: aiosqlite.Connection | None = None
        self.documents: DocumentStore | None = None
        self.records: RecordStore | None = None
        self.chat: ChatStore | None = None

    async def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._create_schema()
        await self._db.commit()
        self.documents = DocumentStore(self._db)
        self.records = RecordStore(self._db)
        self.chat = ChatStore(self._db)
        logger.info("Store initialized at %s", self._path)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def _create_schema(self) -> None:
        assert self._db is not None
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
            """
        )


class DocumentStore:
    """Agent documents: flat rkey -> content.

    Used for notes (`self`, `operator`, `skill:*`, `task:*`, free-form) and
    custom tool definitions (`customtool:*` entries).
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def create(self, rkey: str, content: str) -> dict[str, Any]:
        now = _now()
        try:
            await self._db.execute(
                "INSERT INTO agent_documents (rkey, content, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (rkey, content, now, now),
            )
            await self._db.commit()
        except aiosqlite.IntegrityError as e:
            raise StoreNotFound(f"agent_document '{rkey}' already exists") from e
        return {"rkey": rkey, "content": content, "createdAt": now, "updatedAt": now}

    async def update(self, rkey: str, content: str) -> dict[str, Any]:
        now = _now()
        cur = await self._db.execute(
            "UPDATE agent_documents SET content = ?, updated_at = ? WHERE rkey = ?",
            (content, now, rkey),
        )
        if cur.rowcount == 0:
            raise StoreNotFound(f"agent_document '{rkey}' not found")
        await self._db.commit()
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
        cur = await self._db.execute(
            "DELETE FROM agent_documents WHERE rkey = ?", (rkey,)
        )
        if cur.rowcount == 0:
            raise StoreNotFound(f"agent_document '{rkey}' not found")
        await self._db.commit()


class RecordStore:
    """Private records: (collection, rkey) -> JSON record.

    Used for user tracking data.
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def create(
        self, collection: str, rkey: str, record: dict[str, Any]
    ) -> dict[str, Any]:
        now = _now()
        payload = json.dumps(record)
        try:
            await self._db.execute(
                "INSERT INTO private_records "
                "(collection, rkey, record, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (collection, rkey, payload, now, now),
            )
            await self._db.commit()
        except aiosqlite.IntegrityError as e:
            raise StoreNotFound(
                f"private_record '{collection}/{rkey}' already exists"
            ) from e
        return {"value": record}

    async def update(
        self, collection: str, rkey: str, record: dict[str, Any]
    ) -> dict[str, Any]:
        now = _now()
        payload = json.dumps(record)
        cur = await self._db.execute(
            "UPDATE private_records SET record = ?, updated_at = ? "
            "WHERE collection = ? AND rkey = ?",
            (payload, now, collection, rkey),
        )
        if cur.rowcount == 0:
            raise StoreNotFound(f"private_record '{collection}/{rkey}' not found")
        await self._db.commit()
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
        cur = await self._db.execute(
            "DELETE FROM private_records WHERE collection = ? AND rkey = ?",
            (collection, rkey),
        )
        if cur.rowcount == 0:
            raise StoreNotFound(f"private_record '{collection}/{rkey}' not found")
        await self._db.commit()


class ChatStore:
    """Agent chat: sessions and their append-only messages.

    Used for persisted TUI transcripts across conversations.
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def create_session(
        self, title: str = "", rkey: str | None = None
    ) -> dict[str, Any]:
        if rkey is None:
            rkey = _new_rkey()
        now = _now()
        await self._db.execute(
            "INSERT INTO chat_sessions (rkey, title, created_at) VALUES (?, ?, ?)",
            (rkey, title, now),
        )
        await self._db.commit()
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
        await self._db.execute(
            "UPDATE chat_sessions SET title = ? WHERE rkey = ?",
            (title, rkey),
        )
        await self._db.commit()

    async def ensure_session(
        self, rkey: str, title: str = ""
    ) -> dict[str, Any]:
        """Create a session with the given rkey if it doesn't exist; idempotent."""
        now = _now()
        await self._db.execute(
            "INSERT OR IGNORE INTO chat_sessions (rkey, title, created_at) "
            "VALUES (?, ?, ?)",
            (rkey, title, now),
        )
        await self._db.commit()
        existing = await self.get_session(rkey)
        assert existing is not None
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
        cur = await self._db.execute(
            "DELETE FROM chat_sessions WHERE rkey = ?", (rkey,)
        )
        if cur.rowcount == 0:
            raise StoreNotFound(f"chat_session '{rkey}' not found")
        await self._db.commit()

    async def create_message(
        self, session_id: str, sender: str, content: str
    ) -> dict[str, Any]:
        now = _now()
        cur = await self._db.execute(
            "INSERT INTO chat_messages (session_id, sender, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, sender, content, now),
        )
        await self._db.commit()
        return {
            "id": cur.lastrowid,
            "sessionId": session_id,
            "sender": sender,
            "content": content,
            "createdAt": now,
        }

    async def list_messages(
        self, session_id: str, limit: int = 10, cursor: str | None = None
    ) -> dict[str, Any]:
        limit = max(1, min(limit, 500))
        cursor_id = int(cursor) if cursor else 0
        cur = await self._db.execute(
            "SELECT id, sender, content, created_at FROM chat_messages "
            "WHERE session_id = ? AND id > ? ORDER BY id LIMIT ?",
            (session_id, cursor_id, limit + 1),
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

"""Penny's long-term memory in an Engram space on her own ATProto account.

Episodic and factual entries are canonical in the space (see
https://github.com/haileyok/engram-garden). The local `memory_entries` table
mirrors them: it keeps the integer ids the memory tools use, powers keyword
search and the web UI, and holds writes that couldn't be pushed yet. The agent
never sees any of this; it keeps calling memory.add / search / update / delete.

How an entry maps to a space record:
  - text      <- content
  - tags      <- `type:<type>`, `scope:<scope>`, then the entry's own tags
  - source    <- the full scope (the scope tag is cut to the tag size limit)
  - createdAt <- the entry's created_at

Engram has no update, so changing an entry writes the new record and then
deletes the old one; the local id stays the same.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import engram_garden as eg
from engram_garden import EngramError

from src.config import CONFIG
from src.memory.store import ENGRAM_TYPES

logger = logging.getLogger(__name__)

TYPE_PREFIX = "type:"
SCOPE_PREFIX = "scope:"
#: Engram allows 16 tags per memory; two are ours.
MAX_USER_TAGS = 14
#: Engram's per-tag limit, in bytes.
MAX_TAG_BYTES = 128
#: Engram's per-memory limit is 30000 bytes; leave room for the 'text' trim.
MAX_TEXT_BYTES = 29000
#: How long a write waits for the space before it's left for the sync to retry.
PUSH_TIMEOUT = 30.0
#: Stop a flush after this many failures in a row (the space is probably down).
MAX_CONSECUTIVE_FAILURES = 3
#: A search of the space gives up after this long, and after any failure the
#: space isn't searched for SEARCH_BACKOFF seconds (recall falls back to local
#: vectors), so a down or not-yet-approved appview costs one slow turn, not all.
SEARCH_TIMEOUT = 10.0
SEARCH_BACKOFF = 60.0
SEARCH_PAUSED = "memory space search paused"

SPACE_CONFIG_KEY = "engram_space_uri"
DEFAULT_SPACE_NAME = "memory"


def _clip_bytes(s: str, n: int) -> str:
    b = s.encode()
    return s if len(b) <= n else b[:n].decode(errors="ignore")


def tags_for(entry_type: str, scope: str, user_tags: list[str]) -> list[str]:
    """The space tags for an entry."""
    tags = [TYPE_PREFIX + entry_type, _clip_bytes(SCOPE_PREFIX + scope, MAX_TAG_BYTES)]
    seen = set(tags)
    for t in user_tags:
        t = _clip_bytes(str(t).strip(), MAX_TAG_BYTES)
        if t and t not in seen:
            seen.add(t)
            tags.append(t)
    if len(tags) > 2 + MAX_USER_TAGS:
        logger.warning("memory has %d tags; the space keeps %d", len(tags) - 2, MAX_USER_TAGS)
        tags = tags[: 2 + MAX_USER_TAGS]
    return tags


def parse_tags(m: eg.Memory) -> tuple[str, str, list[str]]:
    """(type, scope, user tags) of a memory read from the space."""
    entry_type, scope_tag, user = "", "", []
    for t in m.tags:
        if t.startswith(TYPE_PREFIX) and not entry_type:
            entry_type = t[len(TYPE_PREFIX) :]
        elif t.startswith(SCOPE_PREFIX) and not scope_tag:
            scope_tag = t[len(SCOPE_PREFIX) :]
        else:
            user.append(t)
    if entry_type not in ENGRAM_TYPES:
        entry_type = "factual"
    return entry_type, m.source or scope_tag or "imported", user


def _parse_time(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


class EngramSync:
    """Keeps the local mirror and the memory space in step."""

    def __init__(self, memory_store: Any, spaces: eg.Spaces, space: str) -> None:
        self._ms = memory_store
        self.spaces = spaces
        self.space = space
        self._lock = asyncio.Lock()
        self.last_error = ""
        self._search_blocked_until = 0.0
        self._search_block_reason = ""

    @property
    def did(self) -> str:
        return self.spaces.client.did

    async def aclose(self) -> None:
        await self.spaces.aclose()

    # -- writes (called by MemoryStore after its own commit) --

    async def after_add(self, id: int) -> None:
        await self._guarded(self._push_id(id), f"push memory {id}")

    async def after_update(self, id: int) -> None:
        await self._guarded(self._replace(id), f"update memory {id}")

    async def after_delete(self, uri: str) -> None:
        await self._guarded(self._forget_or_tombstone(uri), "delete from the memory space")

    async def _guarded(self, coro: Any, what: str) -> None:
        """Run a write to the space; on failure leave it for the sync to retry."""
        try:
            await asyncio.wait_for(coro, PUSH_TIMEOUT)
        except Exception as e:  # noqa: BLE001 - a failed push never fails the memory write
            self.last_error = f"{what}: {e}"
            logger.warning("Engram: couldn't %s (will retry): %s", what, e)

    async def _remember(self, entry: dict[str, Any]) -> str:
        meta = entry.get("metadata") or {}
        text = entry["content"]
        if len(text.encode()) > MAX_TEXT_BYTES:
            logger.warning("memory %s is over the space's size limit; storing its first %d bytes there", entry["id"], MAX_TEXT_BYTES)
            text = _clip_bytes(text, MAX_TEXT_BYTES)
        out = await self.spaces.remember(
            text,
            tags=tags_for(entry["type"], entry["scope"], list(meta.get("tags") or [])),
            source=entry["scope"][:2000],
            space=self.space,
            created_at=_parse_time(entry.get("createdAt", "")),
        )
        if out.note:
            logger.warning("Engram: %s", out.note)
        return out.uri

    async def _push_id(self, id: int) -> None:
        async with self._lock:
            if await self._ms.get_engram_uri(id):
                return
            entry = await self._ms.get_entry(id)
            if entry is None or entry["type"] not in ENGRAM_TYPES:
                return
            uri = await self._remember(entry)
            await self._ms.set_engram_uri(id, uri)

    async def _replace(self, id: int) -> None:
        """The entry changed: write the new record, then delete the old one."""
        async with self._lock:
            old = await self._ms.get_engram_uri(id)
            if old is None:
                return  # not pushed yet; the sync pushes its latest content
            entry = await self._ms.get_entry(id)
            if entry is None:
                return
            try:
                uri = await self._remember(entry)
            except Exception:
                # Leave the entry unpushed, and the old record to be deleted.
                await self._ms.set_engram_uri(id, None)
                await self._ms.add_tombstone(old)
                raise
            await self._ms.set_engram_uri(id, uri)
        await self._forget_or_tombstone(old)

    async def _forget_or_tombstone(self, uri: str) -> None:
        try:
            await self.spaces.forget(uri)
        except EngramError as e:
            if "not found" in str(e).lower() or "RecordNotFound" in str(e):
                return
            await self._ms.add_tombstone(uri)
            raise
        except Exception:
            await self._ms.add_tombstone(uri)
            raise

    # -- catching up --

    async def flush(self, progress: Callable[[str], None] | None = None) -> dict[str, int]:
        """Retry deletes, then push every unpushed entry. Stops early if the space keeps failing."""
        say = progress or (lambda _msg: None)
        done = {"deleted": 0, "pushed": 0, "failed": 0}
        failures = 0
        for uri in await self._ms.list_tombstones():
            try:
                async with self._lock:
                    await self.spaces.forget(uri)
                await self._ms.clear_tombstone(uri)
                done["deleted"] += 1
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                if "not found" in msg or "recordnotfound" in msg:
                    await self._ms.clear_tombstone(uri)
                    continue
                failures += 1
                done["failed"] += 1
                logger.warning("Engram: couldn't delete %s yet: %s", uri, e)
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    return done
        total = await self._ms.count_unpushed()
        for id in await self._ms.unpushed_ids():
            try:
                await asyncio.wait_for(self._push_id(id), PUSH_TIMEOUT)
                done["pushed"] += 1
                failures = 0
                if done["pushed"] % 25 == 0:
                    say(f"pushed {done['pushed']} of {total}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                done["failed"] += 1
                self.last_error = f"push memory {id}: {e}"
                logger.warning("Engram: couldn't push memory %s: %s", id, e)
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    break
        return done

    async def pull(self) -> int:
        """Add local rows for memories in the space that this machine doesn't have
        (written elsewhere, or after a lost database). Nothing is ever deleted
        here. Returns how many were added."""
        known = await self._ms.known_engram_uris()
        added, cursor = 0, ""
        while True:
            page = await self.spaces.list(limit=100, cursor=cursor, author=self.did, space=self.space)
            for m in page.memories:
                if m.uri not in known and await self._ingest(m):
                    added += 1
            cursor = page.cursor
            if not cursor or not page.memories:
                return added

    async def _ingest(self, m: eg.Memory) -> int | None:
        entry_type, scope, tags = parse_tags(m)
        return await self._ms.ingest_remote(
            type=entry_type,
            scope=scope,
            content=m.text,
            metadata={"tags": tags} if tags else {},
            created_at=m.created_at,
            engram_uri=m.uri,
        )

    async def run(self, interval: float) -> None:
        """Flush and pull now and then, until cancelled."""
        while True:
            try:
                await self.flush()
                added = await self.pull()
                if added:
                    logger.info("Engram: added %d memories from the space", added)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self.last_error = str(e)
                if "UnknownSpace" in str(e):
                    # Not an error to repeat: the space's owner hasn't approved the appview yet.
                    logger.info("Engram: the appview doesn't index the memory space yet (run engram-setup for the approval link)")
                else:
                    logger.warning("Engram sync failed: %s", e)
            await asyncio.sleep(interval)

    # -- search --

    async def semantic_search(
        self, query: str, limit: int, scope: str | None = None
    ) -> list[dict[str, Any]]:
        """Vector search in the space, as [{id, score}] with local ids (best first).
        Memories the space has but this machine doesn't are added locally first."""
        if time.monotonic() < self._search_blocked_until:
            raise EngramError(f"{SEARCH_PAUSED} ({self._search_block_reason})")
        tags = [_clip_bytes(SCOPE_PREFIX + scope, MAX_TAG_BYTES)] if scope else None
        try:
            found = await asyncio.wait_for(
                self.spaces.recall(query, limit=min(max(limit, 1), 50), tags=tags, space=self.space),
                SEARCH_TIMEOUT,
            )
        except Exception as e:  # noqa: BLE001
            self._search_blocked_until = time.monotonic() + SEARCH_BACKOFF
            self._search_block_reason = str(e) or type(e).__name__
            raise
        ids = await self._ms.ids_by_engram_uri([m.uri for m in found.memories])
        out: list[dict[str, Any]] = []
        for m in found.memories:
            id = ids.get(m.uri)
            if id is None:
                id = await self._ingest(m)
            if id is not None:
                out.append({"id": id, "score": (m.similarity or 0) / 1000.0})
        return out


# -- wiring --


async def space_uri(store: Any) -> str:
    """The configured memory space: ENGRAM_SPACE_URI, else the one `engram-setup` saved."""
    if CONFIG.engram_space_uri:
        return CONFIG.engram_space_uri
    cur = await store._db.execute("SELECT value FROM memory_config WHERE key = ?", (SPACE_CONFIG_KEY,))
    row = await cur.fetchone()
    return row[0] if row else ""


def engram_settings(secret_manager: Any, space: str = "") -> eg.Settings:
    """Settings for the agent's account (the ATProto secrets) and local embedding model."""

    def secret(name: str) -> str:
        return (secret_manager.get_secret(name) or "") if secret_manager is not None else ""

    ident, password = secret("ATPROTO_IDENTIFIER"), secret("ATPROTO_APP_PASSWORD")
    if not ident or not password:
        raise eg.SettingsError("ATPROTO_IDENTIFIER and ATPROTO_APP_PASSWORD secrets are needed for the memory space")
    s = eg.Settings(
        appview_url=CONFIG.engram_appview_url,
        account=eg.Account(
            handle=ident,
            sign_in=eg.SIGN_IN_PASSWORD,
            password=password,
            pds_host=secret("ATPROTO_PDS_URL"),
        ),
        embed=eg.Embed(
            url=CONFIG.embedding_endpoint.rstrip("/") + "/v1",
            model=CONFIG.embedding_model,
        ),
    )
    if space:
        s.add_space(space, DEFAULT_SPACE_NAME)
    s.apply_defaults()
    return s


async def open_sync(store: Any, secret_manager: Any) -> EngramSync | None:
    """Sign in and attach the memory space to the store. None when no space is configured."""
    uri = await space_uri(store)
    if not uri:
        return None
    spaces = await eg.open_spaces(engram_settings(secret_manager, uri))
    sync = EngramSync(store.memory, spaces, DEFAULT_SPACE_NAME)
    store.memory.set_engram(sync)
    return sync


async def save_space_uri(store: Any, uri: str) -> None:
    await store._db.execute(
        "INSERT INTO memory_config (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (SPACE_CONFIG_KEY, uri),
    )
    await store._db.commit()


async def setup_space(store: Any, secret_manager: Any, name: str = DEFAULT_SPACE_NAME) -> dict[str, Any]:
    """Create the agent's memory space on her account (once), declare the local
    embedding model, and remember the space. Safe to run again. Returns what was
    done and, if the appview can't read the space yet, the link her account's
    owner opens (signed in as her) to allow it."""
    spaces = await eg.open_spaces(engram_settings(secret_manager))
    try:
        me = spaces.client.did
        uri = f"at://{me}/space/{eg.SPACE_TYPE}/{name}"
        existing = {s.uri for s in (await spaces.list_spaces()).spaces}
        created = uri not in existing
        if created:
            made = await spaces.create_space(name, model=CONFIG.embedding_model)
            uri = made.space.uri
        else:
            spaces.settings.add_space(uri, name)
        declared = False
        try:
            model = (await spaces.model(space=uri)).model
        except EngramError:
            model = (await spaces.set_model(action=eg.lex.DECLARE, space=uri, model=CONFIG.embedding_model)).model
            declared = True
        await save_space_uri(store, uri)
        idx = await spaces.index_space(space=uri)
        return {
            "uri": uri,
            "account": me,
            "created": created,
            "model_declared_now": declared,
            "model": model,
            "indexing": idx.state,
            "grant_link": idx.link,
        }
    finally:
        await spaces.aclose()

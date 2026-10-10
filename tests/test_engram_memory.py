"""Memory in the agent's Engram space: write-through, retries, and search.

The space is replaced by a small in-process stand-in with the same surface as
engram_garden.Spaces, so these test Victrola's side (mapping, queueing, id
stability, search merging), not the network.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime

import engram_garden as eg
import numpy as np
import pytest

from src.memory.engram import MAX_TEXT_BYTES, EngramSync, parse_tags, tags_for
from src.memory.search import SearchEngine
from src.store.store import Store

ME = "did:plc:penny"
SPACE = "memory"


class FakeSpaces:
    """Just enough of engram_garden.Spaces."""

    def __init__(self) -> None:
        self.records: dict[str, dict] = {}
        self.fail = False
        self.fail_forget = False
        self.calls: list[str] = []
        self.recall_modes: list[str] = []
        self._n = 0
        self.client = type("C", (), {"did": ME})()

    def _check(self, fail: bool) -> None:
        if fail:
            raise eg.EngramError("the space is down")

    async def remember(self, text, tags=None, source="", space="", created_at=None):
        self.calls.append("remember")
        self._check(self.fail)
        self._n += 1
        uri = f"at://{ME}/space/garden.engram.space/memory/{ME}/garden.engram.memory/r{self._n:04d}"
        self.records[uri] = {
            "text": text.strip(),
            "tags": list(tags or []),
            "source": source,
            "created_at": created_at,
            "author": ME,
        }
        return eg.RememberOut(uri=uri, cid="bafy")

    async def forget(self, uri):
        self.calls.append("forget")
        self._check(self.fail or self.fail_forget)
        if uri not in self.records:
            raise eg.EngramError("deleting memory: 404 RecordNotFound: no such record")
        del self.records[uri]
        return eg.ForgetOut(deleted=uri)

    def _memory(self, uri, r, similarity=None):
        ca = r["created_at"]
        return eg.Memory(
            uri=uri,
            author=r["author"],
            text=r["text"],
            tags=r["tags"],
            source=r["source"],
            created_at=ca.isoformat() if isinstance(ca, datetime) else (ca or "2026-01-01T00:00:00.000Z"),
            similarity=similarity,
            space=SPACE,
        )

    async def list(self, limit=0, cursor="", author="", tags=None, space=""):
        self.calls.append("list")
        self._check(self.fail)
        items = [(u, r) for u, r in sorted(self.records.items(), reverse=True) if not author or r["author"] == author]
        start = int(cursor or 0)
        page = items[start : start + (limit or 25)]
        nxt = str(start + len(page)) if start + len(page) < len(items) else ""
        return eg.MemoriesOut(memories=[self._memory(u, r) for u, r in page], cursor=nxt)

    async def recall(self, query, limit=0, author="", tags=None, since="", space="", mode=""):
        self.calls.append("recall")
        self.recall_modes.append(mode)
        self._check(self.fail)
        q = set(re.findall(r"\w+", query.lower()))
        scored = []
        for uri, r in self.records.items():
            if tags and not set(tags) <= set(r["tags"]):
                continue
            words = set(re.findall(r"\w+", r["text"].lower()))
            overlap = len(q & words)
            if overlap:
                scored.append((overlap, uri, r))
        scored.sort(key=lambda t: -t[0])
        return eg.MemoriesOut(memories=[self._memory(u, r, similarity=min(1000, 300 + 100 * o)) for o, u, r in scored[: limit or 10]])

    async def aclose(self):
        pass


class WordEmbeddings:
    """Deterministic 768-dim vectors from words, so local vector search works offline."""

    async def embed(self, text: str) -> bytes:
        v = np.zeros(768, dtype=np.float32)
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 768] += 1
        return v.tobytes()


@pytest.fixture
async def env(tmp_path):
    store = Store(path=tmp_path / "t.db")
    await store.initialize()
    store.memory.set_embedding_client(WordEmbeddings(), dimensions=768)
    spaces = FakeSpaces()
    sync = EngramSync(store.memory, spaces, SPACE)
    store.memory.set_engram(sync)
    engine = SearchEngine(store=store.memory, embedding_client=WordEmbeddings())
    engine.set_engram(sync)
    yield store, spaces, sync, engine
    await store._db.close()


async def add(store, content, type="factual", scope="topic", tags=None):
    return await store.memory.add_entry(type, scope, content, metadata={"tags": tags} if tags else {})


# ---- tags ----


def test_tags_round_trip():
    tags = tags_for("episodic", "check_notifications", ["delve", "social"])
    assert tags == ["type:episodic", "scope:check_notifications", "delve", "social"]
    m = eg.Memory(uri="u", author=ME, text="t", tags=tags, source="check_notifications")
    assert parse_tags(m) == ("episodic", "check_notifications", ["delve", "social"])


def test_long_scope_keeps_full_text_in_source_and_clips_the_tag():
    scope = "penny-archive:day:" + "x" * 200
    tags = tags_for("episodic", scope, [])
    assert len(tags[1].encode()) <= 128
    m = eg.Memory(uri="u", author=ME, text="t", tags=tags, source=scope)
    assert parse_tags(m)[1] == scope  # source carries the whole scope


def test_tags_are_deduplicated_and_capped():
    tags = tags_for("factual", "s", ["a", "a", " b ", ""] + [f"t{i}" for i in range(30)])
    assert tags[:4] == ["type:factual", "scope:s", "a", "b"]
    assert len(tags) == 16


def test_unknown_or_missing_type_reads_as_factual():
    assert parse_tags(eg.Memory(uri="u", author=ME, text="t", tags=["x"]))[0] == "factual"
    assert parse_tags(eg.Memory(uri="u", author=ME, text="t", tags=["type:self"]))[0] == "factual"
    assert parse_tags(eg.Memory(uri="u", author=ME, text="t"))[1] == "imported"


# ---- writes ----


async def test_add_pushes_with_type_scope_tags_and_original_time(env):
    store, spaces, _, _ = env
    e = await add(store, "mama likes tea", "episodic", "chat", ["tea"])
    uri = await store.memory.get_engram_uri(e["id"])
    assert uri in spaces.records
    r = spaces.records[uri]
    assert r["text"] == "mama likes tea"
    assert r["tags"] == ["type:episodic", "scope:chat", "tea"]
    assert r["source"] == "chat"
    assert r["created_at"] == datetime.fromisoformat(e["createdAt"])
    assert await store.memory.count_unpushed() == 0


async def test_prompt_types_stay_local(env):
    store, spaces, _, _ = env
    await store.memory.add_entry("self", "self", "i am penny")
    await store.memory.add_entry("operator", "operator", "mama is hailey")
    await store.memory.add_entry("skill", "skill:deploy", "steps")
    assert spaces.records == {} and "remember" not in spaces.calls
    assert await store.memory.count_unpushed() == 0


async def test_a_down_space_never_fails_a_write_and_the_entry_is_pushed_later(env):
    store, spaces, sync, _ = env
    spaces.fail = True
    e = await add(store, "written while down")
    assert e["id"] and await store.memory.get_engram_uri(e["id"]) is None
    assert await store.memory.count_unpushed() == 1
    assert sync.last_error
    spaces.fail = False
    done = await sync.flush()
    assert done["pushed"] == 1 and await store.memory.count_unpushed() == 0
    assert len(spaces.records) == 1


async def test_flush_gives_up_after_repeated_failures(env):
    store, spaces, sync, _ = env
    spaces.fail = True
    for i in range(6):
        await add(store, f"entry {i}")
    spaces.calls.clear()
    done = await sync.flush()
    assert done["pushed"] == 0 and done["failed"] == 3
    assert spaces.calls.count("remember") == 3
    assert await store.memory.count_unpushed() == 6


async def test_update_replaces_the_record_and_keeps_the_id(env):
    store, spaces, _, _ = env
    e = await add(store, "first version", tags=["a"])
    old = await store.memory.get_engram_uri(e["id"])
    await store.memory.update_entry(e["id"], content="second version")
    new = await store.memory.get_engram_uri(e["id"])
    assert new and new != old
    assert old not in spaces.records and spaces.records[new]["text"] == "second version"
    assert spaces.records[new]["tags"] == ["type:factual", "scope:topic", "a"]
    assert (await store.memory.get_entry(e["id"]))["content"] == "second version"
    # tags-only change rewrites too
    await store.memory.update_entry(e["id"], metadata={"tags": ["b"]})
    assert spaces.records[await store.memory.get_engram_uri(e["id"])]["tags"][-1] == "b"
    assert len(spaces.records) == 1


async def test_a_failed_update_push_is_retried_without_leaving_a_duplicate(env):
    store, spaces, sync, _ = env
    e = await add(store, "before")
    old = await store.memory.get_engram_uri(e["id"])
    spaces.fail = True
    await store.memory.update_entry(e["id"], content="after")
    assert await store.memory.get_engram_uri(e["id"]) is None
    assert await store.memory.list_tombstones() == [old]
    spaces.fail = False
    await sync.flush()
    assert await store.memory.list_tombstones() == []
    assert [r["text"] for r in spaces.records.values()] == ["after"]


async def test_updating_an_unpushed_entry_makes_no_remote_calls(env):
    store, spaces, _, _ = env
    spaces.fail = True
    e = await add(store, "pending")
    spaces.calls.clear()
    await store.memory.update_entry(e["id"], content="still pending")
    assert spaces.calls == []


async def test_delete_forgets_the_record(env):
    store, spaces, _, _ = env
    e = await add(store, "to delete")
    assert await store.memory.delete_entry(e["id"])
    assert spaces.records == {}
    assert await store.memory.list_tombstones() == []


async def test_delete_while_down_is_retried(env):
    store, spaces, sync, _ = env
    e = await add(store, "to delete")
    uri = await store.memory.get_engram_uri(e["id"])
    spaces.fail_forget = True
    assert await store.memory.delete_entry(e["id"])
    assert await store.memory.list_tombstones() == [uri] and uri in spaces.records
    spaces.fail_forget = False
    done = await sync.flush()
    assert done["deleted"] == 1 and spaces.records == {}


async def test_deleting_a_record_that_is_already_gone_is_fine(env):
    store, spaces, sync, _ = env
    e = await add(store, "gone elsewhere")
    spaces.records.clear()
    assert await store.memory.delete_entry(e["id"])
    assert await store.memory.list_tombstones() == []


async def test_very_long_entries_are_trimmed_for_the_space_only(env):
    store, spaces, _, _ = env
    e = await add(store, "word " * 8000)
    (r,) = spaces.records.values()
    assert len(r["text"].encode()) <= MAX_TEXT_BYTES
    assert len((await store.memory.get_entry(e["id"]))["content"]) == 40000


# ---- pull ----


async def test_pull_adds_memories_the_space_has_and_we_do_not(env):
    store, spaces, sync, _ = env
    local = await add(store, "already here")
    await spaces.remember("from another machine", tags=["type:episodic", "scope:chat", "x"], source="chat", space=SPACE)
    await spaces.remember("no tags at all", space=SPACE)
    assert await sync.pull() == 2
    rows = {e["content"]: e for e in (await store.memory.list_entries())["entries"]}
    assert rows["from another machine"]["type"] == "episodic"
    assert rows["from another machine"]["scope"] == "chat"
    assert rows["from another machine"]["metadata"] == {"tags": ["x"]}
    assert rows["no tags at all"]["type"] == "factual" and rows["no tags at all"]["scope"] == "imported"
    assert await store.memory.count_unpushed() == 0
    assert await sync.pull() == 0  # idempotent
    assert (await store.memory.get_entry(local["id"]))["content"] == "already here"


async def test_pull_never_deletes_local_entries(env):
    store, spaces, sync, _ = env
    e = await add(store, "kept")
    spaces.records.clear()
    await sync.pull()
    assert await store.memory.get_entry(e["id"]) is not None


async def test_pull_pages(env):
    store, spaces, sync, _ = env
    for i in range(7):
        await spaces.remember(f"remote {i}", space=SPACE)
    # a page size smaller than the list
    orig = spaces.list

    async def small(limit=0, **kw):
        return await orig(limit=3, **kw)

    spaces.list = small
    assert await sync.pull() == 7


# ---- search ----


async def test_search_asks_the_space_for_meaning_only(env):
    # Keywords are matched locally; the space's hybrid ranking would count
    # them twice.
    store, spaces, _, engine = env
    await add(store, "the staging cluster lives in us-east")
    spaces.recall_modes.clear()
    await engine.search("staging cluster", limit=5)
    assert spaces.recall_modes == ["vector"]


async def test_search_takes_its_vector_half_from_the_space(env):
    store, spaces, _, engine = env
    a = await add(store, "the staging cluster lives in us-east")
    await add(store, "bananas are yellow")
    spaces.calls.clear()
    out = await engine.search("staging cluster", limit=5)
    assert "recall" in spaces.calls
    assert out[0]["id"] == a["id"] and out[0]["matched_by"] in ("both", "semantic", "keyword")


async def test_search_filters_by_type_and_tags_after_mapping(env):
    store, spaces, _, engine = env
    ep = await add(store, "deploy notes for tuesday", "episodic", "chat", ["ops"])
    await add(store, "deploy notes for friday", "factual", "chat", ["other"])
    only_ep = await engine.search("deploy notes", types=["episodic"], limit=5)
    assert [r["id"] for r in only_ep] == [ep["id"]]
    tagged = await engine.search("deploy notes", tags=["ops"], limit=5)
    assert [r["id"] for r in tagged] == [ep["id"]]


async def test_search_by_scope_asks_the_space_for_that_scope_tag(env):
    store, spaces, _, engine = env
    a = await add(store, "alpha notes", "factual", "scope-a")
    await add(store, "alpha notes too", "factual", "scope-b")
    out = await engine.search("alpha notes", scope="scope-a", limit=5)
    assert [r["id"] for r in out] == [a["id"]]


async def test_search_falls_back_to_local_vectors_when_the_space_is_down(env):
    store, spaces, _, engine = env
    a = await add(store, "quarterly planning meeting agenda")
    await add(store, "completely unrelated gardening tips")
    spaces.fail = True
    out = await engine.search("planning meeting agenda", limit=3)
    assert out and out[0]["id"] == a["id"]


async def test_unpushed_entries_are_still_found_semantically(env):
    store, spaces, _, engine = env
    spaces.fail = True
    pending = await add(store, "kubernetes upgrade runbook steps")
    spaces.fail = False  # the space is back but this entry hasn't been pushed
    out = await engine.search("kubernetes runbook", limit=3)
    assert pending["id"] in {r["id"] for r in out}


async def test_search_brings_in_memories_the_space_has_that_we_lack(env):
    store, spaces, _, engine = env
    await spaces.remember("penny wrote this on another machine about otters", tags=["type:episodic", "scope:chat"], source="chat", space=SPACE)
    out = await engine.search("otters", limit=3)
    assert out and out[0]["content"].endswith("otters") and out[0]["type"] == "episodic"
    assert await store.memory.get_engram_uri(out[0]["id"])


async def test_without_a_space_search_is_unchanged(env):
    store, spaces, _, engine = env
    engine.set_engram(None)
    a = await add(store, "local only search")
    spaces.calls.clear()
    out = await engine.search("local only search", limit=3)
    assert out[0]["id"] == a["id"] and "recall" not in spaces.calls


# ---- a down or unapproved appview ----


async def test_a_failing_search_pauses_searching_the_space_then_resumes(env, monkeypatch):
    store, spaces, sync, engine = env
    a = await add(store, "gardening schedule for spring")
    spaces.fail = True
    spaces.calls.clear()
    assert (await engine.search("gardening schedule", limit=3))[0]["id"] == a["id"]  # local fallback
    assert (await engine.search("gardening schedule", limit=3))[0]["id"] == a["id"]
    assert spaces.calls.count("recall") == 1  # the second search didn't ask the space
    spaces.fail = False
    monkeypatch.setattr(sync, "_search_blocked_until", 0.0)
    await engine.search("gardening schedule", limit=3)
    assert spaces.calls.count("recall") == 2
    assert sync._search_block_reason == "the space is down"


async def test_a_hung_search_times_out(env, monkeypatch):
    import asyncio

    import src.memory.engram as engram_mod

    store, spaces, sync, engine = env
    a = await add(store, "slow appview test entry")

    async def hang(*args, **kw):
        await asyncio.sleep(30)

    monkeypatch.setattr(spaces, "recall", hang)
    monkeypatch.setattr(engram_mod, "SEARCH_TIMEOUT", 0.05)
    out = await engine.search("slow appview", limit=3)
    assert out and out[0]["id"] == a["id"]


async def test_the_sync_loop_survives_an_unindexed_space(env):
    import asyncio

    store, spaces, sync, _ = env

    async def unknown(*a, **kw):
        raise eg.EngramError("UnknownSpace: this service does not index that space")

    spaces.list = unknown
    spaces.fail = True
    e = await add(store, "pushed even though the appview can't list yet")
    spaces.fail = False
    assert await store.memory.get_engram_uri(e["id"]) is None
    task = asyncio.create_task(sync.run(0.01))
    await asyncio.sleep(0.1)
    assert not task.done()  # the loop outlived the failing pull
    task.cancel()
    # flush ran before the pull failed, so the entry made it to the space
    assert await store.memory.get_engram_uri(e["id"]) is not None
    assert "UnknownSpace" in sync.last_error

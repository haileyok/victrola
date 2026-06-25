"""Tests for the agent_documents → memory_entries migration skip list.

MCP OAuth tokens (``mcptoken:*``) and MCP server configs (``mcpserver:*``)
must never be migrated into ``memory_entries`` — doing so exposes them to
the agent via ``memory.search`` and RAG recall.
"""

import pytest

from src.memory.migration import migrate_documents_to_memory


@pytest.fixture
async def store(tmp_path):
    from src.store.store import Store

    s = Store(path=tmp_path / "test.db")
    await s.initialize()
    yield s
    await s.close()


async def _seed_docs(store, entries: dict[str, str]) -> None:
    """Insert rkey→content pairs into agent_documents."""
    for rkey, content in entries.items():
        await store.documents.create(rkey, content)


async def _entry_scopes(store) -> set[str]:
    """Return the set of all scopes currently in memory_entries."""
    result = await store.memory.list_entries(limit=500)
    return {e["scope"] for e in result["entries"]}


@pytest.mark.asyncio
async def test_mcptoken_not_migrated(store):
    """mcptoken: docs must not appear in memory_entries."""
    await _seed_docs(store, {
        "mcptoken:fastmail": '{"access_token": "secret-abc", "refresh_token": "refresh-xyz"}',
        "operator": "Operator is Hailey.",
    })

    migrated = await migrate_documents_to_memory(store, embedding_client=None)
    assert migrated == 1  # only 'operator' migrated

    scopes = await _entry_scopes(store)
    assert "mcptoken:fastmail" not in scopes
    assert "operator" in scopes


@pytest.mark.asyncio
async def test_mcpserver_not_migrated(store):
    """mcpserver: docs must not appear in memory_entries."""
    await _seed_docs(store, {
        "mcpserver:github": '{"transport": "stdio", "command": "npx"}',
        "operator": "Operator is Hailey.",
    })

    migrated = await migrate_documents_to_memory(store, embedding_client=None)
    assert migrated == 1

    scopes = await _entry_scopes(store)
    assert "mcpserver:github" not in scopes
    assert "operator" in scopes


@pytest.mark.asyncio
async def test_customtool_still_skipped(store):
    """customtool: docs are still skipped (pre-existing behavior)."""
    await _seed_docs(store, {
        "customtool:mytool": '{"definition": "stuff"}',
        "operator": "Operator is Hailey.",
    })

    migrated = await migrate_documents_to_memory(store, embedding_client=None)
    assert migrated == 1

    scopes = await _entry_scopes(store)
    assert "customtool:mytool" not in scopes
    assert "operator" in scopes


@pytest.mark.asyncio
async def test_normal_docs_still_migrated(store):
    """Normal docs (skill, task, free-form) still migrate alongside skipped ones."""
    await _seed_docs(store, {
        "mcptoken:fastmail": '{"access_token": "secret"}',
        "mcpserver:github": '{"transport": "stdio"}',
        "customtool:mytool": '{"definition": "stuff"}',
        "skill:deploy": "How to deploy the app.",
        "operator": "Operator is Hailey.",
        "self": "I am the agent.",
    })

    migrated = await migrate_documents_to_memory(store, embedding_client=None)
    assert migrated == 3  # skill:deploy, operator, self

    scopes = await _entry_scopes(store)
    assert "mcptoken:fastmail" not in scopes
    assert "mcpserver:github" not in scopes
    assert "customtool:mytool" not in scopes
    assert "skill:deploy" in scopes
    assert "operator" in scopes
    assert "self" in scopes

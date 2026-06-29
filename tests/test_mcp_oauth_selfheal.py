"""Tests for the interactive-reauth client scope self-heal.

A client registered before the offline_access fix (PR #55) has no
``offline_access`` in its scope. Reusing it makes the SDK authorize with
``offline_access`` against a client that wasn't registered for it, which Fastmail
rejects with ``invalid_scope``. An interactive connect must drop such a stale
registration so the SDK re-registers fresh; a background reconnect must not.
"""

import pytest
from mcp.shared.auth import OAuthClientInformationFull

from src.store.store import Store
from src.tools.mcp import MCPManager, MCPOAuthTokenStorage, MCPServerConfig
from src.tools.registry import ToolRegistry


def _client_info(scope: str | None) -> OAuthClientInformationFull:
    data = {
        "client_id": "cid-stale",
        "redirect_uris": ["http://localhost:8989/callback"],
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    }
    if scope is not None:
        data["scope"] = scope
    return OAuthClientInformationFull.model_validate(data)


async def _seed(store: Store, name: str, scope: str | None) -> None:
    s = MCPOAuthTokenStorage(store, name)
    await s.set_client_info(_client_info(scope))


@pytest.mark.asyncio
async def test_clear_stale_drops_client_missing_offline_access(tmp_path):
    store = Store(path=tmp_path / "store.db")
    await store.initialize()
    s = MCPOAuthTokenStorage(store, "fastmail")
    await _seed(store, "fastmail", "https://www.fastmail.com/dev/mcp")

    assert await s.clear_stale_client_info() is True
    assert await s.get_client_info() is None  # dropped
    await store.close()


@pytest.mark.asyncio
async def test_clear_stale_keeps_client_with_offline_access(tmp_path):
    store = Store(path=tmp_path / "store.db")
    await store.initialize()
    s = MCPOAuthTokenStorage(store, "fastmail")
    await _seed(store, "fastmail", "https://www.fastmail.com/dev/mcp offline_access")

    assert await s.clear_stale_client_info() is False
    assert await s.get_client_info() is not None  # kept
    await store.close()


@pytest.mark.asyncio
async def test_clear_stale_keeps_client_with_null_scope(tmp_path):
    # A server that didn't echo a scope on registration can't be inferred;
    # don't drop (leave to the operator's deauthorize if needed).
    store = Store(path=tmp_path / "store.db")
    await store.initialize()
    s = MCPOAuthTokenStorage(store, "fastmail")
    await _seed(store, "fastmail", None)

    assert await s.clear_stale_client_info() is False
    assert await s.get_client_info() is not None
    await store.close()


@pytest.mark.asyncio
async def test_clear_stale_noop_when_no_client(tmp_path):
    store = Store(path=tmp_path / "store.db")
    await store.initialize()
    s = MCPOAuthTokenStorage(store, "fastmail")
    assert await s.clear_stale_client_info() is False
    await store.close()

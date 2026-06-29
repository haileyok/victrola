"""Tests that MCP OAuth client registration is persisted.

Without persisting the dynamically-registered client, the SDK re-registers a new
client on every refresh/reconnect, invalidating the saved refresh token and
forcing a full re-auth. These tests pin the round-trip and the clear behavior.
"""

import pytest
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from src.store.store import Store
from src.tools.mcp import MCPManager, MCPOAuthTokenStorage, MCPServerConfig
from src.tools.registry import ToolRegistry


def _client_info(client_id: str = "cid-1") -> OAuthClientInformationFull:
    return OAuthClientInformationFull.model_validate(
        {
            "client_id": client_id,
            "redirect_uris": ["http://localhost:8989/callback"],
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
        }
    )


@pytest.mark.asyncio
async def test_oauth_storage_round_trips_client_info(tmp_path):
    store = Store(path=tmp_path / "store.db")
    await store.initialize()

    s = MCPOAuthTokenStorage(store, "fastmail")
    assert await s.get_client_info() is None  # nothing persisted yet

    await s.set_client_info(_client_info("cid-1"))
    got = await s.get_client_info()
    assert got is not None and got.client_id == "cid-1"

    # The update path (document already exists) also works.
    await s.set_client_info(_client_info("cid-2"))
    assert (await s.get_client_info()).client_id == "cid-2"

    await store.close()


@pytest.mark.asyncio
async def test_clear_oauth_tokens_removes_tokens_and_client_info(tmp_path):
    store = Store(path=tmp_path / "store.db")
    await store.initialize()
    manager = MCPManager(store=store, secret_manager=None, registry=ToolRegistry())
    await manager.create_server(
        MCPServerConfig(
            name="fastmail",
            transport="streamable_http",
            url="https://api.fastmail.com/mcp",
            auth_type="oauth",
        )
    )

    s = MCPOAuthTokenStorage(store, "fastmail")
    await s.set_tokens(
        OAuthToken.model_validate(
            {"access_token": "a", "token_type": "Bearer", "refresh_token": "r"}
        )
    )
    await s.set_client_info(_client_info("cid-1"))
    assert await s.get_tokens() is not None
    assert await s.get_client_info() is not None

    msg = await manager.clear_oauth_tokens("fastmail")
    assert "cleared" in msg.lower()
    assert await s.get_tokens() is None
    assert await s.get_client_info() is None

    await store.close()

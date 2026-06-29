"""Tests for the runtime MCP OAuth SDK shims.

Cover the offline_access scope augmentation (so Fastmail issues a refresh token)
and the refresh-after-reload behavior (so a reloaded token with unknown expiry is
refreshed instead of forcing interactive re-auth).
"""

from types import SimpleNamespace

from mcp.client.auth import oauth2
from mcp.client.auth.oauth2 import OAuthContext

from src.tools.mcp_oauth_patches import apply_mcp_oauth_patches

_FM = "https://www.fastmail.com/dev/mcp"


def _md(scopes):
    # Minimal stand-in for PRM/AS metadata: only ``scopes_supported`` is read.
    return SimpleNamespace(scopes_supported=scopes)


def test_offline_access_appended_when_advertised():
    apply_mcp_oauth_patches()
    # Mirrors Fastmail: the WWW-Authenticate scope omits offline_access, but the
    # protected-resource / authorization-server metadata advertise it.
    scope = oauth2.get_client_metadata_scopes(
        _FM,
        _md([_FM, "offline_access"]),
        _md([_FM, "offline_access"]),
    )
    assert scope is not None
    parts = scope.split()
    assert "offline_access" in parts
    assert _FM in parts


def test_offline_access_not_appended_when_not_advertised():
    apply_mcp_oauth_patches()
    scope = oauth2.get_client_metadata_scopes(
        "https://example.com/api",
        _md(["https://example.com/api"]),
        _md(["https://example.com/api"]),
    )
    assert scope == "https://example.com/api"


def test_offline_access_not_duplicated():
    apply_mcp_oauth_patches()
    scope = oauth2.get_client_metadata_scopes(
        f"{_FM} offline_access",
        _md([_FM, "offline_access"]),
        _md([_FM, "offline_access"]),
    )
    assert scope.split().count("offline_access") == 1


def test_offline_access_from_prm_only_when_as_metadata_missing():
    # The 403 step-up call passes no authorization-server metadata.
    apply_mcp_oauth_patches()
    scope = oauth2.get_client_metadata_scopes(_FM, _md([_FM, "offline_access"]))
    assert "offline_access" in scope.split()


def test_no_scope_stays_none():
    apply_mcp_oauth_patches()
    assert oauth2.get_client_metadata_scopes(None, None, None) is None


def test_reloaded_token_with_refresh_is_invalid_to_force_refresh():
    apply_mcp_oauth_patches()
    ctx = SimpleNamespace(
        current_tokens=SimpleNamespace(access_token="a", refresh_token="r"),
        token_expiry_time=None,
        can_refresh_token=lambda: True,
    )
    assert OAuthContext.is_token_valid(ctx) is False


def test_reloaded_token_without_refresh_stays_valid():
    apply_mcp_oauth_patches()
    ctx = SimpleNamespace(
        current_tokens=SimpleNamespace(access_token="a", refresh_token=None),
        token_expiry_time=None,
        can_refresh_token=lambda: False,
    )
    # No refresh possible -> fall through to stock behavior (loaded token usable).
    assert OAuthContext.is_token_valid(ctx) is True

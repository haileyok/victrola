"""Runtime shims for OAuth gaps in the pinned ``mcp`` SDK (1.28.x).

Two upstream fixes we depend on are not in any released ``mcp`` version as of
1.28.1, so we install small, additive, idempotent patches at runtime. Each is
guarded to disable itself once the SDK ships the corresponding fix.

1. offline_access scope (SEP-2207 / python-sdk PR #2039)
   The SDK's ``get_client_metadata_scopes`` uses the ``WWW-Authenticate`` scope
   verbatim. Fastmail's challenge advertises only its resource scope and omits
   ``offline_access``, so the SDK never requests it and the server never issues a
   refresh token. Without a refresh token ``can_refresh_token()`` is always False
   and every access-token expiry forces a full (interactive) re-authorization.
   Fix: when the authorization server advertises ``offline_access``, append it to
   the requested scope so a refresh token is issued. Our client always registers
   the ``refresh_token`` grant, so requesting it is always appropriate here.

2. refresh after reload (python-sdk PR #2875)
   An ``OAuthContext`` rebuilt from storage restores tokens but not
   ``token_expiry_time``, so ``is_token_valid()`` returns True for a stale access
   token and the proactive-refresh branch is skipped -- forcing a full re-auth on
   restart/reconnect even when a refresh token is available. Fix: treat a loaded
   token with unknown expiry as needing refresh, but only when a refresh is
   actually possible, so a server that issues no refresh token keeps using its
   loaded access token as before.

Remove this module once the pinned ``mcp`` release carries both fixes.
"""

from __future__ import annotations

import inspect
import logging

logger = logging.getLogger(__name__)

_applied = False


def apply_mcp_oauth_patches() -> None:
    """Install the OAuth shims once. Safe to call repeatedly."""
    global _applied
    if _applied:
        return
    _applied = True
    _patch_offline_access_scope()
    _patch_refresh_after_reload()


def _advertises_offline_access(*metadata: object) -> bool:
    """True if any of the given PRM/AS metadata lists ``offline_access``."""
    for md in metadata:
        supported = getattr(md, "scopes_supported", None)
        if supported and "offline_access" in supported:
            return True
    return False


def _patch_offline_access_scope() -> None:
    from mcp.client.auth import oauth2

    original = oauth2.get_client_metadata_scopes

    if getattr(original, "_victrola_shim", False):
        return
    # The SEP-2207 implementation gained a ``client_grant_types`` parameter and
    # appends offline_access itself; if present, the SDK already handles this.
    if "client_grant_types" in inspect.signature(original).parameters:
        return

    def get_client_metadata_scopes(
        www_authenticate_scope,
        protected_resource_metadata,
        authorization_server_metadata=None,
        *args,
        **kwargs,
    ):
        scope = original(
            www_authenticate_scope,
            protected_resource_metadata,
            authorization_server_metadata,
            *args,
            **kwargs,
        )
        if (
            scope
            and "offline_access" not in scope.split()
            and _advertises_offline_access(
                authorization_server_metadata, protected_resource_metadata
            )
        ):
            scope = f"{scope} offline_access"
        return scope

    get_client_metadata_scopes._victrola_shim = True
    oauth2.get_client_metadata_scopes = get_client_metadata_scopes
    logger.debug("Applied MCP OAuth shim: offline_access scope augmentation")


def _patch_refresh_after_reload() -> None:
    from mcp.client.auth.oauth2 import OAuthContext

    original_is_token_valid = OAuthContext.is_token_valid

    if getattr(original_is_token_valid, "_victrola_shim", False):
        return

    def is_token_valid(self) -> bool:
        # A provider rebuilt from storage loses ``token_expiry_time``, so the
        # stock check treats a stale token as valid and skips the proactive
        # refresh. When expiry is unknown but a refresh is possible, prefer
        # refreshing over reusing a possibly-expired access token.
        if (
            self.current_tokens is not None
            and self.token_expiry_time is None
            and self.can_refresh_token()
        ):
            return False
        return original_is_token_valid(self)

    is_token_valid._victrola_shim = True
    OAuthContext.is_token_valid = is_token_valid
    logger.debug("Applied MCP OAuth shim: refresh reloaded token with unknown expiry")

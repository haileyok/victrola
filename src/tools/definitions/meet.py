"""Built-in tool for creating Google Meet links.

Uses the Google Meet REST API (spaces.create) to generate a standalone meeting
space. This returns a ``meetingUri`` like ``https://meet.google.com/abc-mnop-xyz``
that can be attached to a calendar event or shared directly.

This is the proper way to programmatically create a Meet link — it does not
create a calendar event, so there is no duplication when the real event lives
in Fastmail (or any other calendar).

Required secrets (configured via the web interface Secrets page):
  - ``GOOGLE_CLIENT_ID``      — OAuth client ID from Google Cloud Console
  - ``GOOGLE_CLIENT_SECRET``  — OAuth client secret
  - ``GOOGLE_REFRESH_TOKEN``  — Long-lived refresh token from one-time authorization

See the README "Google Meet links" section for one-time OAuth setup instructions.
"""

import logging
from typing import Any

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter

logger = logging.getLogger(__name__)

# Secret names — read from the SecretManager, never exposed to the agent.
_CLIENT_ID_SECRET = "GOOGLE_CLIENT_ID"
_CLIENT_SECRET_SECRET = "GOOGLE_CLIENT_SECRET"
_REFRESH_TOKEN_SECRET = "GOOGLE_REFRESH_TOKEN"

# Google OAuth 2.0 token endpoint.
_TOKEN_URL = "https://oauth2.googleapis.com/token"

# Google Meet REST API — spaces.create endpoint.
_MEET_SPACES_URL = "https://meet.googleapis.com/v2/spaces"

# Map our friendly access_type values to the API's AccessType enum.
_ACCESS_TYPE_MAP: dict[str, str] = {
    "open": "OPEN",
    "trusted": "TRUSTED",
    "restricted": "RESTRICTED",
}


async def _get_access_token(
    ctx: ToolContext,
    client_id: str,
    client_secret: str,
    refresh_token: str,
) -> str:
    """Exchange a refresh token for a fresh access token.

    Raises RuntimeError on failure so the caller can surface a clean error.
    """
    try:
        resp = await ctx.http_client.post(
            _TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    except Exception as e:
        logger.exception("Google OAuth token refresh failed (network error)")
        raise RuntimeError(
            f"Failed to contact Google OAuth endpoint: {e}"
        ) from e

    if resp.status_code >= 400:
        raise RuntimeError(
            f"Google OAuth token refresh failed (HTTP {resp.status_code})"
        )

    try:
        token_data = resp.json()
    except Exception:
        raise RuntimeError(
            f"Google OAuth returned non-JSON response (HTTP {resp.status_code})"
        )

    if not isinstance(token_data, dict):
        raise RuntimeError(
            f"Google OAuth returned unexpected response type: {type(token_data).__name__}"
        )

    access_token = token_data.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("Google OAuth response did not contain a valid access_token")
    return access_token


@TOOL_REGISTRY.tool(
    name="meet.create_link",
    description="""Create a standalone Google Meet link.

Returns a ``meeting_uri`` (e.g. ``https://meet.google.com/abc-mnop-xyz``) and a
``meeting_code`` that can be attached to a calendar event or shared directly.

This uses the Google Meet REST API to create a meeting *space* — it does not
create a calendar event, so there is no duplication when the real event lives
in another calendar (Fastmail, etc.).

Requires three secrets configured via the web interface:
``GOOGLE_CLIENT_ID``, ``GOOGLE_CLIENT_SECRET``, ``GOOGLE_REFRESH_TOKEN``.

The optional ``access_type`` controls who can join without knocking:
  - ``open`` (default) — anyone with the link can join without knocking
  - ``trusted`` — org members and invited users join without knocking; others knock
  - ``restricted`` — only invited users join without knocking; everyone else knocks
""",
    parameters=[
        ToolParameter(
            name="access_type",
            type="string",
            description="Who can join without knocking. One of: open (default), trusted, restricted.",
            required=False,
            default="open",
        ),
    ],
)
async def create_link(
    ctx: ToolContext, access_type: str = "open"
) -> dict[str, Any]:
    sm = ctx.secret_manager
    if sm is None:
        return {"error": "Secret manager not available"}

    try:
        client_id = sm.get_secret(_CLIENT_ID_SECRET)
        client_secret = sm.get_secret(_CLIENT_SECRET_SECRET)
        refresh_token = sm.get_secret(_REFRESH_TOKEN_SECRET)
    except Exception as e:
        logger.exception("Failed to read Google Meet secrets")
        return {"error": f"Failed to read secrets: {e}"}

    missing = [
        name
        for name, val in [
            (_CLIENT_ID_SECRET, client_id),
            (_CLIENT_SECRET_SECRET, client_secret),
            (_REFRESH_TOKEN_SECRET, refresh_token),
        ]
        if not val
    ]
    if missing:
        return {
            "error": (
                "Google Meet not configured. Set the following secrets via the "
                "web interface (Secrets page): "
                + ", ".join(f"`{m}`" for m in missing)
            )
        }

    # Validate access_type early so we don't waste an API call on a bad value.
    if not isinstance(access_type, str):
        return {
            "error": (
                f"access_type must be a string, got {type(access_type).__name__}"
            )
        }
    normalized = access_type.strip().lower() if access_type else "open"
    api_access_type = _ACCESS_TYPE_MAP.get(normalized)
    if api_access_type is None:
        return {
            "error": (
                f"Invalid access_type '{access_type}'. Must be one of: "
                + ", ".join(f"'{k}'" for k in _ACCESS_TYPE_MAP)
            )
        }

    # Step 1: refresh the access token.
    try:
        access_token = await _get_access_token(
            ctx, client_id, client_secret, refresh_token
        )
    except RuntimeError as e:
        return {"error": str(e)}

    # Step 2: create a meeting space.
    body: dict[str, Any] = {"config": {"accessType": api_access_type}}

    try:
        resp = await ctx.http_client.post(
            _MEET_SPACES_URL,
            json=body,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
    except Exception as e:
        logger.exception("Google Meet spaces.create failed (network error)")
        return {"error": f"Failed to contact Google Meet API: {e}"}

    if resp.status_code >= 400:
        return {
            "error": f"Google Meet API returned HTTP {resp.status_code}"
        }

    try:
        data = resp.json()
    except Exception:
        return {
            "error": (
                f"Google Meet API returned non-JSON response "
                f"(HTTP {resp.status_code})"
            )
        }

    if not isinstance(data, dict):
        return {
            "error": (
                f"Google Meet API returned unexpected response type: "
                f"{type(data).__name__}"
            )
        }

    meeting_uri = data.get("meetingUri")
    meeting_code = data.get("meetingCode")
    space_name = data.get("name")

    if not isinstance(meeting_uri, str) or not meeting_uri:
        return {
            "error": "Google Meet API response did not contain a valid meetingUri",
        }

    # Optional fields — only pass through if they're strings.
    meeting_code = meeting_code if isinstance(meeting_code, str) else None
    space_name = space_name if isinstance(space_name, str) else None

    return {
        "success": True,
        "meeting_uri": meeting_uri,
        "meeting_code": meeting_code,
        "space_name": space_name,
    }

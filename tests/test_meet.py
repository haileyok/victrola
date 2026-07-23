"""Tests for meet.create_link tool."""

from unittest.mock import AsyncMock, MagicMock

import httpx

from src.tools.definitions.meet import create_link
from src.tools.registry import ToolContext


def _mock_ctx(http_client=None, secret_manager=None):
    """Build a ToolContext with a mock http_client and optional secret_manager."""
    ctx = MagicMock(spec=ToolContext)
    if http_client is None:
        http_client = MagicMock()
        http_client.post = AsyncMock()
    ctx.http_client = http_client
    ctx.secret_manager = secret_manager
    return ctx


def _make_response(status_code=200, text="OK", json_data=None):
    """Build a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def _mock_secret_manager(
    client_id="test-client-id",
    client_secret="test-secret",
    refresh_token="test-refresh",
):
    sm = MagicMock()
    sm.get_secret.side_effect = lambda name: {
        "GOOGLE_CLIENT_ID": client_id,
        "GOOGLE_CLIENT_SECRET": client_secret,
        "GOOGLE_REFRESH_TOKEN": refresh_token,
    }.get(name)
    return sm


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_create_link_success():
    """meet.create_link returns meeting_uri and meeting_code on success."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    # First call: OAuth token refresh. Second call: spaces.create.
    token_resp = _make_response(200, json_data={"access_token": "ya29.test-token"})
    spaces_resp = _make_response(
        200,
        json_data={
            "name": "spaces/abc123",
            "meetingUri": "https://meet.google.com/abc-mnop-xyz",
            "meetingCode": "abc-mnop-xyz",
        },
    )
    ctx.http_client.post.side_effect = [token_resp, spaces_resp]

    result = await create_link(ctx)

    assert result["success"] is True
    assert result["meeting_uri"] == "https://meet.google.com/abc-mnop-xyz"
    assert result["meeting_code"] == "abc-mnop-xyz"
    assert result["space_name"] == "spaces/abc123"

    # Verify token refresh call
    token_call = ctx.http_client.post.call_args_list[0]
    assert token_call.args[0] == "https://oauth2.googleapis.com/token"
    token_data = token_call.kwargs["data"]
    assert token_data["grant_type"] == "refresh_token"
    assert token_data["client_id"] == "test-client-id"
    assert token_data["client_secret"] == "test-secret"
    assert token_data["refresh_token"] == "test-refresh"

    # Verify spaces.create call
    spaces_call = ctx.http_client.post.call_args_list[1]
    assert spaces_call.args[0] == "https://meet.googleapis.com/v2/spaces"
    assert spaces_call.kwargs["headers"]["Authorization"] == "Bearer ya29.test-token"
    assert spaces_call.kwargs["json"]["config"]["accessType"] == "OPEN"


async def test_create_link_with_restricted_access():
    """meet.create_link passes access_type=RESTRICTED to the API."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    token_resp = _make_response(200, json_data={"access_token": "ya29.test-token"})
    spaces_resp = _make_response(
        200,
        json_data={
            "name": "spaces/xyz",
            "meetingUri": "https://meet.google.com/aaa-bbbb-ccc",
            "meetingCode": "aaa-bbbb-ccc",
        },
    )
    ctx.http_client.post.side_effect = [token_resp, spaces_resp]

    result = await create_link(ctx, access_type="restricted")

    assert result["success"] is True
    spaces_call = ctx.http_client.post.call_args_list[1]
    assert spaces_call.kwargs["json"]["config"]["accessType"] == "RESTRICTED"


async def test_create_link_with_trusted_access():
    """meet.create_link passes access_type=TRUSTED to the API."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    token_resp = _make_response(200, json_data={"access_token": "ya29.test-token"})
    spaces_resp = _make_response(
        200,
        json_data={
            "name": "spaces/xyz",
            "meetingUri": "https://meet.google.com/aaa-bbbb-ccc",
            "meetingCode": "aaa-bbbb-ccc",
        },
    )
    ctx.http_client.post.side_effect = [token_resp, spaces_resp]

    result = await create_link(ctx, access_type="trusted")

    assert result["success"] is True
    spaces_call = ctx.http_client.post.call_args_list[1]
    assert spaces_call.kwargs["json"]["config"]["accessType"] == "TRUSTED"


async def test_create_link_case_insensitive_access_type():
    """meet.create_link normalizes access_type to lowercase."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    token_resp = _make_response(200, json_data={"access_token": "ya29.test-token"})
    spaces_resp = _make_response(
        200,
        json_data={
            "meetingUri": "https://meet.google.com/aaa-bbbb-ccc",
            "meetingCode": "aaa-bbbb-ccc",
        },
    )
    ctx.http_client.post.side_effect = [token_resp, spaces_resp]

    result = await create_link(ctx, access_type="RESTRICTED")

    assert result["success"] is True
    spaces_call = ctx.http_client.post.call_args_list[1]
    assert spaces_call.kwargs["json"]["config"]["accessType"] == "RESTRICTED"


# ---------------------------------------------------------------------------
# Error paths — secrets
# ---------------------------------------------------------------------------


async def test_create_link_missing_secrets():
    """meet.create_link returns a clear error when secrets are missing."""
    sm = MagicMock()
    sm.get_secret.return_value = None  # All secrets missing
    ctx = _mock_ctx(secret_manager=sm)

    result = await create_link(ctx)

    assert "error" in result
    assert "GOOGLE_CLIENT_ID" in result["error"]
    assert "GOOGLE_CLIENT_SECRET" in result["error"]
    assert "GOOGLE_REFRESH_TOKEN" in result["error"]
    ctx.http_client.post.assert_not_called()


async def test_create_link_partial_missing_secrets():
    """meet.create_link lists only the missing secrets."""
    sm = MagicMock()
    sm.get_secret.side_effect = lambda name: {
        "GOOGLE_CLIENT_ID": "test-client-id",
        "GOOGLE_CLIENT_SECRET": None,
        "GOOGLE_REFRESH_TOKEN": "test-refresh",
    }.get(name)
    ctx = _mock_ctx(secret_manager=sm)

    result = await create_link(ctx)

    assert "error" in result
    assert "GOOGLE_CLIENT_SECRET" in result["error"]
    assert "GOOGLE_CLIENT_ID" not in result["error"]
    assert "GOOGLE_REFRESH_TOKEN" not in result["error"]
    ctx.http_client.post.assert_not_called()


async def test_create_link_no_secret_manager():
    """meet.create_link returns error when secret manager is unavailable."""
    ctx = _mock_ctx(secret_manager=None)

    result = await create_link(ctx)

    assert "error" in result
    assert "Secret manager" in result["error"]


async def test_create_link_secret_manager_raises():
    """meet.create_link handles exceptions from secret_manager.get_secret()."""
    sm = MagicMock()
    sm.get_secret.side_effect = RuntimeError("database is locked")
    ctx = _mock_ctx(secret_manager=sm)

    result = await create_link(ctx)

    assert "error" in result
    assert "Failed to read secrets" in result["error"]
    ctx.http_client.post.assert_not_called()


# ---------------------------------------------------------------------------
# Error paths — access_type validation
# ---------------------------------------------------------------------------


async def test_create_link_invalid_access_type():
    """meet.create_link rejects an invalid access_type without making API calls."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    result = await create_link(ctx, access_type="bogus")

    assert "error" in result
    assert "bogus" in result["error"]
    ctx.http_client.post.assert_not_called()


async def test_create_link_non_string_access_type():
    """meet.create_link rejects a non-string access_type."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    result = await create_link(ctx, access_type=123)

    assert "error" in result
    assert "string" in result["error"]
    ctx.http_client.post.assert_not_called()


# ---------------------------------------------------------------------------
# Error paths — OAuth token refresh
# ---------------------------------------------------------------------------


async def test_create_link_oauth_error():
    """meet.create_link handles OAuth token refresh failure."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    # OAuth token refresh returns an error
    ctx.http_client.post.return_value = _make_response(
        400, text='{"error":"invalid_grant"}'
    )

    result = await create_link(ctx)

    assert "error" in result
    assert "400" in result["error"]
    # Only one call should have been made (the token refresh, not spaces.create)
    assert ctx.http_client.post.call_count == 1


async def test_create_link_oauth_network_error():
    """meet.create_link handles network errors during token refresh."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    ctx.http_client.post.side_effect = httpx.ConnectError("connection refused")

    result = await create_link(ctx)

    assert "error" in result
    assert "Failed to contact Google OAuth" in result["error"]


async def test_create_link_oauth_non_json_response():
    """meet.create_link handles non-JSON OAuth response."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    resp = _make_response(200, text="<html>Gateway Timeout</html>")
    resp.json.side_effect = ValueError("not JSON")
    ctx.http_client.post.return_value = resp

    result = await create_link(ctx)

    assert "error" in result
    assert "non-JSON" in result["error"]


async def test_create_link_oauth_wrong_shape_response():
    """meet.create_link handles JSON of wrong type from OAuth."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    ctx.http_client.post.return_value = _make_response(
        200, json_data=["unexpected", "array"]
    )

    result = await create_link(ctx)

    assert "error" in result
    assert "unexpected response type" in result["error"].lower()


async def test_create_link_oauth_missing_access_token():
    """meet.create_link handles OAuth response without access_token."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    ctx.http_client.post.return_value = _make_response(
        200, json_data={"token_type": "Bearer"}
    )

    result = await create_link(ctx)

    assert "error" in result
    assert "access_token" in result["error"]


async def test_create_link_oauth_non_string_access_token():
    """meet.create_link handles non-string access_token in OAuth response."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    ctx.http_client.post.return_value = _make_response(
        200, json_data={"access_token": 12345}
    )

    result = await create_link(ctx)

    assert "error" in result
    assert "access_token" in result["error"]


# ---------------------------------------------------------------------------
# Error paths — Meet API
# ---------------------------------------------------------------------------


async def test_create_link_meet_api_error():
    """meet.create_link handles Google Meet API errors."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    token_resp = _make_response(200, json_data={"access_token": "ya29.test-token"})
    error_resp = _make_response(403, text='{"error":"permission_denied"}')
    ctx.http_client.post.side_effect = [token_resp, error_resp]

    result = await create_link(ctx)

    assert "error" in result
    assert "403" in result["error"]
    # Error should not contain raw upstream body
    assert "permission_denied" not in result["error"]


async def test_create_link_meet_api_network_error():
    """meet.create_link handles network errors during spaces.create."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    token_resp = _make_response(200, json_data={"access_token": "ya29.test-token"})
    ctx.http_client.post.side_effect = [token_resp, httpx.ConnectError("timeout")]

    result = await create_link(ctx)

    assert "error" in result
    assert "Failed to contact Google Meet API" in result["error"]


async def test_create_link_meet_api_non_json_response():
    """meet.create_link handles non-JSON Meet API response."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    token_resp = _make_response(200, json_data={"access_token": "ya29.test-token"})
    meet_resp = _make_response(200, text="<html>error</html>")
    meet_resp.json.side_effect = ValueError("not JSON")
    ctx.http_client.post.side_effect = [token_resp, meet_resp]

    result = await create_link(ctx)

    assert "error" in result
    assert "non-JSON" in result["error"]


async def test_create_link_meet_api_wrong_shape_response():
    """meet.create_link handles JSON of wrong type from Meet API."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    token_resp = _make_response(200, json_data={"access_token": "ya29.test-token"})
    ctx.http_client.post.side_effect = [
        token_resp,
        _make_response(200, json_data="just a string"),
    ]

    result = await create_link(ctx)

    assert "error" in result
    assert "unexpected response type" in result["error"].lower()


async def test_create_link_missing_meeting_uri_in_response():
    """meet.create_link handles a response without meetingUri."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    token_resp = _make_response(200, json_data={"access_token": "ya29.test-token"})
    bad_resp = _make_response(200, json_data={"name": "spaces/abc"})
    ctx.http_client.post.side_effect = [token_resp, bad_resp]

    result = await create_link(ctx)

    assert "error" in result
    assert "meetingUri" in result["error"]
    # Should not expose raw upstream response
    assert "raw_response" not in result


async def test_create_link_non_string_meeting_uri():
    """meet.create_link rejects a non-string meetingUri."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    token_resp = _make_response(200, json_data={"access_token": "ya29.test-token"})
    bad_resp = _make_response(
        200, json_data={"meetingUri": {"unexpected": True}}
    )
    ctx.http_client.post.side_effect = [token_resp, bad_resp]

    result = await create_link(ctx)

    assert "error" in result
    assert "meetingUri" in result["error"]


async def test_create_link_non_string_optional_fields():
    """meet.create_link nulls out non-string optional fields instead of passing them through."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    token_resp = _make_response(200, json_data={"access_token": "ya29.test-token"})
    spaces_resp = _make_response(
        200,
        json_data={
            "meetingUri": "https://meet.google.com/abc-mnop-xyz",
            "meetingCode": 12345,
            "name": ["spaces", "weird"],
        },
    )
    ctx.http_client.post.side_effect = [token_resp, spaces_resp]

    result = await create_link(ctx)

    assert result["success"] is True
    assert result["meeting_uri"] == "https://meet.google.com/abc-mnop-xyz"
    assert result["meeting_code"] is None
    assert result["space_name"] is None


async def test_create_link_success_without_optional_fields():
    """meet.create_link succeeds when meetingCode and name are absent."""
    sm = _mock_secret_manager()
    ctx = _mock_ctx(secret_manager=sm)

    token_resp = _make_response(200, json_data={"access_token": "ya29.test-token"})
    spaces_resp = _make_response(
        200,
        json_data={"meetingUri": "https://meet.google.com/abc-mnop-xyz"},
    )
    ctx.http_client.post.side_effect = [token_resp, spaces_resp]

    result = await create_link(ctx)

    assert result["success"] is True
    assert result["meeting_uri"] == "https://meet.google.com/abc-mnop-xyz"
    assert result["meeting_code"] is None
    assert result["space_name"] is None

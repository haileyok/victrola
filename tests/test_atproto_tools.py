"""Tests for the built-in atproto.* tools, against a fake PDS/identity network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from src.config import CONFIG
from src.tools.definitions import atproto as ap
from src.tools.registry import TOOL_REGISTRY, ToolContext

AGENT_DID = "did:plc:agent123"
AGENT_PDS = "https://pds.example.com"
OTHER_DID = "did:plc:other456"
OTHER_PDS = "https://other-pds.example.net"


def _did_doc(did: str, handle: str, pds: str) -> dict[str, Any]:
    return {
        "id": did,
        "alsoKnownAs": [f"at://{handle}"],
        "service": [
            {
                "id": "#atproto_pds",
                "type": "AtprotoPersonalDataServer",
                "serviceEndpoint": pds,
            }
        ],
    }


class FakeNetwork:
    """Routes requests to a fake PDS, PLC directory, and handle endpoints."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.access_token = "access-1"
        self.logins = 0
        self.refreshes = 0
        self.expire_next = False  # make the next authed call fail ExpiredToken
        self.refresh_fails = False

    def _authed(self, request: httpx.Request) -> httpx.Response | None:
        if request.headers.get("authorization") != f"Bearer {self.access_token}":
            return httpx.Response(401, json={"error": "AuthMissing"})
        if self.expire_next:
            self.expire_next = False
            return httpx.Response(400, json={"error": "ExpiredToken", "message": "expired"})
        return None

    def _session_body(self) -> dict[str, Any]:
        return {
            "did": AGENT_DID,
            "handle": "agent.example.com",
            "accessJwt": self.access_token,
            "refreshJwt": "refresh-1",
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = request.url
        host, path = url.host, url.path

        # Identity
        if host == "agent.example.com" and path == "/.well-known/atproto-did":
            return httpx.Response(200, text=AGENT_DID)
        if host == "other.example.org" and path == "/.well-known/atproto-did":
            return httpx.Response(404)
        if host == "public.api.bsky.app" and path.endswith("resolveHandle"):
            if url.params.get("handle") == "other.example.org":
                return httpx.Response(200, json={"did": OTHER_DID})
            return httpx.Response(400, json={"error": "InvalidRequest"})
        if host == "plc.directory":
            did = path.lstrip("/")
            if did == AGENT_DID:
                return httpx.Response(200, json=_did_doc(AGENT_DID, "agent.example.com", AGENT_PDS))
            if did == OTHER_DID:
                return httpx.Response(200, json=_did_doc(OTHER_DID, "other.example.org", OTHER_PDS))
            return httpx.Response(404)

        # Other user's PDS (public reads only)
        if host == "other-pds.example.net":
            if path == "/xrpc/com.atproto.repo.getRecord":
                return httpx.Response(
                    200,
                    json={"uri": f"at://{OTHER_DID}/x/y", "cid": "bafyother", "value": {"text": "hi"}},
                )
            if path == "/xrpc/com.atproto.repo.listRecords":
                return httpx.Response(200, json={"records": [], "cursor": None})
            return httpx.Response(404)

        # Agent's PDS
        if host == "pds.example.com":
            if path == "/xrpc/com.atproto.server.createSession":
                body = json.loads(request.content)
                if body != {"identifier": "agent.example.com", "password": "app-pw"}:
                    return httpx.Response(401, json={"error": "AuthenticationRequired"})
                self.logins += 1
                return httpx.Response(200, json=self._session_body())
            if path == "/xrpc/com.atproto.server.refreshSession":
                if self.refresh_fails or request.headers.get("authorization") != "Bearer refresh-1":
                    return httpx.Response(400, json={"error": "ExpiredToken"})
                self.refreshes += 1
                self.access_token = f"access-{self.refreshes + 1}"
                return httpx.Response(200, json=self._session_body())
            if path in ("/xrpc/com.atproto.repo.getRecord", "/xrpc/com.atproto.repo.listRecords"):
                return httpx.Response(200, json={"uri": "at://self", "cid": "bafyself", "value": {}})
            if (err := self._authed(request)) is not None:
                return err
            if path == "/xrpc/com.atproto.repo.createRecord":
                body = json.loads(request.content)
                return httpx.Response(
                    200,
                    json={"uri": f"at://{AGENT_DID}/{body['collection']}/3kabc", "cid": "bafynew"},
                )
            if path == "/xrpc/com.atproto.repo.putRecord":
                return httpx.Response(200, json={"uri": "at://x", "cid": "bafyput"})
            if path == "/xrpc/com.atproto.repo.deleteRecord":
                return httpx.Response(200, json={})
            if path == "/xrpc/com.atproto.repo.uploadBlob":
                return httpx.Response(
                    200,
                    json={
                        "blob": {
                            "$type": "blob",
                            "ref": {"$link": "bafyblob"},
                            "mimeType": request.headers["content-type"],
                            "size": len(request.content),
                        }
                    },
                )
            if path == "/xrpc/app.bsky.notification.listNotifications":
                return httpx.Response(200, json={"notifications": [], "seen": dict(url.params)})
            if path == "/xrpc/app.bsky.notification.updateSeen":
                return httpx.Response(200)
            if path == "/xrpc/com.example.custom.thing":
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(501, json={"error": "MethodNotImplemented"})

        return httpx.Response(599)

    def last(self, path: str) -> httpx.Request:
        return [r for r in self.requests if r.url.path == path][-1]


@pytest.fixture(autouse=True)
def _reset_session():
    ap._reset_session_cache()
    yield
    ap._reset_session_cache()


@pytest.fixture
def net() -> FakeNetwork:
    return FakeNetwork()


def _ctx(net: FakeNetwork, secrets: dict[str, str] | None = None) -> Any:
    if secrets is None:
        secrets = {"ATPROTO_IDENTIFIER": "agent.example.com", "ATPROTO_APP_PASSWORD": "app-pw"}
    ctx = MagicMock(spec=ToolContext)
    ctx.http_client = httpx.AsyncClient(transport=httpx.MockTransport(net))
    sm = MagicMock()
    sm.get_secret.side_effect = lambda name: secrets.get(name)
    ctx.secret_manager = sm
    return ctx


# --- registration -----------------------------------------------------------


def test_all_tools_registered():
    names = {t.name for t in TOOL_REGISTRY.all_tools() if t.name.startswith("atproto.")}
    assert names == {
        "atproto.whoami",
        "atproto.resolve",
        "atproto.create_record",
        "atproto.put_record",
        "atproto.delete_record",
        "atproto.get_record",
        "atproto.list_records",
        "atproto.upload_blob",
        "atproto.query",
        "atproto.procedure",
    }


def test_no_tool_has_object_first_param():
    """The registry only unwraps single-object calls when param 1 isn't an object."""
    for t in TOOL_REGISTRY.all_tools():
        if t.name.startswith("atproto.") and t.parameters:
            assert t.parameters[0].type != "object", t.name


# --- login & session --------------------------------------------------------


async def test_whoami_logs_in_via_resolved_pds(net):
    result = await ap.whoami(_ctx(net))
    assert result == {"did": AGENT_DID, "handle": "agent.example.com", "pds": AGENT_PDS}
    assert net.logins == 1


async def test_session_is_cached(net):
    ctx = _ctx(net)
    await ap.whoami(ctx)
    await ap.whoami(ctx)
    await ap.create_record(ctx, "app.bsky.feed.post", {"text": "x", "createdAt": "now"})
    assert net.logins == 1


async def test_pds_url_secret_skips_resolution(net):
    ctx = _ctx(
        net,
        {
            "ATPROTO_IDENTIFIER": "agent.example.com",
            "ATPROTO_APP_PASSWORD": "app-pw",
            "ATPROTO_PDS_URL": "https://pds.example.com/",
        },
    )
    await ap.whoami(ctx)
    assert not any(r.url.host in ("plc.directory", "agent.example.com") for r in net.requests)


async def test_changed_secrets_force_new_login(net):
    secrets = {"ATPROTO_IDENTIFIER": "agent.example.com", "ATPROTO_APP_PASSWORD": "app-pw"}
    ctx = _ctx(net, secrets)
    await ap.whoami(ctx)
    secrets["ATPROTO_APP_PASSWORD"] = "wrong"
    result = await ap.whoami(ctx)
    assert result["status"] == 401
    assert "logging in failed" in result["error"]


async def test_missing_secrets_explain_setup(net):
    result = await ap.whoami(_ctx(net, {}))
    assert "ATPROTO_IDENTIFIER" in result["error"]
    assert "ATPROTO_APP_PASSWORD" in result["error"]
    assert net.requests == []


async def test_expired_token_refreshes_and_retries(net):
    ctx = _ctx(net)
    await ap.whoami(ctx)
    net.expire_next = True
    result = await ap.create_record(ctx, "app.bsky.feed.post", {"text": "x"})
    assert result["cid"] == "bafynew"
    assert net.refreshes == 1
    assert net.last("/xrpc/com.atproto.repo.createRecord").headers["authorization"] == "Bearer access-2"


async def test_failed_refresh_falls_back_to_login(net):
    ctx = _ctx(net)
    await ap.whoami(ctx)
    net.expire_next = True
    net.refresh_fails = True
    result = await ap.create_record(ctx, "app.bsky.feed.post", {"text": "x"})
    assert result["cid"] == "bafynew"
    assert net.logins == 2


async def test_tokens_never_returned_to_agent(net):
    result = await ap.whoami(_ctx(net))
    assert "access-1" not in json.dumps(result)
    assert "refresh-1" not in json.dumps(result)


# --- record writes ----------------------------------------------------------


async def test_create_record_fills_type_and_repo(net):
    ctx = _ctx(net)
    result = await ap.create_record(
        ctx, "app.bsky.feed.post", {"text": "hello", "createdAt": "2026-10-06T00:00:00Z"}
    )
    assert result == {"uri": f"at://{AGENT_DID}/app.bsky.feed.post/3kabc", "cid": "bafynew"}
    body = json.loads(net.last("/xrpc/com.atproto.repo.createRecord").content)
    assert body["repo"] == AGENT_DID
    assert body["record"]["$type"] == "app.bsky.feed.post"
    assert "rkey" not in body and "validate" not in body


async def test_create_record_keeps_explicit_type_and_options(net):
    ctx = _ctx(net)
    await ap.create_record(
        ctx, "com.example.thing", {"$type": "com.example.thing", "a": 1}, rkey="self", validate=False
    )
    body = json.loads(net.last("/xrpc/com.atproto.repo.createRecord").content)
    assert body["rkey"] == "self"
    assert body["validate"] is False


async def test_put_record_sends_swap(net):
    ctx = _ctx(net)
    result = await ap.put_record(
        ctx, "app.bsky.actor.profile", "self", {"displayName": "Bot"}, swap_record="bafyold"
    )
    assert result["cid"] == "bafyput"
    body = json.loads(net.last("/xrpc/com.atproto.repo.putRecord").content)
    assert body["swapRecord"] == "bafyold"
    assert body["record"]["$type"] == "app.bsky.actor.profile"


async def test_delete_record(net):
    result = await ap.delete_record(_ctx(net), "app.bsky.feed.post", "3kabc")
    assert result["success"] is True
    body = json.loads(net.last("/xrpc/com.atproto.repo.deleteRecord").content)
    assert body == {"repo": AGENT_DID, "collection": "app.bsky.feed.post", "rkey": "3kabc"}


async def test_pds_errors_are_reported(net):
    result = await ap.query(_ctx(net), "com.example.unknown.method")
    assert result["status"] == 501
    assert result["details"]["error"] == "MethodNotImplemented"


# --- reads --------------------------------------------------------------------


async def test_get_record_from_other_repo_by_handle(net):
    # No login needed to read someone else's repo.
    result = await ap.get_record(_ctx(net, {}), "app.bsky.feed.post", "3kxyz", repo="@other.example.org")
    assert result["value"] == {"text": "hi"}
    req = net.last("/xrpc/com.atproto.repo.getRecord")
    assert req.url.host == "other-pds.example.net"
    assert req.url.params["repo"] == OTHER_DID
    assert "authorization" not in req.headers


async def test_list_records_clamps_limit_and_defaults_to_own_repo(net):
    ctx = _ctx(net)
    await ap.list_records(ctx, "app.bsky.feed.post", limit=1000, reverse=True)
    req = net.last("/xrpc/com.atproto.repo.listRecords")
    assert req.url.host == "pds.example.com"
    assert req.url.params["repo"] == AGENT_DID
    assert req.url.params["limit"] == "100"
    assert req.url.params["reverse"] == "true"


async def test_resolve(net):
    result = await ap.resolve(_ctx(net, {}), "other.example.org")
    assert result == {"did": OTHER_DID, "handle": "other.example.org", "pds": OTHER_PDS}


# --- blobs --------------------------------------------------------------------


async def test_upload_blob_from_workspace(net, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(CONFIG, "workspace_dir", str(tmp_path))
    (tmp_path / "cat.png").write_bytes(b"\x89PNG fake")
    result = await ap.upload_blob(_ctx(net), "cat.png")
    assert result["blob"]["mimeType"] == "image/png"
    assert result["blob"]["size"] == len(b"\x89PNG fake")


async def test_upload_blob_rejects_paths_outside_workspace(net, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(CONFIG, "workspace_dir", str(tmp_path / "ws"))
    (tmp_path / "ws").mkdir()
    result = await ap.upload_blob(_ctx(net), "../secrets.json")
    assert "outside the workspace" in result["error"]
    assert not any(r.url.path.endswith("uploadBlob") for r in net.requests)


# --- raw XRPC -------------------------------------------------------------------


async def test_query_defaults_bsky_proxy_and_encodes_params(net):
    ctx = _ctx(net)
    result = await ap.query(
        ctx,
        "app.bsky.notification.listNotifications",
        params={"limit": 10, "priority": True, "cursor": None},
    )
    assert result["seen"] == {"limit": "10", "priority": "true"}
    req = net.last("/xrpc/app.bsky.notification.listNotifications")
    assert req.headers["atproto-proxy"] == "did:web:api.bsky.app#bsky_appview"


async def test_query_proxy_none_and_custom(net):
    ctx = _ctx(net)
    await ap.query(ctx, "app.bsky.notification.listNotifications", proxy="none")
    assert "atproto-proxy" not in net.last("/xrpc/app.bsky.notification.listNotifications").headers
    await ap.query(ctx, "com.example.custom.thing", proxy="did:web:svc.example.com#my_svc")
    assert net.last("/xrpc/com.example.custom.thing").headers["atproto-proxy"] == (
        "did:web:svc.example.com#my_svc"
    )


async def test_query_without_known_prefix_sends_no_proxy(net):
    await ap.query(_ctx(net), "com.example.custom.thing")
    assert "atproto-proxy" not in net.last("/xrpc/com.example.custom.thing").headers


async def test_procedure_with_empty_response(net):
    result = await ap.procedure(
        _ctx(net), "app.bsky.notification.updateSeen", input={"seenAt": "2026-10-06T00:00:00Z"}
    )
    assert result == {}
    req = net.last("/xrpc/app.bsky.notification.updateSeen")
    assert req.method == "POST"
    assert json.loads(req.content) == {"seenAt": "2026-10-06T00:00:00Z"}


async def test_registry_unwraps_single_object_call(net):
    """Agents often call tools.atproto.query({nsid, params}); that must work."""
    ctx = _ctx(net)
    result = await TOOL_REGISTRY.execute(
        ctx,
        "atproto.query",
        {"nsid": {"nsid": "com.example.custom.thing", "params": {"a": 1}}},
    )
    assert result == {"ok": True}


# --- validation -----------------------------------------------------------------


@pytest.mark.parametrize(
    "nsid",
    ["../../admin", "com.example/../x", "app.bsky.feed.post?x=1", "nodots", "", None, "a..b.c"],
)
async def test_bad_nsids_rejected_before_any_request(net, nsid):
    result = await ap.query(_ctx(net), nsid)
    assert "NSID" in result["error"]
    assert net.requests == []


@pytest.mark.parametrize("rkey", ["..", ".", "a/b", "a b", "", "x" * 513])
async def test_bad_rkeys_rejected(net, rkey):
    result = await ap.delete_record(_ctx(net), "app.bsky.feed.post", rkey)
    assert "record key" in result["error"]


@pytest.mark.parametrize("repo", ["http://evil", "a/b", "did:plc:", "not a handle"])
async def test_bad_repos_rejected(net, repo):
    result = await ap.get_record(_ctx(net), "app.bsky.feed.post", "abc", repo=repo)
    assert "handle or DID" in result["error"]
    assert net.requests == []


async def test_bad_proxy_rejected(net):
    result = await ap.query(_ctx(net), "com.example.custom.thing", proxy="https://evil.example")
    assert "proxy" in result["error"]


async def test_record_must_be_object(net):
    result = await ap.create_record(_ctx(net), "app.bsky.feed.post", "just text")  # type: ignore[arg-type]
    assert "record must be a JSON object" in result["error"]

"""Built-in AT Protocol tools: records, blobs, identity, and raw XRPC.

These give the agent general access to the AT Protocol network through its own
account. Higher-level, app-specific behavior (Bluesky posting helpers,
notification digests, etc.) is meant to be built by the agent as custom tools
on top of these, by calling ``tools.atproto.*`` from custom tool code. Custom
tools therefore never need the account password.

Authentication uses an app password (Bluesky Settings -> Privacy and security
-> App passwords). App passwords cannot change account settings or delete the
account, which bounds what the agent can do.

Required secrets (configured via the web interface Secrets page):
  - ``ATPROTO_IDENTIFIER``   - the agent account's handle or DID
  - ``ATPROTO_APP_PASSWORD`` - an app password for that account
Optional:
  - ``ATPROTO_PDS_URL``      - the account's PDS URL. If unset, it is found by
    resolving the identifier's DID document.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from src.tools.definitions.workspace_upload import _read_workspace_file
from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter

logger = logging.getLogger(__name__)

_IDENTIFIER_SECRET = "ATPROTO_IDENTIFIER"
_PASSWORD_SECRET = "ATPROTO_APP_PASSWORD"
_PDS_URL_SECRET = "ATPROTO_PDS_URL"

_PLC_DIRECTORY = "https://plc.directory"
# Public AppView endpoint, used only as a fallback for handle resolution when
# the handle's own /.well-known/atproto-did lookup fails (e.g. DNS-only handles).
_PUBLIC_APPVIEW = "https://public.api.bsky.app"

# Services that calls in these namespaces are sent to when the agent doesn't
# pass `proxy`. Set explicitly rather than relying on PDS defaults, which the
# protocol spec leaves unspecified.
_DEFAULT_PROXIES: dict[str, str] = {
    "app.bsky.": "did:web:api.bsky.app#bsky_appview",
    "chat.bsky.": "did:web:api.bsky.chat#bsky_chat",
}

# Syntax checks. NSIDs, rkeys, and repos end up in URL paths/queries, so they
# are validated before use to rule out path injection.
_NSID_RE = re.compile(
    r"^[a-zA-Z](?:[a-zA-Z0-9-]{0,62}[a-zA-Z0-9])?"
    r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,62}[a-zA-Z0-9])?)+"
    r"\.[a-zA-Z][a-zA-Z0-9]{0,62}$"
)
_RKEY_RE = re.compile(r"^[A-Za-z0-9._:~-]{1,512}$")
_DID_RE = re.compile(r"^did:[a-z]+:[A-Za-z0-9._:%-]{1,2048}$")
_HANDLE_RE = re.compile(
    r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
    r"[a-zA-Z](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$"
)
_PROXY_RE = re.compile(r"^did:[a-z]+:[A-Za-z0-9._:%-]+#[A-Za-z0-9_-]+$")

# Errors that mean "the access token is no longer usable; refresh and retry".
_EXPIRED_ERRORS = {"ExpiredToken", "InvalidToken"}

_MAX_LIST_LIMIT = 100


class AtprotoError(Exception):
    """An error that is returned to the agent as ``{"error": ...}``."""

    def __init__(self, message: str, status: int | None = None, details: Any = None):
        super().__init__(message)
        self.status = status
        self.details = details

    def to_result(self) -> dict[str, Any]:
        result: dict[str, Any] = {"error": str(self)}
        if self.status is not None:
            result["status"] = self.status
        if self.details is not None:
            result["details"] = self.details
        return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check_nsid(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _NSID_RE.match(value):
        raise AtprotoError(
            f"{field} must be an NSID like 'app.bsky.feed.post', got {value!r}"
        )
    return value


def _check_rkey(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not _RKEY_RE.match(value)
        or value in (".", "..")
    ):
        raise AtprotoError(f"invalid record key: {value!r}")
    return value


def _check_identifier(value: Any, field: str = "repo") -> str:
    if isinstance(value, str):
        value = value.strip().removeprefix("@")
        if _DID_RE.match(value) or _HANDLE_RE.match(value):
            return value
    raise AtprotoError(f"{field} must be a handle or DID, got {value!r}")


def _resolve_proxy(nsid: str, proxy: Any) -> str | None:
    """Return the atproto-proxy header value for a call, or None for no header."""
    if proxy is None or proxy == "":
        for prefix, service in _DEFAULT_PROXIES.items():
            if nsid.startswith(prefix):
                return service
        return None
    if proxy == "none":
        return None
    if not isinstance(proxy, str) or not _PROXY_RE.match(proxy):
        raise AtprotoError(
            "proxy must look like 'did:web:example.com#service_id', or 'none' "
            f"to send the call to the PDS itself; got {proxy!r}"
        )
    return proxy


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _error_from_response(resp: httpx.Response, what: str) -> AtprotoError:
    details: Any = None
    message = f"{what} failed (HTTP {resp.status_code})"
    try:
        body = resp.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        details = {k: body[k] for k in ("error", "message") if k in body} or None
        if body.get("error"):
            message += f": {body['error']}"
            if body.get("message"):
                message += f" - {body['message']}"
    return AtprotoError(message, status=resp.status_code, details=details)


def _is_expired(resp: httpx.Response) -> bool:
    if resp.status_code == 401:
        return True
    if resp.status_code != 400:
        return False
    try:
        body = resp.json()
    except Exception:
        return False
    return isinstance(body, dict) and body.get("error") in _EXPIRED_ERRORS


def _json_or_raise(resp: httpx.Response, what: str) -> Any:
    if resp.status_code >= 400:
        raise _error_from_response(resp, what)
    if not resp.content:
        return {}
    try:
        return resp.json()
    except Exception:
        raise AtprotoError(f"{what} returned a non-JSON response", status=resp.status_code)


async def _request(
    ctx: ToolContext, method: str, url: str, what: str, **kwargs: Any
) -> httpx.Response:
    try:
        return await ctx.http_client.request(method, url, **kwargs)
    except httpx.HTTPError as e:
        raise AtprotoError(f"{what} failed: {type(e).__name__}: {e}") from e


# ---------------------------------------------------------------------------
# Identity resolution
# ---------------------------------------------------------------------------


def _https_host(host: str) -> str:
    """Validate a bare hostname used to build an https:// URL."""
    if not _HANDLE_RE.match(host):
        raise AtprotoError(f"not a valid hostname: {host!r}")
    return host


async def _resolve_handle(ctx: ToolContext, handle: str) -> str:
    host = _https_host(handle.lower())
    # 1. The handle's own well-known endpoint (works for most hosted handles).
    try:
        resp = await ctx.http_client.get(f"https://{host}/.well-known/atproto-did")
        if resp.status_code == 200:
            did = resp.text.strip()
            if _DID_RE.match(did):
                return did
    except httpx.HTTPError:
        pass
    # 2. Fall back to a resolver that also handles DNS TXT-only handles.
    resp = await _request(
        ctx,
        "GET",
        f"{_PUBLIC_APPVIEW}/xrpc/com.atproto.identity.resolveHandle",
        f"resolving handle {handle}",
        params={"handle": host},
    )
    data = _json_or_raise(resp, f"resolving handle {handle}")
    did = data.get("did") if isinstance(data, dict) else None
    if not isinstance(did, str) or not _DID_RE.match(did):
        raise AtprotoError(f"could not resolve handle {handle!r} to a DID")
    return did


async def _fetch_did_document(ctx: ToolContext, did: str) -> dict[str, Any]:
    if did.startswith("did:plc:"):
        url = f"{_PLC_DIRECTORY}/{quote(did, safe=':')}"
    elif did.startswith("did:web:"):
        # atproto only supports hostname-level did:web (no paths/ports).
        url = f"https://{_https_host(did.removeprefix('did:web:'))}/.well-known/did.json"
    else:
        raise AtprotoError(f"unsupported DID method: {did!r}")
    resp = await _request(ctx, "GET", url, f"fetching DID document for {did}")
    doc = _json_or_raise(resp, f"fetching DID document for {did}")
    if not isinstance(doc, dict) or doc.get("id") != did:
        raise AtprotoError(f"DID document for {did} is malformed or for a different DID")
    return doc


def _pds_from_doc(doc: dict[str, Any]) -> str:
    for svc in doc.get("service") or []:
        if not isinstance(svc, dict):
            continue
        if str(svc.get("id", "")).endswith("#atproto_pds") and svc.get(
            "type"
        ) == "AtprotoPersonalDataServer":
            endpoint = svc.get("serviceEndpoint")
            if isinstance(endpoint, str) and urlparse(endpoint).scheme == "https":
                return endpoint.rstrip("/")
    raise AtprotoError(f"DID document for {doc.get('id')} lists no https PDS endpoint")


def _handle_from_doc(doc: dict[str, Any]) -> str | None:
    for aka in doc.get("alsoKnownAs") or []:
        if isinstance(aka, str) and aka.startswith("at://"):
            return aka.removeprefix("at://")
    return None


@dataclass
class _Identity:
    did: str
    handle: str | None
    pds: str


async def _resolve_identity(ctx: ToolContext, identifier: str) -> _Identity:
    identifier = _check_identifier(identifier, "identifier")
    did = identifier if identifier.startswith("did:") else await _resolve_handle(ctx, identifier)
    doc = await _fetch_did_document(ctx, did)
    return _Identity(did=did, handle=_handle_from_doc(doc), pds=_pds_from_doc(doc))


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


@dataclass
class _Session:
    key: tuple[str, str, str]  # (identifier, password, pds override) it was made with
    did: str
    handle: str
    pds: str
    access_jwt: str
    refresh_jwt: str


# One cached session per process. createSession is heavily rate-limited on
# Bluesky's PDSes, so logging in on every call is not an option.
_session: _Session | None = None
_session_lock = asyncio.Lock()


def _reset_session_cache() -> None:
    """Forget the cached session (used by tests, which each run their own loop)."""
    global _session, _session_lock
    _session = None
    _session_lock = asyncio.Lock()


def _read_credentials(ctx: ToolContext) -> tuple[str, str, str]:
    sm = ctx.secret_manager
    if sm is None:
        raise AtprotoError("Secret manager not available")
    identifier = sm.get_secret(_IDENTIFIER_SECRET) or ""
    password = sm.get_secret(_PASSWORD_SECRET) or ""
    pds_override = (sm.get_secret(_PDS_URL_SECRET) or "").strip().rstrip("/")
    missing = [
        name
        for name, val in ((_IDENTIFIER_SECRET, identifier), (_PASSWORD_SECRET, password))
        if not val
    ]
    if missing:
        raise AtprotoError(
            "AT Protocol account not configured. Ask the operator to set these "
            "secrets on the web interface's Secrets page: "
            + ", ".join(f"`{m}`" for m in missing)
        )
    if pds_override and urlparse(pds_override).scheme not in ("https", "http"):
        raise AtprotoError(f"{_PDS_URL_SECRET} must be an http(s) URL")
    return identifier.strip(), password, pds_override


def _session_from_response(key: tuple[str, str, str], pds: str, data: Any) -> _Session:
    if not isinstance(data, dict):
        raise AtprotoError("session response was not a JSON object")
    fields = {k: data.get(k) for k in ("did", "handle", "accessJwt", "refreshJwt")}
    if not all(isinstance(v, str) and v for v in fields.values()):
        raise AtprotoError("session response is missing did/handle/tokens")
    return _Session(
        key=key,
        did=fields["did"],
        handle=fields["handle"],
        pds=pds,
        access_jwt=fields["accessJwt"],
        refresh_jwt=fields["refreshJwt"],
    )


async def _login(ctx: ToolContext, key: tuple[str, str, str]) -> _Session:
    identifier, password, pds_override = key
    pds = pds_override or (await _resolve_identity(ctx, identifier)).pds
    resp = await _request(
        ctx,
        "POST",
        f"{pds}/xrpc/com.atproto.server.createSession",
        "logging in",
        json={"identifier": identifier, "password": password},
    )
    session = _session_from_response(key, pds, _json_or_raise(resp, "logging in"))
    logger.info("atproto: logged in as %s (%s) via %s", session.handle, session.did, pds)
    return session


async def _get_session(ctx: ToolContext) -> _Session:
    global _session
    key = _read_credentials(ctx)
    async with _session_lock:
        if _session is None or _session.key != key:
            _session = await _login(ctx, key)
        return _session


async def _renew_session(ctx: ToolContext, stale: _Session) -> _Session:
    """Refresh the session after an expired-token error, or log in again."""
    global _session
    async with _session_lock:
        if _session is not None and _session is not stale:
            return _session  # another call already renewed it
        resp = await _request(
            ctx,
            "POST",
            f"{stale.pds}/xrpc/com.atproto.server.refreshSession",
            "refreshing session",
            headers={"Authorization": f"Bearer {stale.refresh_jwt}"},
        )
        if resp.status_code < 400:
            _session = _session_from_response(
                stale.key, stale.pds, _json_or_raise(resp, "refreshing session")
            )
        else:
            _session = await _login(ctx, stale.key)
        return _session


async def _authed_request(
    ctx: ToolContext,
    method: str,
    nsid: str,
    what: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: Any = None,
    content: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    """Make an authenticated XRPC call to the agent's PDS, renewing once if expired."""
    session = await _get_session(ctx)
    for attempt in range(2):
        req_headers = dict(headers or {})
        req_headers["Authorization"] = f"Bearer {session.access_jwt}"
        kwargs: dict[str, Any] = {"headers": req_headers}
        if params:
            kwargs["params"] = params
        if content is not None:
            kwargs["content"] = content
        elif json_body is not None:
            kwargs["json"] = json_body
        resp = await _request(ctx, method, f"{session.pds}/xrpc/{nsid}", what, **kwargs)
        if attempt == 0 and _is_expired(resp):
            session = await _renew_session(ctx, session)
            continue
        return _json_or_raise(resp, what)
    raise AssertionError("unreachable")


def _clean_params(params: Any) -> dict[str, Any] | None:
    if params is None:
        return None
    if not isinstance(params, dict):
        raise AtprotoError("params must be an object of query parameters")
    cleaned: dict[str, Any] = {}
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, bool):
            cleaned[k] = "true" if v else "false"
        else:
            cleaned[k] = v
    return cleaned


def _with_type(collection: str, record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise AtprotoError("record must be a JSON object")
    if "$type" not in record:
        record = {"$type": collection, **record}
    return record


async def _repo_location(ctx: ToolContext, repo: Any) -> tuple[str, str]:
    """Return (repo identifier to send, base URL of the PDS hosting it)."""
    if repo is None or repo == "":
        session = await _get_session(ctx)
        return session.did, session.pds
    repo = _check_identifier(repo)
    # Avoid resolution (and login) round-trips for the agent's own repo when
    # a session already exists.
    if _session is not None and repo in (_session.did, _session.handle):
        return _session.did, _session.pds
    identity = await _resolve_identity(ctx, repo)
    return identity.did, identity.pds


def _tool(func):
    """Turn AtprotoError into the agent-facing {"error": ...} result."""

    async def wrapper(ctx: ToolContext, *args: Any, **kwargs: Any) -> Any:
        try:
            return await func(ctx, *args, **kwargs)
        except AtprotoError as e:
            return e.to_result()

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@TOOL_REGISTRY.tool(
    name="atproto.whoami",
    description=(
        "Return the AT Protocol account you act as: its DID, handle, and PDS URL. "
        "Logs in if needed. Fails with setup instructions if the operator hasn't "
        "configured the account."
    ),
    parameters=[],
)
@_tool
async def whoami(ctx: ToolContext) -> dict[str, Any]:
    session = await _get_session(ctx)
    return {"did": session.did, "handle": session.handle, "pds": session.pds}


@TOOL_REGISTRY.tool(
    name="atproto.resolve",
    description=(
        "Resolve any handle or DID on the AT Protocol network to its DID, its "
        "handle (as claimed in the DID document), and the URL of the PDS that "
        "hosts its repo. Does not require the account to be configured."
    ),
    parameters=[
        ToolParameter(
            name="identifier",
            type="string",
            description="A handle (e.g. 'alice.bsky.social') or DID (e.g. 'did:plc:...').",
        ),
    ],
)
@_tool
async def resolve(ctx: ToolContext, identifier: str) -> dict[str, Any]:
    identity = await _resolve_identity(ctx, identifier)
    return {"did": identity.did, "handle": identity.handle, "pds": identity.pds}


@TOOL_REGISTRY.tool(
    name="atproto.create_record",
    description=(
        "Create a record in your own repo (com.atproto.repo.createRecord). "
        "Returns its `uri` (at://...) and `cid`. If the record has no `$type`, "
        "it is set to the collection. Records must follow the collection's "
        "Lexicon; e.g. an 'app.bsky.feed.post' needs `text` and an ISO 8601 "
        "`createdAt`. To attach images, upload them first with "
        "atproto.upload_blob and put the returned blob in the record."
    ),
    parameters=[
        ToolParameter(
            name="collection",
            type="string",
            description="Collection NSID, e.g. 'app.bsky.feed.post'.",
        ),
        ToolParameter(name="record", type="object", description="The record body."),
        ToolParameter(
            name="rkey",
            type="string",
            description="Record key. Omit to let the PDS generate one (a TID).",
            required=False,
        ),
        ToolParameter(
            name="validate",
            type="boolean",
            description=(
                "true: require Lexicon validation; false: skip it; omit: validate "
                "only if the PDS knows the Lexicon."
            ),
            required=False,
        ),
    ],
)
@_tool
async def create_record(
    ctx: ToolContext,
    collection: str,
    record: dict[str, Any],
    rkey: str | None = None,
    validate: bool | None = None,
) -> Any:
    collection = _check_nsid(collection, "collection")
    session = await _get_session(ctx)
    body: dict[str, Any] = {
        "repo": session.did,
        "collection": collection,
        "record": _with_type(collection, record),
    }
    if rkey is not None:
        body["rkey"] = _check_rkey(rkey)
    if validate is not None:
        body["validate"] = bool(validate)
    return await _authed_request(
        ctx, "POST", "com.atproto.repo.createRecord", "createRecord", json_body=body
    )


@TOOL_REGISTRY.tool(
    name="atproto.put_record",
    description=(
        "Create or overwrite the record at a given key in your own repo "
        "(com.atproto.repo.putRecord). Use this to update a record: pass the "
        "full new record, not a patch. Pass `swap_record` (the CID you last "
        "read) to fail instead of overwriting if it changed since."
    ),
    parameters=[
        ToolParameter(name="collection", type="string", description="Collection NSID."),
        ToolParameter(name="rkey", type="string", description="Record key."),
        ToolParameter(name="record", type="object", description="The full new record body."),
        ToolParameter(
            name="swap_record",
            type="string",
            description="Only write if the current record has this CID.",
            required=False,
        ),
        ToolParameter(
            name="validate",
            type="boolean",
            description="Same as for atproto.create_record.",
            required=False,
        ),
    ],
)
@_tool
async def put_record(
    ctx: ToolContext,
    collection: str,
    rkey: str,
    record: dict[str, Any],
    swap_record: str | None = None,
    validate: bool | None = None,
) -> Any:
    collection = _check_nsid(collection, "collection")
    session = await _get_session(ctx)
    body: dict[str, Any] = {
        "repo": session.did,
        "collection": collection,
        "rkey": _check_rkey(rkey),
        "record": _with_type(collection, record),
    }
    if swap_record:
        body["swapRecord"] = swap_record
    if validate is not None:
        body["validate"] = bool(validate)
    return await _authed_request(
        ctx, "POST", "com.atproto.repo.putRecord", "putRecord", json_body=body
    )


@TOOL_REGISTRY.tool(
    name="atproto.delete_record",
    description=(
        "Delete a record from your own repo (com.atproto.repo.deleteRecord). "
        "Deleting a record that doesn't exist succeeds. Pass `swap_record` to "
        "only delete if the record still has that CID."
    ),
    parameters=[
        ToolParameter(name="collection", type="string", description="Collection NSID."),
        ToolParameter(name="rkey", type="string", description="Record key."),
        ToolParameter(
            name="swap_record",
            type="string",
            description="Only delete if the current record has this CID.",
            required=False,
        ),
    ],
)
@_tool
async def delete_record(
    ctx: ToolContext, collection: str, rkey: str, swap_record: str | None = None
) -> Any:
    collection = _check_nsid(collection, "collection")
    session = await _get_session(ctx)
    body: dict[str, Any] = {
        "repo": session.did,
        "collection": collection,
        "rkey": _check_rkey(rkey),
    }
    if swap_record:
        body["swapRecord"] = swap_record
    result = await _authed_request(
        ctx, "POST", "com.atproto.repo.deleteRecord", "deleteRecord", json_body=body
    )
    return {"success": True, **(result if isinstance(result, dict) else {})}


@TOOL_REGISTRY.tool(
    name="atproto.get_record",
    description=(
        "Fetch one record from any repo on the network "
        "(com.atproto.repo.getRecord), read directly from the PDS hosting it. "
        "Returns `uri`, `cid`, and `value`. For an at:// URI "
        "'at://<repo>/<collection>/<rkey>', pass its three parts."
    ),
    parameters=[
        ToolParameter(name="collection", type="string", description="Collection NSID."),
        ToolParameter(name="rkey", type="string", description="Record key."),
        ToolParameter(
            name="repo",
            type="string",
            description="Handle or DID of the repo. Omit for your own repo.",
            required=False,
        ),
        ToolParameter(
            name="cid",
            type="string",
            description="Fetch a specific version of the record by CID.",
            required=False,
        ),
    ],
)
@_tool
async def get_record(
    ctx: ToolContext,
    collection: str,
    rkey: str,
    repo: str | None = None,
    cid: str | None = None,
) -> Any:
    collection = _check_nsid(collection, "collection")
    rkey = _check_rkey(rkey)
    repo_did, pds = await _repo_location(ctx, repo)
    params = {"repo": repo_did, "collection": collection, "rkey": rkey}
    if cid:
        params["cid"] = cid
    resp = await _request(
        ctx, "GET", f"{pds}/xrpc/com.atproto.repo.getRecord", "getRecord", params=params
    )
    return _json_or_raise(resp, "getRecord")


@TOOL_REGISTRY.tool(
    name="atproto.list_records",
    description=(
        "List records in one collection of any repo on the network "
        "(com.atproto.repo.listRecords), newest first by default. Returns "
        "`records` (each with `uri`, `cid`, `value`) and a `cursor` to pass back "
        "for the next page when there are more."
    ),
    parameters=[
        ToolParameter(name="collection", type="string", description="Collection NSID."),
        ToolParameter(
            name="repo",
            type="string",
            description="Handle or DID of the repo. Omit for your own repo.",
            required=False,
        ),
        ToolParameter(
            name="limit",
            type="number",
            description=f"Records per page, 1-{_MAX_LIST_LIMIT}.",
            required=False,
            default=50,
        ),
        ToolParameter(
            name="cursor",
            type="string",
            description="Cursor from a previous page.",
            required=False,
        ),
        ToolParameter(
            name="reverse",
            type="boolean",
            description="List oldest first instead.",
            required=False,
            default=False,
        ),
    ],
)
@_tool
async def list_records(
    ctx: ToolContext,
    collection: str,
    repo: str | None = None,
    limit: int = 50,
    cursor: str | None = None,
    reverse: bool = False,
) -> Any:
    collection = _check_nsid(collection, "collection")
    try:
        limit = max(1, min(_MAX_LIST_LIMIT, int(limit)))
    except (TypeError, ValueError):
        raise AtprotoError(f"limit must be a number, got {limit!r}")
    repo_did, pds = await _repo_location(ctx, repo)
    params: dict[str, Any] = {"repo": repo_did, "collection": collection, "limit": limit}
    if cursor:
        params["cursor"] = cursor
    if reverse:
        params["reverse"] = "true"
    resp = await _request(
        ctx, "GET", f"{pds}/xrpc/com.atproto.repo.listRecords", "listRecords", params=params
    )
    return _json_or_raise(resp, "listRecords")


@TOOL_REGISTRY.tool(
    name="atproto.upload_blob",
    description=(
        "Upload a workspace file (e.g. an image) to your PDS "
        "(com.atproto.repo.uploadBlob). Returns `blob`: put that object inside a "
        "record (e.g. in an app.bsky.embed.images embed) to attach the file. "
        "Blobs that no record references are garbage-collected by the PDS. "
        "PDSes and apps enforce their own size limits (Bluesky images: about 1 MB)."
    ),
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Workspace-relative path of the file, e.g. 'photo.jpg'.",
        ),
        ToolParameter(
            name="mime_type",
            type="string",
            description="MIME type. Guessed from the file extension if omitted.",
            required=False,
        ),
    ],
)
@_tool
async def upload_blob(ctx: ToolContext, path: str, mime_type: str | None = None) -> Any:
    try:
        data, name = _read_workspace_file(path)
    except ValueError as e:
        raise AtprotoError(str(e))
    mime = mime_type or mimetypes.guess_type(name)[0] or "application/octet-stream"
    return await _authed_request(
        ctx,
        "POST",
        "com.atproto.repo.uploadBlob",
        "uploadBlob",
        content=data,
        headers={"Content-Type": mime},
    )


_PROXY_PARAM = ToolParameter(
    name="proxy",
    type="string",
    description=(
        "Service to forward the call to via your PDS, as 'did#service_id' "
        "(the atproto-proxy header). Defaults: app.bsky.* -> "
        "'did:web:api.bsky.app#bsky_appview', chat.bsky.* -> "
        "'did:web:api.bsky.chat#bsky_chat' (DMs need an app password with "
        "direct message access); anything else goes to the PDS itself. Pass "
        "'none' to force the PDS itself."
    ),
    required=False,
)


@TOOL_REGISTRY.tool(
    name="atproto.query",
    description=(
        "Call any XRPC query (HTTP GET) as your account, through your PDS. "
        "Use this for everything the record tools don't cover: e.g. "
        "'app.bsky.notification.listNotifications', "
        "'app.bsky.feed.getTimeline', 'app.bsky.actor.getProfile', or any other "
        "app's or service's API. Returns the response JSON."
    ),
    parameters=[
        ToolParameter(
            name="nsid",
            type="string",
            description="Method NSID, e.g. 'app.bsky.notification.listNotifications'.",
        ),
        ToolParameter(
            name="params",
            type="object",
            description="Query parameters. Arrays become repeated parameters.",
            required=False,
        ),
        _PROXY_PARAM,
    ],
)
@_tool
async def query(
    ctx: ToolContext,
    nsid: str,
    params: dict[str, Any] | None = None,
    proxy: str | None = None,
) -> Any:
    nsid = _check_nsid(nsid, "nsid")
    proxy_header = _resolve_proxy(nsid, proxy)
    return await _authed_request(
        ctx,
        "GET",
        nsid,
        nsid,
        params=_clean_params(params),
        headers={"atproto-proxy": proxy_header} if proxy_header else None,
    )


@TOOL_REGISTRY.tool(
    name="atproto.procedure",
    description=(
        "Call any XRPC procedure (HTTP POST with a JSON body) as your account, "
        "through your PDS. E.g. 'app.bsky.notification.updateSeen' or "
        "'com.atproto.repo.applyWrites'. Prefer the dedicated record tools for "
        "single record writes. Returns the response JSON (or {} if empty)."
    ),
    parameters=[
        ToolParameter(name="nsid", type="string", description="Method NSID."),
        ToolParameter(
            name="input",
            type="object",
            description="JSON request body. Omit for procedures without input.",
            required=False,
        ),
        ToolParameter(
            name="params",
            type="object",
            description="Query parameters, if the procedure takes any.",
            required=False,
        ),
        _PROXY_PARAM,
    ],
)
@_tool
async def procedure(
    ctx: ToolContext,
    nsid: str,
    input: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    proxy: str | None = None,
) -> Any:
    nsid = _check_nsid(nsid, "nsid")
    if input is not None and not isinstance(input, dict):
        raise AtprotoError("input must be a JSON object")
    proxy_header = _resolve_proxy(nsid, proxy)
    return await _authed_request(
        ctx,
        "POST",
        nsid,
        nsid,
        params=_clean_params(params),
        json_body=input,
        headers={"atproto-proxy": proxy_header} if proxy_header else None,
    )

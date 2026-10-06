"""Tests for bluesky.get_notifications against a fake AppView.

The fake mirrors the real AppView's rules: a notification is read when
``seenAt > indexedAt`` (strict), results are newest first with cursors, and
updateSeen stores the given time as-is (it can move backwards).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from src.tools.definitions import atproto as ap
from src.tools.definitions import bluesky as bs
from src.tools.registry import TOOL_REGISTRY, ToolContext

PDS = "https://pds.example.com"
T0 = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class FakeAppView:
    def __init__(self, count: int, seen_at: datetime | None) -> None:
        # notifications[i] is i seconds after T0; listed newest first.
        self.notifs = [
            {
                "uri": f"at://did:plc:fan{i}/app.bsky.feed.like/{i}",
                "cid": f"bafy{i}",
                "author": {
                    "did": f"did:plc:fan{i}",
                    "handle": f"fan{i}.test",
                    "displayName": f"Fan {i}",
                    "avatar": "https://cdn.example/big.jpg",
                    "viewer": {"muted": False},
                    "labels": [],
                },
                "reason": "like",
                "reasonSubject": "at://did:plc:agent/app.bsky.feed.post/abc",
                "record": {"$type": "app.bsky.feed.like"},
                "indexedAt": _iso(T0 + timedelta(seconds=i)),
                "labels": [],
            }
            for i in range(count)
        ]
        self.seen_at = seen_at
        self.update_calls: list[str] = []
        self.list_calls = 0
        self.fail_update = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/xrpc/com.atproto.server.createSession":
            return httpx.Response(
                200,
                json={"did": "did:plc:agent", "handle": "agent.test", "accessJwt": "a", "refreshJwt": "r"},
            )
        assert request.headers.get("authorization") == "Bearer a"
        assert request.headers.get("atproto-proxy") == "did:web:api.bsky.app#bsky_appview"
        if path == "/xrpc/app.bsky.notification.listNotifications":
            self.list_calls += 1
            limit = int(request.url.params["limit"])
            start = int(request.url.params.get("cursor", "0"))
            newest_first = list(reversed(self.notifs))
            page = newest_first[start : start + limit]
            items = []
            for n in page:
                indexed = datetime.fromisoformat(n["indexedAt"])
                items.append({**n, "isRead": self.seen_at is not None and self.seen_at > indexed})
            body: dict[str, Any] = {"notifications": items}
            if start + limit < len(newest_first):
                body["cursor"] = str(start + limit)
            if self.seen_at is not None:
                body["seenAt"] = _iso(self.seen_at)
            return httpx.Response(200, json=body)
        if path == "/xrpc/app.bsky.notification.updateSeen":
            if self.fail_update:
                return httpx.Response(500, json={"error": "InternalServerError"})
            seen = json.loads(request.content)["seenAt"]
            self.update_calls.append(seen)
            self.seen_at = datetime.fromisoformat(seen)
            return httpx.Response(200)
        return httpx.Response(404)

    def unread_count(self) -> int:
        return sum(
            1
            for n in self.notifs
            if not (self.seen_at and self.seen_at > datetime.fromisoformat(n["indexedAt"]))
        )


@pytest.fixture(autouse=True)
def _reset():
    ap._reset_session_cache()
    bs._reset_lock()
    yield
    ap._reset_session_cache()


def _ctx(view: FakeAppView) -> Any:
    ctx = MagicMock(spec=ToolContext)
    ctx.http_client = httpx.AsyncClient(transport=httpx.MockTransport(view))
    secrets = {
        "ATPROTO_IDENTIFIER": "agent.test",
        "ATPROTO_APP_PASSWORD": "pw",
        "ATPROTO_PDS_URL": PDS,
    }
    ctx.secret_manager = MagicMock(get_secret=lambda n: secrets.get(n))
    return ctx


def test_registered():
    assert TOOL_REGISTRY.get("bluesky.get_notifications") is not None


async def test_returns_unread_and_marks_them_read():
    # 10 notifications, the first 4 already read.
    view = FakeAppView(10, seen_at=T0 + timedelta(seconds=3, milliseconds=500))
    result = await bs.get_notifications(_ctx(view))
    assert [n["cid"] for n in result["notifications"]] == [f"bafy{i}" for i in range(4, 10)]
    assert result["marked_read"] is True
    assert result["more_unread"] is False
    assert view.unread_count() == 0
    # Marker is exactly 1 ms past the newest returned notification.
    assert view.update_calls == [_iso(T0 + timedelta(seconds=9, milliseconds=1))]
    assert result["seen_at"] == view.update_calls[0]


async def test_second_call_returns_nothing_and_does_not_update():
    view = FakeAppView(5, seen_at=T0 - timedelta(seconds=1))
    ctx = _ctx(view)
    await bs.get_notifications(ctx)
    result = await bs.get_notifications(ctx)
    assert result["notifications"] == []
    assert result["marked_read"] is False
    assert len(view.update_calls) == 1


async def test_oldest_first_batches_never_skip_unread():
    """With more unread than `limit`, nothing is marked read without being returned."""
    view = FakeAppView(25, seen_at=T0 - timedelta(seconds=1))
    ctx = _ctx(view)
    seen: list[str] = []
    for _ in range(3):
        result = await bs.get_notifications(ctx, limit=10)
        seen += [n["cid"] for n in result["notifications"]]
        # Everything still unread is newer than everything returned so far.
        assert view.unread_count() == 25 - len(seen)
    assert seen == [f"bafy{i}" for i in range(25)]
    assert result["more_unread"] is False


async def test_more_unread_flag():
    view = FakeAppView(15, seen_at=T0 - timedelta(seconds=1))
    result = await bs.get_notifications(_ctx(view), limit=10)
    assert result["more_unread"] is True
    assert view.unread_count() == 5


async def test_pages_back_to_find_oldest_unread():
    # 250 unread spans three 100-item pages.
    view = FakeAppView(250, seen_at=T0 - timedelta(seconds=1))
    result = await bs.get_notifications(_ctx(view), limit=5)
    assert [n["cid"] for n in result["notifications"]] == [f"bafy{i}" for i in range(5)]
    assert view.list_calls == 3


async def test_scan_cap_reports_skipped_older_unread():
    view = FakeAppView(bs._MAX_PAGES * bs._PAGE_SIZE + 50, seen_at=None)
    result = await bs.get_notifications(_ctx(view), limit=5)
    assert "note" in result
    assert view.list_calls == bs._MAX_PAGES
    # Returned batch is the oldest of what was scanned (the newest 1000).
    assert result["notifications"][0]["cid"] == "bafy50"


async def test_mark_read_failure_still_returns_notifications():
    view = FakeAppView(3, seen_at=T0 - timedelta(seconds=1))
    view.fail_update = True
    result = await bs.get_notifications(_ctx(view))
    assert len(result["notifications"]) == 3
    assert result["marked_read"] is False
    assert "mark_read_error" in result
    assert view.unread_count() == 3


async def test_notifications_are_compacted():
    view = FakeAppView(1, seen_at=T0 - timedelta(seconds=1))
    result = await bs.get_notifications(_ctx(view))
    n = result["notifications"][0]
    assert n["author"] == {"did": "did:plc:fan0", "handle": "fan0.test", "displayName": "Fan 0"}
    assert n["reason"] == "like"
    assert n["reason_subject"] == "at://did:plc:agent/app.bsky.feed.post/abc"
    assert n["record"] == {"$type": "app.bsky.feed.like"}
    assert "avatar" not in json.dumps(n)


async def test_list_failure_is_reported():
    def broken(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("createSession"):
            return httpx.Response(
                200, json={"did": "did:plc:agent", "handle": "a.test", "accessJwt": "a", "refreshJwt": "r"}
            )
        return httpx.Response(502, json={"error": "UpstreamFailure"})

    ctx = _ctx(FakeAppView(0, None))
    ctx.http_client = httpx.AsyncClient(transport=httpx.MockTransport(broken))
    result = await bs.get_notifications(ctx)
    assert result["status"] == 502

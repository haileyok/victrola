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
from src.tools.definitions import delve
from src.tools.registry import TOOL_REGISTRY, ToolContext

PDS = "https://pds.example.com"
T0 = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class FakeAppView:
    def __init__(
        self,
        count: int,
        seen_at: datetime | None,
        reasons: list[str] | None = None,
        prefix: str = "app.bsky.notification",
        proxy: str = "did:web:api.bsky.app#bsky_appview",
    ) -> None:
        self.prefix = prefix
        self.proxy = proxy
        # notifications[i] is i seconds after T0; listed newest first.
        # reasons[i] is notification i's reason (default: all replies).
        reasons = reasons or ["reply"] * count
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
                "reason": reasons[i],
                "reasonSubject": "at://did:plc:agent/app.bsky.feed.post/abc",
                "record": {"$type": "app.bsky.feed.post", "text": f"hi {i}"},
                "indexedAt": _iso(T0 + timedelta(seconds=i)),
                "labels": [],
            }
            for i in range(count)
        ]
        self.seen_at = seen_at
        self.update_calls: list[str] = []
        self.list_calls = 0
        self.fail_update = False
        self.unread_count_calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/xrpc/com.atproto.server.createSession":
            return httpx.Response(
                200,
                json={"did": "did:plc:agent", "handle": "agent.test", "accessJwt": "a", "refreshJwt": "r"},
            )
        assert request.headers.get("authorization") == "Bearer a"
        assert request.headers.get("atproto-proxy") == self.proxy
        if path == f"/xrpc/{self.prefix}.listNotifications":
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
        if path == f"/xrpc/{self.prefix}.getUnreadCount":
            self.unread_count_calls += 1
            return httpx.Response(200, json={"count": self.unread_count()})
        if path == f"/xrpc/{self.prefix}.updateSeen":
            if self.fail_update:
                return httpx.Response(500, json={"error": "InternalServerError"})
            seen = json.loads(request.content)["seenAt"]
            self.update_calls.append(seen)
            self.seen_at = datetime.fromisoformat(seen)
            return httpx.Response(200)
        return httpx.Response(404)

    def is_unread(self, n: dict[str, Any]) -> bool:
        return not (self.seen_at and self.seen_at > datetime.fromisoformat(n["indexedAt"]))

    def unread_count(self, reasons: tuple[str, ...] | None = None) -> int:
        return sum(1 for n in self.notifs if self.is_unread(n) and (reasons is None or n["reason"] in reasons))


@pytest.fixture(autouse=True)
def _reset():
    ap._reset_session_cache()
    bs._reset_lock()
    delve._reset_lock()
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
    assert n["reason"] == "reply"
    assert n["reason_subject"] == "at://did:plc:agent/app.bsky.feed.post/abc"
    assert n["record"] == {"$type": "app.bsky.feed.post", "text": "hi 0"}
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


# --- post-only filtering --------------------------------------------------------

MIXED = ["like", "reply", "follow", "mention", "like", "quote", "repost", "like"]


async def test_only_post_notifications_returned():
    view = FakeAppView(len(MIXED), seen_at=T0 - timedelta(seconds=1), reasons=MIXED)
    result = await bs.get_notifications(_ctx(view))
    assert [n["reason"] for n in result["notifications"]] == ["reply", "mention", "quote"]
    # Marked read through the newest post (index 5); the like at index 7 and
    # repost at 6 are newer, so they stay unread.
    assert view.update_calls == [_iso(T0 + timedelta(seconds=5, milliseconds=1))]
    assert view.unread_count() == 2
    assert view.unread_count(bs.POST_REASONS) == 0


async def test_no_post_notifications_marks_nothing():
    view = FakeAppView(4, seen_at=T0 - timedelta(seconds=1), reasons=["like", "follow", "repost", "like"])
    result = await bs.get_notifications(_ctx(view))
    assert result["notifications"] == []
    assert result["marked_read"] is False
    assert view.update_calls == []


async def test_post_batches_never_skip_unread_posts():
    reasons = ["like", "reply"] * 15  # 15 replies interleaved with 15 likes
    view = FakeAppView(len(reasons), seen_at=T0 - timedelta(seconds=1), reasons=reasons)
    ctx = _ctx(view)
    seen: list[str] = []
    for _ in range(3):
        result = await bs.get_notifications(ctx, limit=5)
        seen += [n["cid"] for n in result["notifications"]]
        assert view.unread_count(bs.POST_REASONS) == 15 - len(seen)
    assert seen == [f"bafy{i}" for i in range(1, 30, 2)]
    assert result["more_unread"] is False


async def test_reasons_override():
    view = FakeAppView(len(MIXED), seen_at=T0 - timedelta(seconds=1), reasons=MIXED)
    result = await bs.get_notifications(_ctx(view), reasons=["like"])
    assert [n["cid"] for n in result["notifications"]] == ["bafy0", "bafy4", "bafy7"]


@pytest.mark.parametrize("bad", [[], "reply", [""], [1]])
async def test_bad_reasons_rejected(bad):
    view = FakeAppView(1, seen_at=None)
    result = await bs.get_notifications(_ctx(view), reasons=bad)
    assert "reasons" in result["error"]
    assert view.list_calls == 0


async def test_unread_count_counts_only_posts():
    view = FakeAppView(len(MIXED), seen_at=T0 - timedelta(seconds=1), reasons=MIXED)
    assert await bs.unread_count(_ctx(view)) == {"count": 3}
    assert view.update_calls == []  # never marks read


async def test_unread_count_zero_when_only_likes():
    view = FakeAppView(3, seen_at=T0 - timedelta(seconds=1), reasons=["like", "follow", "like"])
    assert await bs.unread_count(_ctx(view)) == {"count": 0}


async def test_unread_count_skips_scan_when_nothing_unread():
    view = FakeAppView(5, seen_at=T0 + timedelta(hours=1))
    assert await bs.unread_count(_ctx(view)) == {"count": 0}
    assert view.unread_count_calls == 1
    assert view.list_calls == 0


async def test_unread_count_reasons_override():
    view = FakeAppView(len(MIXED), seen_at=T0 - timedelta(seconds=1), reasons=MIXED)
    assert await bs.unread_count(_ctx(view), reasons=["like", "follow"]) == {"count": 4}


# --- delve.town ---------------------------------------------------------------

DELVE_PREFIX = "town.delve.notification"
DELVE_PROXY = "did:web:api.delve.town#bsky_appview"


def _delve_view(count: int, seen_at: datetime | None, reasons: list[str] | None = None) -> FakeAppView:
    # The fake asserts every call uses town.delve.* paths and the delve proxy;
    # any app.bsky.* call would 404.
    return FakeAppView(count, seen_at, reasons, prefix=DELVE_PREFIX, proxy=DELVE_PROXY)


def test_delve_tools_registered():
    assert TOOL_REGISTRY.get("delve.get_notifications") is not None
    assert TOOL_REGISTRY.get("delve.unread_count").condition_safe is True


async def test_delve_get_notifications_posts_only_and_marks_read():
    view = _delve_view(len(MIXED), seen_at=T0 - timedelta(seconds=1), reasons=MIXED)
    result = await delve.get_notifications(_ctx(view))
    assert [n["reason"] for n in result["notifications"]] == ["reply", "mention", "quote"]
    assert result["marked_read"] is True
    assert view.update_calls == [_iso(T0 + timedelta(seconds=5, milliseconds=1))]
    assert view.unread_count(bs.POST_REASONS) == 0


async def test_delve_batches_never_skip_unread():
    view = _delve_view(25, seen_at=T0 - timedelta(seconds=1))
    ctx = _ctx(view)
    seen: list[str] = []
    for _ in range(3):
        result = await delve.get_notifications(ctx, limit=10)
        seen += [n["cid"] for n in result["notifications"]]
        assert view.unread_count() == 25 - len(seen)
    assert seen == [f"bafy{i}" for i in range(25)]


async def test_delve_unread_count():
    view = _delve_view(len(MIXED), seen_at=T0 - timedelta(seconds=1), reasons=MIXED)
    assert await delve.unread_count(_ctx(view)) == {"count": 3}
    assert view.update_calls == []


async def test_delve_and_bluesky_read_markers_are_separate():
    """Reading delve notifications must not touch Bluesky's (and vice versa)."""
    bsky_view = FakeAppView(3, seen_at=T0 - timedelta(seconds=1))
    delve_view = _delve_view(3, seen_at=T0 - timedelta(seconds=1))
    await delve.get_notifications(_ctx(delve_view))
    assert delve_view.unread_count() == 0
    assert bsky_view.update_calls == [] and bsky_view.unread_count() == 3


def test_delve_description_points_at_delve_lexicons():
    desc = TOOL_REGISTRY.get("delve.get_notifications").description
    assert "delve.town" in desc and "town.delve.notification.listNotifications" in desc
    assert "app.bsky" not in desc

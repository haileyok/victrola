"""Bluesky-specific convenience tools, built on the atproto session.

Most Bluesky behavior is left to agent-written custom tools that call
``tools.atproto.*``. Notifications are the exception: marking them read
correctly is subtle enough (see ``get_notifications``) to be worth a built-in.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from src.tools.definitions.atproto import (
    _DEFAULT_PROXIES,
    AtprotoError,
    _authed_request,
    _tool,
)
from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter

_APPVIEW_PROXY = _DEFAULT_PROXIES["app.bsky."]
_LIST_NSID = "app.bsky.notification.listNotifications"
_UPDATE_SEEN_NSID = "app.bsky.notification.updateSeen"
_UNREAD_NSID = "app.bsky.notification.getUnreadCount"

_PAGE_SIZE = 100
# How far back to page looking for the oldest unread notification.
_MAX_PAGES = 10
_MAX_LIMIT = 100

# Serializes calls so two concurrent runs can't return the same notifications.
_lock = asyncio.Lock()


def _reset_lock() -> None:
    """Recreate the lock (used by tests, which each run their own event loop)."""
    global _lock
    _lock = asyncio.Lock()


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _format_time(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _compact(notif: dict[str, Any]) -> dict[str, Any]:
    """Keep the fields an agent needs; full views are large (avatars, labels, viewer state)."""
    author = notif.get("author") or {}
    out: dict[str, Any] = {
        "reason": notif.get("reason"),
        "uri": notif.get("uri"),
        "cid": notif.get("cid"),
        "author": {
            k: author[k] for k in ("did", "handle", "displayName") if author.get(k)
        },
        "indexed_at": notif.get("indexedAt"),
        "record": notif.get("record"),
    }
    if notif.get("reasonSubject"):
        out["reason_subject"] = notif["reasonSubject"]
    return out


async def _list_page(ctx: ToolContext, cursor: str | None) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": _PAGE_SIZE}
    if cursor:
        params["cursor"] = cursor
    data = await _authed_request(
        ctx,
        "GET",
        _LIST_NSID,
        _LIST_NSID,
        params=params,
        headers={"atproto-proxy": _APPVIEW_PROXY},
    )
    if not isinstance(data, dict) or not isinstance(data.get("notifications"), list):
        raise AtprotoError(f"{_LIST_NSID} returned an unexpected response")
    return data


@TOOL_REGISTRY.tool(
    name="bluesky.unread_count",
    description=(
        "Return how many unread Bluesky notifications your account has, as "
        "{\"count\": n}, without marking anything read. Read-only, so schedule "
        "condition scripts can call it to wake you only when there's something "
        "new: e.g. `const r = await tools.bluesky.unread_count(); "
        "output({ wake: r.count > 0 });`. Use bluesky.get_notifications to read "
        "(and mark read) the notifications themselves."
    ),
    parameters=[],
    condition_safe=True,
)
@_tool
async def unread_count(ctx: ToolContext) -> dict[str, Any]:
    data = await _authed_request(
        ctx,
        "GET",
        _UNREAD_NSID,
        _UNREAD_NSID,
        headers={"atproto-proxy": _APPVIEW_PROXY},
    )
    count = data.get("count") if isinstance(data, dict) else None
    if not isinstance(count, int):
        raise AtprotoError(f"{_UNREAD_NSID} returned an unexpected response")
    return {"count": count}


@TOOL_REGISTRY.tool(
    name="bluesky.get_notifications",
    description=(
        "Get your Bluesky account's unread notifications and mark them as read. "
        "Returns the OLDEST unread notifications first (up to `limit`, oldest to "
        "newest), and marks read exactly those and anything older, so nothing is "
        "marked read without being returned. If `more_unread` is true, newer "
        "unread notifications remain: call again to get the next batch. Each "
        "notification has `reason` (like, repost, follow, mention, reply, quote, "
        "...), `uri`, `cid`, `author`, `indexed_at`, `record` (the liked/replied/"
        "etc. record's content), and `reason_subject` (the subject's at:// URI, "
        "e.g. your post that was liked). `marked_read` says whether the read "
        "marker was updated; if `mark_read_error` is present, the notifications "
        "were returned but NOT marked read. For read notifications or other "
        "filters, use atproto.query with app.bsky.notification.listNotifications."
    ),
    parameters=[
        ToolParameter(
            name="limit",
            type="number",
            description=f"Maximum notifications to return, 1-{_MAX_LIMIT}.",
            required=False,
            default=50,
        ),
    ],
)
@_tool
async def get_notifications(ctx: ToolContext, limit: int = 50) -> dict[str, Any]:
    try:
        limit = max(1, min(_MAX_LIMIT, int(limit)))
    except (TypeError, ValueError):
        raise AtprotoError(f"limit must be a number, got {limit!r}")

    async with _lock:
        # Page backwards (newest first) collecting unread notifications until we
        # reach one that's already read, or run out.
        unread: list[dict[str, Any]] = []
        server_seen_at: datetime | None = None
        cursor: str | None = None
        reached_read = False
        for _ in range(_MAX_PAGES):
            page = await _list_page(ctx, cursor)
            if server_seen_at is None:
                server_seen_at = _parse_time(page.get("seenAt"))
            for notif in page["notifications"]:
                if not isinstance(notif, dict):
                    continue
                if notif.get("isRead"):
                    reached_read = True
                    break
                unread.append(notif)
            cursor = page.get("cursor")
            if reached_read or not cursor:
                break
        # True when paging stopped at the cap while still seeing unread ones.
        older_unread_beyond_scan = not reached_read and bool(cursor)

        if not unread:
            return {
                "notifications": [],
                "more_unread": False,
                "marked_read": False,
                "seen_at": _format_time(server_seen_at) if server_seen_at else None,
            }

        # Oldest first; take the oldest `limit` and mark read through the newest
        # of those. The AppView treats a notification as read only when
        # seenAt > indexedAt, so the marker goes 1 ms past it.
        unread.reverse()
        batch = unread[:limit]
        more_unread = len(unread) > limit
        newest = max(
            (t for n in batch if (t := _parse_time(n.get("indexedAt"))) is not None),
            default=None,
        )

        result: dict[str, Any] = {
            "notifications": [_compact(n) for n in batch],
            "more_unread": more_unread,
            "marked_read": False,
        }
        if older_unread_beyond_scan:
            result["note"] = (
                f"More than {_MAX_PAGES * _PAGE_SIZE} unread notifications; the oldest "
                "beyond that were not returned and are now marked read."
            )

        if newest is None:
            result["mark_read_error"] = "notifications had no usable indexedAt timestamps"
            return result
        seen_at = newest + timedelta(milliseconds=1)
        # updateSeen stores the value as-is, so never move the marker backwards.
        if server_seen_at is not None and seen_at <= server_seen_at:
            result["marked_read"] = True
            result["seen_at"] = _format_time(server_seen_at)
            return result
        try:
            await _authed_request(
                ctx,
                "POST",
                _UPDATE_SEEN_NSID,
                _UPDATE_SEEN_NSID,
                json_body={"seenAt": _format_time(seen_at)},
                headers={"atproto-proxy": _APPVIEW_PROXY},
            )
        except AtprotoError as e:
            result["mark_read_error"] = str(e)
            return result
        result["marked_read"] = True
        result["seen_at"] = _format_time(seen_at)
        return result

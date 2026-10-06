"""Bluesky-specific convenience tools, built on the atproto session.

Most Bluesky behavior is left to agent-written custom tools that call
``tools.atproto.*``. Notifications are the exception: marking them read
correctly is subtle enough (see ``fetch_notifications``) to be worth a built-in.

The notification logic is shared with other AppViews that use the same
notification lexicons under a different NSID prefix (see delve.py).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from src.tools.definitions.atproto import (
    _DEFAULT_PROXIES,
    AtprotoError,
    _authed_request,
    _tool,
)
from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter

_PAGE_SIZE = 100
# How far back to page looking for the oldest unread notification.
_MAX_PAGES = 10
_MAX_LIMIT = 100

# By default the agent only gets notifications that are posts addressed to it
# or about its posts. Likes, reposts, follows, etc. are skipped: they never
# wake it, and get marked read along the way without being returned.
POST_REASONS = ("mention", "reply", "quote")


@dataclass
class NotificationService:
    """An AppView exposing the Bluesky-shaped notification lexicons.

    ``nsid_prefix`` is e.g. "app.bsky.notification"; ``proxy`` is the
    atproto-proxy target ("did#service_id") the PDS forwards calls to.
    """

    name: str
    nsid_prefix: str
    proxy: str
    # Serializes get_notifications calls so two concurrent runs can't return
    # the same notifications. Per service: each AppView has its own read marker.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def list_nsid(self) -> str:
        return f"{self.nsid_prefix}.listNotifications"

    @property
    def unread_nsid(self) -> str:
        return f"{self.nsid_prefix}.getUnreadCount"

    @property
    def update_seen_nsid(self) -> str:
        return f"{self.nsid_prefix}.updateSeen"

    def reset_lock(self) -> None:
        """Recreate the lock (used by tests, which each run their own event loop)."""
        self.lock = asyncio.Lock()


BLUESKY = NotificationService(
    name="Bluesky",
    nsid_prefix="app.bsky.notification",
    proxy=_DEFAULT_PROXIES["app.bsky."],
)


def _reset_lock() -> None:
    """Recreate the Bluesky lock (used by tests)."""
    BLUESKY.reset_lock()


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


async def _list_page(ctx: ToolContext, svc: NotificationService, cursor: str | None) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": _PAGE_SIZE}
    if cursor:
        params["cursor"] = cursor
    data = await _authed_request(
        ctx,
        "GET",
        svc.list_nsid,
        svc.list_nsid,
        params=params,
        headers={"atproto-proxy": svc.proxy},
    )
    if not isinstance(data, dict) or not isinstance(data.get("notifications"), list):
        raise AtprotoError(f"{svc.list_nsid} returned an unexpected response")
    return data


def _check_reasons(reasons: Any) -> set[str]:
    if reasons is None:
        return set(POST_REASONS)
    if (
        not isinstance(reasons, list)
        or not reasons
        or not all(isinstance(r, str) and r for r in reasons)
    ):
        raise AtprotoError(
            "reasons must be a non-empty array of notification reasons, e.g. "
            '["mention", "reply", "quote"]'
        )
    return set(reasons)


async def _scan_unread(
    ctx: ToolContext, svc: NotificationService, reasons: set[str]
) -> tuple[list[dict[str, Any]], datetime | None, bool]:
    """Collect unread notifications with the given reasons, newest first.

    Pages back until the first already-read notification of ANY reason: read
    state is a single "seen up to" time, so everything older is read too.
    Returns (matching unread, the server's seenAt, whether paging stopped at
    the page cap while still seeing unread notifications).
    """
    matching: list[dict[str, Any]] = []
    server_seen_at: datetime | None = None
    cursor: str | None = None
    reached_read = False
    for _ in range(_MAX_PAGES):
        page = await _list_page(ctx, svc, cursor)
        if server_seen_at is None:
            server_seen_at = _parse_time(page.get("seenAt"))
        for notif in page["notifications"]:
            if not isinstance(notif, dict):
                continue
            if notif.get("isRead"):
                reached_read = True
                break
            if notif.get("reason") in reasons:
                matching.append(notif)
        cursor = page.get("cursor")
        if reached_read or not cursor:
            break
    return matching, server_seen_at, not reached_read and bool(cursor)


async def count_unread(
    ctx: ToolContext, svc: NotificationService, reasons: list[str] | None
) -> dict[str, Any]:
    """Count unread notifications with the given reasons; never marks anything read."""
    wanted = _check_reasons(reasons)
    # Cheap check first: nothing unread of any kind means nothing to scan.
    data = await _authed_request(
        ctx,
        "GET",
        svc.unread_nsid,
        svc.unread_nsid,
        headers={"atproto-proxy": svc.proxy},
    )
    total = data.get("count") if isinstance(data, dict) else None
    if not isinstance(total, int):
        raise AtprotoError(f"{svc.unread_nsid} returned an unexpected response")
    if total == 0:
        return {"count": 0}
    matching, _, beyond_scan = await _scan_unread(ctx, svc, wanted)
    result: dict[str, Any] = {"count": len(matching)}
    if beyond_scan:
        result["capped"] = True  # more unread exist past the scan limit
    return result


async def fetch_notifications(
    ctx: ToolContext, svc: NotificationService, limit: Any, reasons: list[str] | None
) -> dict[str, Any]:
    """Return the oldest unread matching notifications and mark read through them.

    Read state is one per-account "seen up to" time, so this returns the
    OLDEST unread matching notifications first and moves the marker only past
    the newest one returned: no matching notification is ever marked read
    without being returned.
    """
    try:
        limit = max(1, min(_MAX_LIMIT, int(limit)))
    except (TypeError, ValueError):
        raise AtprotoError(f"limit must be a number, got {limit!r}")
    wanted = _check_reasons(reasons)

    async with svc.lock:
        unread, server_seen_at, older_unread_beyond_scan = await _scan_unread(ctx, svc, wanted)

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
                svc.update_seen_nsid,
                svc.update_seen_nsid,
                json_body={"seenAt": _format_time(seen_at)},
                headers={"atproto-proxy": svc.proxy},
            )
        except AtprotoError as e:
            result["mark_read_error"] = str(e)
            return result
        result["marked_read"] = True
        result["seen_at"] = _format_time(seen_at)
        return result


REASONS_PARAM_DESC = (
    "Which notification reasons count. Default: posts only, "
    f"{list(POST_REASONS)}. Other reasons include like, repost, follow, "
    "like-via-repost, repost-via-repost, subscribed-post, starterpack-joined, "
    "verified, unverified."
)


def unread_count_description(svc: NotificationService, tool_prefix: str) -> str:
    return (
        f"Return how many unread {svc.name} post notifications (mentions, "
        "replies, quotes) your account has, as {\"count\": n}, without marking "
        "anything read. Likes, follows, reposts, etc. are not counted unless you "
        "pass `reasons`. Read-only, so schedule condition scripts can call it to "
        "wake you only when someone has posted to you: e.g. `const r = await "
        f"tools.{tool_prefix}.unread_count(); output({{ wake: r.count > 0 }});`. "
        f"Use {tool_prefix}.get_notifications to read (and mark read) the "
        "notifications."
    )


def get_notifications_description(svc: NotificationService) -> str:
    return (
        f"Get your {svc.name} account's unread post notifications (mentions, "
        "replies, quotes) and mark them as read. Likes, reposts, follows, etc. "
        "are skipped unless you pass `reasons`; skipped ones older than what's "
        "returned get marked read too. Returns the OLDEST unread notifications "
        "first (up to `limit`, oldest to newest), and marks read exactly up to "
        "the newest one returned, so no matching notification is marked read "
        "without being returned. If `more_unread` is true, newer matching "
        "unread notifications remain: call again to get the next batch. Each "
        "notification has `reason` (mention, reply, quote, ...), `uri`, `cid`, "
        "`author`, `indexed_at`, `record` (the post's content), and "
        "`reason_subject` (the at:// URI of your post it's about, for replies "
        "and quotes). `marked_read` says whether the read marker was updated; "
        "if `mark_read_error` is present, the notifications were returned but "
        "NOT marked read. For read notifications or other filters, use "
        f"atproto.query with {svc.list_nsid} and proxy '{svc.proxy}'."
    )


def reasons_param() -> ToolParameter:
    return ToolParameter(
        name="reasons", type="array", description=REASONS_PARAM_DESC, required=False
    )


def limit_param() -> ToolParameter:
    return ToolParameter(
        name="limit",
        type="number",
        description=f"Maximum notifications to return, 1-{_MAX_LIMIT}.",
        required=False,
        default=50,
    )


@TOOL_REGISTRY.tool(
    name="bluesky.unread_count",
    description=unread_count_description(BLUESKY, "bluesky"),
    parameters=[reasons_param()],
    condition_safe=True,
)
@_tool
async def unread_count(ctx: ToolContext, reasons: list[str] | None = None) -> dict[str, Any]:
    return await count_unread(ctx, BLUESKY, reasons)


@TOOL_REGISTRY.tool(
    name="bluesky.get_notifications",
    description=get_notifications_description(BLUESKY),
    parameters=[limit_param(), reasons_param()],
)
@_tool
async def get_notifications(
    ctx: ToolContext, limit: int = 50, reasons: list[str] | None = None
) -> dict[str, Any]:
    return await fetch_notifications(ctx, BLUESKY, limit, reasons)

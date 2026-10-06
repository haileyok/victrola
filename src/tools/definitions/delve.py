"""delve.town notification tools.

delve.town runs its own AppView with the same notification lexicons as
Bluesky, under the town.delve.* namespace. Its AppView is
did:web:api.delve.town (service #bsky_appview at https://api.delve.town), and
calls reach it through the agent's PDS via the atproto-proxy header, using the
same account and session as the atproto tools.

The read-marking logic is shared with the Bluesky tools (see bluesky.py).
delve.town keeps its own read marker, separate from Bluesky's.
"""

from __future__ import annotations

from typing import Any

from src.tools.definitions.atproto import _tool
from src.tools.definitions.bluesky import (
    NotificationService,
    count_unread,
    fetch_notifications,
    get_notifications_description,
    limit_param,
    reasons_param,
    unread_count_description,
)
from src.tools.registry import TOOL_REGISTRY, ToolContext

DELVE = NotificationService(
    name="delve.town",
    nsid_prefix="town.delve.notification",
    proxy="did:web:api.delve.town#bsky_appview",
)


def _reset_lock() -> None:
    """Recreate the delve.town lock (used by tests)."""
    DELVE.reset_lock()


@TOOL_REGISTRY.tool(
    name="delve.unread_count",
    description=unread_count_description(DELVE, "delve"),
    parameters=[reasons_param()],
    condition_safe=True,
)
@_tool
async def unread_count(ctx: ToolContext, reasons: list[str] | None = None) -> dict[str, Any]:
    return await count_unread(ctx, DELVE, reasons)


@TOOL_REGISTRY.tool(
    name="delve.get_notifications",
    description=get_notifications_description(DELVE),
    parameters=[limit_param(), reasons_param()],
)
@_tool
async def get_notifications(
    ctx: ToolContext, limit: int = 50, reasons: list[str] | None = None
) -> dict[str, Any]:
    return await fetch_notifications(ctx, DELVE, limit, reasons)

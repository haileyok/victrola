"""Schedule condition scripts may call only condition-safe tools.

These run real Deno subprocesses (like tests/test_workspace_permissions.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from src.config import CONFIG
from src.tools.definitions import atproto as ap
from src.tools.definitions import bluesky as bs
from src.tools.executor import ToolExecutor
from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolRegistry


@pytest.fixture
def calls() -> list[str]:
    return []


@pytest.fixture
def executor(tmp_path: Path, monkeypatch, calls) -> ToolExecutor:
    monkeypatch.setattr(CONFIG, "workspace_dir", str(tmp_path / "ws"))
    (tmp_path / "ws").mkdir()
    registry = ToolRegistry()

    @registry.tool(name="probe.count", description="read-only", condition_safe=True)
    async def count(ctx: ToolContext) -> dict[str, Any]:
        calls.append("probe.count")
        return {"count": 3}

    @registry.tool(name="probe.side_effect", description="writes something")
    async def side_effect(ctx: ToolContext) -> str:
        calls.append("probe.side_effect")
        return "did it"

    return ToolExecutor(registry=registry, ctx=ToolContext())


async def test_condition_can_call_condition_safe_tool(executor, calls):
    result = await executor.execute_condition_code(
        "const r = await tools.probe.count();\noutput({ wake: r.count > 0, n: r.count });"
    )
    assert result["success"] is True, result
    assert result["output"] == [{"wake": True, "n": 3}] or result["output"] == {"wake": True, "n": 3}
    assert calls == ["probe.count"]


async def test_condition_stub_omits_unsafe_tools(executor, calls):
    result = await executor.execute_condition_code(
        "output({ hasSafe: typeof tools.probe?.count, hasUnsafe: typeof tools.probe?.side_effect });"
    )
    assert result["success"] is True, result
    out = result["output"][0] if isinstance(result["output"], list) else result["output"]
    assert out == {"hasSafe": "function", "hasUnsafe": "undefined"}
    assert calls == []


async def test_condition_forged_call_to_unsafe_tool_is_blocked(executor, calls):
    """Hand-writing the tool-call protocol message must not reach other tools."""
    forged = json.dumps({"__tool_call__": True, "tool": "probe.side_effect", "params": {}})
    result = await executor.execute_condition_code(
        f"await Deno.stdout.write(new TextEncoder().encode({json.dumps(forged)} + '\\n'));\n"
        "await new Promise((r) => setTimeout(r, 2000));\n"
        "output({ wake: true });"
    )
    assert result["success"] is False
    assert "probe.side_effect" in result["error"]
    assert calls == []


async def test_custom_tool_code_still_has_full_tool_access(executor, calls):
    """The restriction applies only to condition scripts."""
    result = await executor.execute_custom_tool_code(
        "const r = await tools.probe.side_effect();\noutput(r);", params={}
    )
    assert result["success"] is True, result
    assert calls == ["probe.side_effect"]


def test_only_read_only_builtins_are_condition_safe():
    """Guard against accidentally exposing a side-effecting tool to conditions."""
    assert TOOL_REGISTRY.condition_safe_tool_names() == {"bluesky.unread_count", "delve.unread_count"}


# --- bluesky.unread_count ---------------------------------------------------


async def test_unread_count(monkeypatch):
    ap._reset_session_cache()
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("createSession"):
            return httpx.Response(
                200, json={"did": "did:plc:a", "handle": "a.test", "accessJwt": "t", "refreshJwt": "r"}
            )
        if request.url.path == "/xrpc/app.bsky.notification.getUnreadCount":
            return httpx.Response(200, json={"count": 9})
        if request.url.path == "/xrpc/app.bsky.notification.listNotifications":
            # 9 unread: 7 replies/mentions/quotes and 2 likes; then a read one.
            reasons = ["reply", "like", "mention", "quote", "reply", "like", "reply", "quote", "mention"]
            items = [{"reason": r, "isRead": False, "indexedAt": "2026-10-06T00:00:00.000Z"} for r in reasons]
            items.append({"reason": "reply", "isRead": True, "indexedAt": "2026-10-05T00:00:00.000Z"})
            return httpx.Response(200, json={"notifications": items})
        return httpx.Response(404)

    ctx = MagicMock(spec=ToolContext)
    ctx.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    secrets = {"ATPROTO_IDENTIFIER": "a.test", "ATPROTO_APP_PASSWORD": "pw", "ATPROTO_PDS_URL": "https://pds.test"}
    ctx.secret_manager = MagicMock(get_secret=lambda n: secrets.get(n))
    try:
        assert await bs.unread_count(ctx) == {"count": 7}
        req = seen[-1]
        assert req.headers["atproto-proxy"] == "did:web:api.bsky.app#bsky_appview"
        # Never marks anything read.
        assert not any(r.url.path.endswith("updateSeen") for r in seen)
    finally:
        ap._reset_session_cache()

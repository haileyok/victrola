"""Custom tool updates must reset approval when they broaden access.

An approved tool's secrets are injected as env vars and, with requires_net,
can be sent off-box. Adding secrets or network access after approval would
grant access the operator never reviewed, so those changes reset approval.
"""

from __future__ import annotations

import pytest

from src.tools.custom import CustomTool


async def _approved_tool(manager, **overrides) -> CustomTool:
    fields = dict(
        name="t1",
        description="d",
        parameters={},
        code="output('hi')",
        secrets=["A"],
        requires_net=False,
    )
    fields.update(overrides)
    tool = CustomTool(**fields)
    await manager.create_tool(tool)
    await manager.approve_tool("t1")
    assert manager._tools["t1"].approved is True
    return manager._tools["t1"]


@pytest.mark.asyncio
async def test_adding_secret_resets_approval(isolated_custom_tool_manager):
    m = isolated_custom_tool_manager
    tool = await _approved_tool(m)
    msg = await m.update_tool("t1", secrets=["A", "B"])
    assert tool.approved is False
    assert "Approval reset" in msg


@pytest.mark.asyncio
async def test_swapping_secret_resets_approval(isolated_custom_tool_manager):
    m = isolated_custom_tool_manager
    tool = await _approved_tool(m)
    await m.update_tool("t1", secrets=["B"])
    assert tool.approved is False


@pytest.mark.asyncio
async def test_removing_secret_keeps_approval(isolated_custom_tool_manager):
    m = isolated_custom_tool_manager
    tool = await _approved_tool(m, secrets=["A", "B"])
    await m.update_tool("t1", secrets=["A"])
    assert tool.approved is True


@pytest.mark.asyncio
async def test_enabling_net_resets_approval(isolated_custom_tool_manager):
    m = isolated_custom_tool_manager
    tool = await _approved_tool(m, requires_net=False)
    await m.update_tool("t1", requires_net=True)
    assert tool.approved is False


@pytest.mark.asyncio
async def test_disabling_net_keeps_approval(isolated_custom_tool_manager):
    m = isolated_custom_tool_manager
    tool = await _approved_tool(m, requires_net=True)
    await m.update_tool("t1", requires_net=False)
    assert tool.approved is True


@pytest.mark.asyncio
async def test_description_change_keeps_approval(isolated_custom_tool_manager):
    m = isolated_custom_tool_manager
    tool = await _approved_tool(m)
    await m.update_tool("t1", description="new")
    assert tool.approved is True


@pytest.mark.asyncio
async def test_reset_is_persisted(isolated_custom_tool_manager):
    """The reset must survive a reload, not just live in memory."""
    m = isolated_custom_tool_manager
    await _approved_tool(m)
    await m.update_tool("t1", secrets=["A", "B"])
    m._tools.clear()
    await m.load_tools()
    assert m._tools["t1"].approved is False

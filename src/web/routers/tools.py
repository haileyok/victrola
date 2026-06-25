"""Custom tools CRUD endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.tools.executor import ToolExecutor
from src.web.dependencies import get_executor
from src.web.schemas import ToolDetailResponse, ToolSummary

router = APIRouter()


def _get_manager(executor: ToolExecutor) -> Any:
    mgr = executor.custom_tool_manager
    if mgr is None:
        raise HTTPException(500, "Custom tool manager not initialized")
    return mgr


def _get_sm(executor: ToolExecutor) -> Any:
    sm = executor.secret_manager
    if sm is None:
        raise HTTPException(500, "Secret manager not initialized")
    return sm


@router.get("/tools", response_model=list[ToolSummary])
def list_tools(
    executor: ToolExecutor = Depends(get_executor),
) -> list[ToolSummary]:
    mgr = _get_manager(executor)
    tools = mgr.list_tools()
    return [
        ToolSummary(
            name=t.name,
            description=t.description,
            approved=t.approved,
            requires_net=t.requires_net,
            secrets=list(t.secrets),
        )
        for t in tools
    ]


@router.get("/tools/{name}", response_model=ToolDetailResponse)
def get_tool(
    name: str,
    executor: ToolExecutor = Depends(get_executor),
) -> ToolDetailResponse:
    mgr = _get_manager(executor)
    tool = mgr.get_tool(name)
    if tool is None:
        raise HTTPException(404, f"Tool '{name}' not found")

    sm = _get_sm(executor)
    known = set(sm.list_secret_names())
    secret_statuses = [
        {"name": s, "status": "set" if s in known else "missing"}
        for s in tool.secrets
    ]
    return ToolDetailResponse(
        name=tool.name,
        description=tool.description,
        approved=tool.approved,
        requires_net=tool.requires_net,
        code=tool.code,
        parameters=tool.parameters,
        secrets=secret_statuses,
    )


@router.post("/tools/{name}/approve")
async def approve_tool(
    name: str,
    executor: ToolExecutor = Depends(get_executor),
) -> dict[str, Any]:
    mgr = _get_manager(executor)
    sm = _get_sm(executor)

    tool = mgr.get_tool(name)
    if tool is None:
        raise HTTPException(404, f"Tool '{name}' not found")

    known = set(sm.list_secret_names())
    missing = [s for s in tool.secrets if s not in known]
    if missing:
        raise HTTPException(
            400,
            detail={
                "message": "Tool has missing secrets",
                "missing_secrets": missing,
            },
        )
    result = await mgr.approve_tool(name)
    return {"message": result}


@router.post("/tools/{name}/revoke")
async def revoke_tool(
    name: str,
    executor: ToolExecutor = Depends(get_executor),
) -> dict[str, Any]:
    mgr = _get_manager(executor)
    tool = mgr.get_tool(name)
    if tool is None:
        raise HTTPException(404, f"Tool '{name}' not found")
    result = await mgr.revoke_tool(name)
    return {"message": result}


@router.delete("/tools/{name}", status_code=204)
async def delete_tool(
    name: str,
    executor: ToolExecutor = Depends(get_executor),
) -> None:
    mgr = _get_manager(executor)
    tool = mgr.get_tool(name)
    if tool is None:
        raise HTTPException(404, f"Tool '{name}' not found")
    await mgr.delete_tool(name)

"""MCP server CRUD + tool approval endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.tools.executor import ToolExecutor
from src.tools.mcp import MCPServerConfig, MCPTool
from src.web.dependencies import get_executor
from src.web.schemas import (
    CreateMCPServerRequest,
    MCPServerDetail,
    MCPServerSummary,
    MCPToolSummary,
    OAuthCallbackRequest,
    ToolActionRequest,
)

router = APIRouter(prefix="/mcp")


def _get_manager(executor: ToolExecutor) -> Any:
    mgr = executor.mcp_manager
    if mgr is None:
        raise HTTPException(500, "MCP manager not initialized")
    return mgr


def _get_sm(executor: ToolExecutor) -> Any:
    sm = executor.secret_manager
    if sm is None:
        raise HTTPException(500, "Secret manager not initialized")
    return sm


def _server_to_summary(config: MCPServerConfig, manager: Any) -> MCPServerSummary:
    return MCPServerSummary(
        name=config.name,
        transport=config.transport,
        enabled=config.enabled,
        connected=manager.is_connected(config.name),
        tools_total=len(config.tools),
        tools_approved=sum(1 for t in config.tools if t.approved),
    )


def _server_to_detail(config: MCPServerConfig, manager: Any, sm: Any) -> MCPServerDetail:
    known = set(sm.list_secret_names())

    # resolve auth token status
    if config.auth_token_secret:
        auth_status = "set" if config.auth_token_secret in known else "missing"
    else:
        auth_status = "none"

    # resolve env secrets status
    env_statuses = [
        {"name": s, "status": "set" if s in known else "missing"}
        for s in config.env_secrets
    ]

    return MCPServerDetail(
        name=config.name,
        transport=config.transport,
        url=config.url,
        command=config.command,
        args=config.args,
        auth_type=config.auth_type,
        auth_token_secret=config.auth_token_secret,
        auth_token_status=auth_status,
        oauth_status="not_configured",  # set below for async callers
        env_secrets=env_statuses,
        enabled=config.enabled,
        connected=manager.is_connected(config.name),
        tools=[
            MCPToolSummary(
                name=t.name,
                description=t.description,
                approved=t.approved,
            )
            for t in config.tools
        ],
    )


async def _server_to_detail_async(config: MCPServerConfig, manager: Any, sm: Any) -> MCPServerDetail:
    """Async version that resolves OAuth status."""
    detail = _server_to_detail(config, manager, sm)
    detail.oauth_status = await manager.get_oauth_status_async(config.name)
    return detail


@router.get("/servers", response_model=list[MCPServerSummary])
def list_servers(
    executor: ToolExecutor = Depends(get_executor),
) -> list[MCPServerSummary]:
    mgr = _get_manager(executor)
    return [_server_to_summary(c, mgr) for c in mgr.list_servers()]


@router.post("/servers", response_model=MCPServerSummary, status_code=201)
async def create_server(
    body: CreateMCPServerRequest,
    executor: ToolExecutor = Depends(get_executor),
) -> MCPServerSummary:
    mgr = _get_manager(executor)
    config = MCPServerConfig(
        name=body.name,
        transport=body.transport,
        url=body.url,
        command=body.command,
        args=body.args,
        auth_type=body.auth_type,
        auth_token_secret=body.auth_token_secret,
        env_secrets=body.env_secrets,
        enabled=body.enabled,
    )
    result = await mgr.create_server(config)
    if result.startswith("Error:"):
        raise HTTPException(400, result)
    return _server_to_summary(config, mgr)


@router.get("/servers/{name}", response_model=MCPServerDetail)
async def get_server(
    name: str,
    executor: ToolExecutor = Depends(get_executor),
) -> MCPServerDetail:
    mgr = _get_manager(executor)
    sm = _get_sm(executor)
    config = mgr.get_server(name)
    if config is None:
        raise HTTPException(404, f"MCP server '{name}' not found")
    return await _server_to_detail_async(config, mgr, sm)


@router.delete("/servers/{name}", status_code=204)
async def delete_server(
    name: str,
    executor: ToolExecutor = Depends(get_executor),
) -> None:
    mgr = _get_manager(executor)
    config = mgr.get_server(name)
    if config is None:
        raise HTTPException(404, f"MCP server '{name}' not found")
    await mgr.delete_server(name)


@router.post("/servers/{name}/connect", response_model=MCPServerDetail)
async def connect_server(
    name: str,
    executor: ToolExecutor = Depends(get_executor),
) -> MCPServerDetail:
    mgr = _get_manager(executor)
    sm = _get_sm(executor)
    config = mgr.get_server(name)
    if config is None:
        raise HTTPException(404, f"MCP server '{name}' not found")

    # For OAuth servers, run connect in the background so the request doesn't block
    # while waiting for the operator to complete the consent flow
    if config.auth_type == "oauth":
        import asyncio
        asyncio.create_task(mgr.connect_server(name))
        # Return immediately — the frontend will poll for the consent URL
        config = mgr.get_server(name)
        return await _server_to_detail_async(config, mgr, sm)

    try:
        await mgr.connect_server(name)
    except Exception as e:
        raise HTTPException(500, f"Failed to connect: {e}")
    config = mgr.get_server(name)
    return await _server_to_detail_async(config, mgr, sm)


@router.post("/servers/{name}/disconnect", response_model=MCPServerDetail)
async def disconnect_server(
    name: str,
    executor: ToolExecutor = Depends(get_executor),
) -> MCPServerDetail:
    mgr = _get_manager(executor)
    sm = _get_sm(executor)
    config = mgr.get_server(name)
    if config is None:
        raise HTTPException(404, f"MCP server '{name}' not found")
    await mgr.disconnect_server(name)
    config = mgr.get_server(name)
    return await _server_to_detail_async(config, mgr, sm)


@router.post("/servers/{name}/refresh", response_model=MCPServerDetail)
async def refresh_server(
    name: str,
    executor: ToolExecutor = Depends(get_executor),
) -> MCPServerDetail:
    mgr = _get_manager(executor)
    sm = _get_sm(executor)
    config = mgr.get_server(name)
    if config is None:
        raise HTTPException(404, f"MCP server '{name}' not found")
    if not mgr.is_connected(name):
        raise HTTPException(400, f"MCP server '{name}' is not connected")
    try:
        await mgr.discover_tools(name)
    except Exception as e:
        raise HTTPException(500, f"Failed to refresh tools: {e}")
    config = mgr.get_server(name)
    return await _server_to_detail_async(config, mgr, sm)


@router.post("/servers/{name}/tools/approve")
async def approve_tool(
    name: str,
    body: ToolActionRequest,
    executor: ToolExecutor = Depends(get_executor),
) -> dict[str, Any]:
    mgr = _get_manager(executor)
    config = mgr.get_server(name)
    if config is None:
        raise HTTPException(404, f"MCP server '{name}' not found")
    result = await mgr.approve_tool(name, body.tool_name)
    if "not found" in result:
        raise HTTPException(404, result)
    return {"message": result}


@router.post("/servers/{name}/tools/revoke")
async def revoke_tool(
    name: str,
    body: ToolActionRequest,
    executor: ToolExecutor = Depends(get_executor),
) -> dict[str, Any]:
    mgr = _get_manager(executor)
    config = mgr.get_server(name)
    if config is None:
        raise HTTPException(404, f"MCP server '{name}' not found")
    result = await mgr.revoke_tool(name, body.tool_name)
    if "not found" in result:
        raise HTTPException(404, result)
    return {"message": result}


@router.post("/servers/{name}/oauth/deauthorize")
async def oauth_deauthorize(
    name: str,
    executor: ToolExecutor = Depends(get_executor),
) -> dict[str, Any]:
    """Clear stored OAuth tokens so the server can be re-authorized."""
    mgr = _get_manager(executor)
    config = mgr.get_server(name)
    if config is None:
        raise HTTPException(404, f"MCP server '{name}' not found")
    if config.auth_type != "oauth":
        raise HTTPException(400, f"MCP server '{name}' does not use OAuth")
    result = await mgr.clear_oauth_tokens(name)
    return {"message": result}


@router.get("/servers/{name}/oauth/consent-url")
async def oauth_consent_url(
    name: str,
    executor: ToolExecutor = Depends(get_executor),
) -> dict[str, Any]:
    """Get the OAuth consent URL for a server (if a connect is in progress)."""
    mgr = _get_manager(executor)
    config = mgr.get_server(name)
    if config is None:
        raise HTTPException(404, f"MCP server '{name}' not found")
    if config.auth_type != "oauth":
        raise HTTPException(400, f"MCP server '{name}' does not use OAuth")
    url = mgr.get_oauth_consent_url(name)
    has_pending = hasattr(mgr, "_oauth_state") and name in getattr(mgr, "_oauth_state", {}) and mgr._oauth_state[name].get("pending_callback") is not None
    return {"consent_url": url, "pending_callback": has_pending}


@router.post("/servers/{name}/oauth/callback")
async def oauth_callback(
    name: str,
    body: OAuthCallbackRequest,
    executor: ToolExecutor = Depends(get_executor),
) -> dict[str, Any]:
    """Submit the OAuth redirect URL pasted by the operator."""
    mgr = _get_manager(executor)
    config = mgr.get_server(name)
    if config is None:
        raise HTTPException(404, f"MCP server '{name}' not found")
    if config.auth_type != "oauth":
        raise HTTPException(400, f"MCP server '{name}' does not use OAuth")
    result = await mgr.submit_oauth_callback(name, body.redirect_url)
    if "No pending" in result:
        raise HTTPException(400, result)
    return {"message": result}

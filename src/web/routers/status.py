"""Dashboard status endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.agent.agent import Agent
from src.tools.executor import ToolExecutor
from src.web.dependencies import get_agent, get_executor
from src.web.schemas import StatusResponse

router = APIRouter()


@router.get("/status", response_model=StatusResponse)
def get_status(
    agent: Agent = Depends(get_agent),
    executor: ToolExecutor = Depends(get_executor),
) -> StatusResponse:
    from src.config import CONFIG

    # model name — try to get from the client, fall back to config
    model_name = getattr(
        getattr(agent, "client", None), "_model_name", CONFIG.model_name
    )

    # scheduler task count
    scheduler = executor.scheduler
    task_count = len(scheduler.list_tasks()) if scheduler else 0

    # schedules pending approval (have condition_code but not approved)
    pending_count = 0
    if scheduler:
        pending_count = sum(
            1
            for t in scheduler.list_tasks()
            if t.condition_code is not None and not t.approved
        )

    # secret count
    sm = executor.secret_manager
    secret_count = len(sm.list_secret_names()) if sm else 0

    # custom tools: pending / approved counts
    approved = 0
    pending = 0
    mgr = executor.custom_tool_manager
    if mgr:
        all_tools = mgr.list_tools()
        approved = sum(1 for t in all_tools if t.approved)
        pending = len(all_tools) - approved

    # discord on/off
    discord_on = bool(sm and sm.get_secret("DISCORD_BOT_TOKEN"))

    return StatusResponse(
        model=model_name,
        discord=discord_on,
        schedules=task_count,
        schedules_pending=pending_count,
        secrets=secret_count,
        custom_tools_approved=approved,
        custom_tools_pending=pending,
    )

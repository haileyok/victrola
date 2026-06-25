"""Schedules CRUD endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.tools.executor import ToolExecutor
from src.web.dependencies import get_executor
from src.web.schemas import CreateScheduleRequest, ScheduleResponse

router = APIRouter()


def _get_scheduler(executor: ToolExecutor) -> Any:
    scheduler = executor.scheduler
    if scheduler is None:
        raise HTTPException(500, "Scheduler not initialized")
    return scheduler


def _to_response(task: Any) -> ScheduleResponse:
    next_run = None
    if task.enabled and task.last_run:
        try:
            last = datetime.fromisoformat(task.last_run)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            next_run = task.config.next_run(last).isoformat()
        except Exception:
            pass
    return ScheduleResponse(
        name=task.name,
        schedule=task.schedule,
        prompt=task.prompt,
        enabled=task.enabled,
        last_run=task.last_run,
        next_run=next_run,
        condition_code=task.condition_code,
        requires_net=task.requires_net,
        secrets=list(task.secrets) if task.secrets else [],
        approved=task.approved,
    )


@router.get("/schedules", response_model=list[ScheduleResponse])
def list_schedules(
    executor: ToolExecutor = Depends(get_executor),
) -> list[ScheduleResponse]:
    scheduler = _get_scheduler(executor)
    tasks = scheduler.list_tasks()
    return [_to_response(t) for t in tasks]


@router.post("/schedules", response_model=ScheduleResponse, status_code=201)
async def create_schedule(
    body: CreateScheduleRequest,
    executor: ToolExecutor = Depends(get_executor),
) -> ScheduleResponse:
    scheduler = _get_scheduler(executor)

    # validate schedule expression before creating
    from src.scheduler.schedule import parse_schedule

    try:
        parse_schedule(body.schedule)
    except ValueError as e:
        raise HTTPException(400, f"Invalid schedule expression: {e}")

    from src.scheduler.scheduler import ScheduledTask

    if scheduler.get_task(body.name) is not None:
        raise HTTPException(409, f"Schedule '{body.name}' already exists")

    # Normalize empty/whitespace condition_code to None
    condition_code = body.condition_code
    if condition_code is not None and not condition_code.strip():
        condition_code = None

    task = ScheduledTask(
        name=body.name,
        schedule=body.schedule,
        prompt=body.prompt,
        enabled=True,
        condition_code=condition_code,
        requires_net=body.requires_net,
        secrets=list(body.secrets) if body.secrets else [],
        approved=False,
    )
    try:
        result = await scheduler.create_task(task)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    if result.startswith("Error:"):
        if "already exists" in result:
            raise HTTPException(409, result)
        raise HTTPException(400, result)
    return _to_response(task)


@router.post("/schedules/{name}/enable", response_model=ScheduleResponse)
async def enable_schedule(
    name: str,
    executor: ToolExecutor = Depends(get_executor),
) -> ScheduleResponse:
    scheduler = _get_scheduler(executor)
    task = scheduler.get_task(name)
    if task is None:
        raise HTTPException(404, f"Schedule '{name}' not found")
    await scheduler.enable_task(name)
    return _to_response(scheduler.get_task(name))


@router.post("/schedules/{name}/disable", response_model=ScheduleResponse)
async def disable_schedule(
    name: str,
    executor: ToolExecutor = Depends(get_executor),
) -> ScheduleResponse:
    scheduler = _get_scheduler(executor)
    task = scheduler.get_task(name)
    if task is None:
        raise HTTPException(404, f"Schedule '{name}' not found")
    await scheduler.disable_task(name)
    return _to_response(scheduler.get_task(name))


@router.post("/schedules/{name}/approve", response_model=ScheduleResponse)
async def approve_schedule(
    name: str,
    executor: ToolExecutor = Depends(get_executor),
) -> ScheduleResponse:
    scheduler = _get_scheduler(executor)
    task = scheduler.get_task(name)
    if task is None:
        raise HTTPException(404, f"Schedule '{name}' not found")

    if task.condition_code is None:
        raise HTTPException(400, "This schedule has no condition code to approve")

    # Check secrets are available (same pattern as tools router)
    sm = executor.secret_manager
    if sm:
        known = set(sm.list_secret_names())
        missing = [s for s in task.secrets if s not in known]
        if missing:
            raise HTTPException(
                400,
                detail={
                    "message": "Schedule has missing secrets",
                    "missing_secrets": missing,
                },
            )

    result = await scheduler.approve_task(name)
    return _to_response(scheduler.get_task(name))


@router.post("/schedules/{name}/revoke", response_model=ScheduleResponse)
async def revoke_schedule(
    name: str,
    executor: ToolExecutor = Depends(get_executor),
) -> ScheduleResponse:
    scheduler = _get_scheduler(executor)
    task = scheduler.get_task(name)
    if task is None:
        raise HTTPException(404, f"Schedule '{name}' not found")
    await scheduler.revoke_task(name)
    return _to_response(scheduler.get_task(name))


@router.post("/schedules/{name}/test")
async def test_schedule(
    name: str,
    executor: ToolExecutor = Depends(get_executor),
) -> dict[str, Any]:
    scheduler = _get_scheduler(executor)
    task = scheduler.get_task(name)
    if task is None:
        raise HTTPException(404, f"Schedule '{name}' not found")
    if not task.condition_code:
        raise HTTPException(400, "This schedule has no condition code to test")

    # Build env from secrets
    env: dict[str, str] = {}
    sm = executor.secret_manager
    if sm:
        for secret_name in task.secrets:
            val = sm.get_secret(secret_name)
            if val:
                env[secret_name.upper()] = val

    result = await executor.execute_condition_code(
        code=task.condition_code,
        env=env,
        allow_net=task.requires_net,
    )
    return result


@router.delete("/schedules/{name}", status_code=204)
async def delete_schedule(
    name: str,
    executor: ToolExecutor = Depends(get_executor),
) -> None:
    scheduler = _get_scheduler(executor)
    task = scheduler.get_task(name)
    if task is None:
        raise HTTPException(404, f"Schedule '{name}' not found")
    await scheduler.delete_task(name)

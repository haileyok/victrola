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

    task = ScheduledTask(
        name=body.name,
        schedule=body.schedule,
        prompt=body.prompt,
        enabled=True,
    )
    try:
        await scheduler.create_task(task)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
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

"""Tests for trigger approval gating and agent-facing tool integration."""

import pytest
from pathlib import Path

from src.scheduler.scheduler import Scheduler, ScheduledTask


@pytest.fixture
async def store(tmp_path):
    from src.store.store import Store

    s = Store(path=tmp_path / "test.db")
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
async def scheduler(store):
    s = Scheduler(store=store.documents)
    return s


# --- approve_task / revoke_task ---

@pytest.mark.asyncio
async def test_approve_task_sets_approved(scheduler):
    task = ScheduledTask(
        name="t1",
        schedule="1h",
        prompt="hello",
        condition_code="output({ wake: true })",
        approved=False,
    )
    scheduler._tasks["t1"] = task

    result = await scheduler.approve_task("t1")

    assert task.approved is True
    assert "approved" in result.lower()


@pytest.mark.asyncio
async def test_approve_task_no_condition_code(scheduler):
    """Approving a task without condition code returns a message."""
    task = ScheduledTask(name="t1", schedule="1h", prompt="hello")
    scheduler._tasks["t1"] = task

    result = await scheduler.approve_task("t1")

    assert task.approved is False  # unchanged
    assert "no condition code" in result.lower()


@pytest.mark.asyncio
async def test_approve_task_not_found(scheduler):
    result = await scheduler.approve_task("nonexistent")
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_revoke_task_sets_unapproved(scheduler):
    task = ScheduledTask(
        name="t1",
        schedule="1h",
        prompt="hello",
        condition_code="output({ wake: true })",
        approved=True,
    )
    scheduler._tasks["t1"] = task

    result = await scheduler.revoke_task("t1")

    assert task.approved is False
    assert "revoked" in result.lower()


# --- update_task approval reset ---

@pytest.mark.asyncio
async def test_update_condition_code_resets_approval(scheduler):
    """Changing condition_code resets approved to False."""
    task = ScheduledTask(
        name="t1",
        schedule="1h",
        prompt="hello",
        condition_code="output({ wake: false })",
        approved=True,
    )
    scheduler._tasks["t1"] = task

    await scheduler.update_task("t1", condition_code="output({ wake: true })")

    assert task.condition_code == "output({ wake: true })"
    assert task.approved is False  # reset


@pytest.mark.asyncio
async def test_update_requires_net_broadening_resets_approval(scheduler):
    """Changing requires_net from False to True resets approval."""
    task = ScheduledTask(
        name="t1",
        schedule="1h",
        prompt="hello",
        condition_code="output({ wake: true })",
        requires_net=False,
        approved=True,
    )
    scheduler._tasks["t1"] = task

    await scheduler.update_task("t1", requires_net=True)

    assert task.requires_net is True
    assert task.approved is False  # reset (broadened permissions)


@pytest.mark.asyncio
async def test_update_requires_net_narrowing_does_not_reset(scheduler):
    """Changing requires_net from True to False does NOT reset approval."""
    task = ScheduledTask(
        name="t1",
        schedule="1h",
        prompt="hello",
        condition_code="output({ wake: true })",
        requires_net=True,
        approved=True,
    )
    scheduler._tasks["t1"] = task

    await scheduler.update_task("t1", requires_net=False)

    assert task.requires_net is False
    assert task.approved is True  # NOT reset (narrowed permissions)


@pytest.mark.asyncio
async def test_update_secrets_add_resets_approval(scheduler):
    """Adding secrets to an approved trigger resets approval."""
    task = ScheduledTask(
        name="t1",
        schedule="1h",
        prompt="hello",
        condition_code="output({ wake: true })",
        secrets=[],
        approved=True,
    )
    scheduler._tasks["t1"] = task

    await scheduler.update_task("t1", secrets=["API_KEY"])

    assert task.secrets == ["API_KEY"]
    assert task.approved is False  # reset — new secret access


@pytest.mark.asyncio
async def test_update_secrets_remove_does_not_reset(scheduler):
    """Removing secrets (narrowing) does NOT reset approval."""
    task = ScheduledTask(
        name="t1",
        schedule="1h",
        prompt="hello",
        condition_code="output({ wake: true })",
        secrets=["API_KEY"],
        approved=True,
    )
    scheduler._tasks["t1"] = task

    await scheduler.update_task("t1", secrets=[])

    assert task.secrets == []
    assert task.approved is True  # NOT reset (narrowed)


@pytest.mark.asyncio
async def test_create_task_rejects_invalid_name(scheduler):
    """Scheduler.create_task should reject names with path separators."""
    from src.scheduler.scheduler import ScheduledTask

    task = ScheduledTask(name="bad/name", schedule="1h", prompt="hello")
    result = await scheduler.create_task(task)

    assert "invalid" in result.lower()
    assert scheduler.get_task("bad/name") is None


# --- AC.6: agent can create schedule with condition_code ---

@pytest.mark.asyncio
async def test_create_schedule_with_condition_code(store):
    """scheduler.create_schedule tool creates task with approved=False."""
    from src.tools.registry import ToolContext

    sched = Scheduler(store=store.documents)
    ctx = ToolContext()
    ctx._scheduler = sched

    from src.tools.definitions.scheduler import create_schedule

    result = await create_schedule(
        ctx,
        name="my_trigger",
        schedule="30m",
        prompt="check stuff",
        condition_code="output({ wake: true })",
        requires_net=True,
        secrets=["API_KEY"],
    )

    task = sched.get_task("my_trigger")
    assert task is not None
    assert task.condition_code == "output({ wake: true })"
    assert task.approved is False  # pending approval
    assert task.requires_net is True
    assert task.secrets == ["API_KEY"]
    assert "pending operator approval" in result

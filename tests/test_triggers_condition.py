"""Tests for trigger condition evaluation in the scheduler."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.scheduler.scheduler import Scheduler, ScheduledTask


@pytest.fixture
async def store(tmp_path):
    from src.store.store import Store

    s = Store(path=tmp_path / "test.db")
    await s.initialize()
    yield s
    await store_close(s)


async def store_close(s):
    await s.close()


@pytest.fixture
async def scheduler(store):
    s = Scheduler(store=store.documents)
    s._on_fire = AsyncMock(return_value="ok")
    return s


def _make_task(**kwargs):
    """Create a task with a last_run 2 hours ago."""
    now = datetime.now(timezone.utc)
    defaults = dict(
        name="test",
        schedule="1h",
        prompt="hello",
        last_run=(now - timedelta(hours=2)).isoformat(),
    )
    defaults.update(kwargs)
    return ScheduledTask(**defaults)


# --- AC.2: wake:false skips agent, advances last_run ---

@pytest.mark.asyncio
async def test_condition_wake_false_skips_agent(scheduler):
    """When condition returns {wake: false}, agent is NOT woken and last_run advances."""
    now = datetime.now(timezone.utc)
    task = _make_task(condition_code="output({ wake: false })", approved=True)
    original_last_run = task.last_run

    scheduler._condition_runner = AsyncMock(
        return_value={"success": True, "output": {"wake": False}}
    )

    await scheduler._fire(task, now)

    assert scheduler._on_fire.call_count == 0  # agent NOT woken
    assert task.last_run == now.isoformat()  # last_run advanced
    assert task._condition_retry_count == 0


# --- AC.3: wake:true wakes the agent ---

@pytest.mark.asyncio
async def test_condition_wake_true_wakes_agent(scheduler):
    """When condition returns {wake: true}, agent IS woken."""
    now = datetime.now(timezone.utc)
    task = _make_task(condition_code="output({ wake: true })", approved=True)

    scheduler._condition_runner = AsyncMock(
        return_value={"success": True, "output": {"wake": True}}
    )

    await scheduler._fire(task, now)

    assert scheduler._on_fire.call_count == 1  # agent woken
    assert task.last_run == now.isoformat()
    assert task._retry_count == 0
    assert task._condition_retry_count == 0


# --- AC.4: unapproved condition task does not fire ---

@pytest.mark.asyncio
async def test_unapproved_condition_skipped(scheduler):
    """Unapproved condition task does not run condition or wake agent; last_run advanced."""
    now = datetime.now(timezone.utc)
    original_last_run = (now - timedelta(hours=2)).isoformat()
    task = _make_task(
        condition_code="output({ wake: true })",
        approved=False,
        last_run=original_last_run,
    )

    scheduler._condition_runner = AsyncMock(
        return_value={"success": True, "output": {"wake": True}}
    )

    await scheduler._fire(task, now)

    assert scheduler._condition_runner.call_count == 0  # condition NOT evaluated
    assert scheduler._on_fire.call_count == 0  # agent NOT woken
    assert task.last_run == now.isoformat()  # last_run advanced (prevents log spam)


# --- AC.10: condition failure 3-strike auto-disable ---

@pytest.mark.asyncio
async def test_condition_failure_does_not_advance_last_run(scheduler):
    """First condition failure does not advance last_run or wake agent."""
    now = datetime.now(timezone.utc)
    original_last_run = (now - timedelta(hours=2)).isoformat()
    task = _make_task(
        condition_code="throw new Error('boom')",
        approved=True,
        last_run=original_last_run,
    )

    scheduler._condition_runner = AsyncMock(
        return_value={"success": False, "error": "boom"}
    )

    await scheduler._fire(task, now)

    assert scheduler._on_fire.call_count == 0
    assert task.last_run == original_last_run  # NOT advanced
    assert task._condition_retry_count == 1


@pytest.mark.asyncio
async def test_condition_failure_second_strike(scheduler):
    """Second failure still does not advance last_run."""
    now = datetime.now(timezone.utc)
    original_last_run = (now - timedelta(hours=2)).isoformat()
    task = _make_task(
        condition_code="throw new Error('boom')",
        approved=True,
        last_run=original_last_run,
    )
    task._condition_retry_count = 1  # already failed once

    scheduler._condition_runner = AsyncMock(
        return_value={"success": False, "error": "boom"}
    )

    await scheduler._fire(task, now)

    assert task.last_run == original_last_run  # NOT advanced
    assert task._condition_retry_count == 2
    assert task.enabled is True  # still enabled


@pytest.mark.asyncio
async def test_condition_failure_third_strike_auto_disables(scheduler):
    """Third consecutive failure auto-disables the task and advances last_run."""
    now = datetime.now(timezone.utc)
    original_last_run = (now - timedelta(hours=2)).isoformat()
    task = _make_task(
        condition_code="throw new Error('boom')",
        approved=True,
        last_run=original_last_run,
    )
    task._condition_retry_count = 2  # already failed twice

    scheduler._condition_runner = AsyncMock(
        return_value={"success": False, "error": "boom"}
    )

    await scheduler._fire(task, now)

    assert task.enabled is False  # auto-disabled
    assert task.last_run == now.isoformat()  # last_run advanced
    assert task._condition_retry_count == 0  # reset
    assert scheduler._on_fire.call_count == 0  # never woken


# --- Condition runner exception ---

@pytest.mark.asyncio
async def test_condition_runner_exception_treated_as_failure(scheduler):
    """If condition_runner itself throws, it's treated as a failure."""
    now = datetime.now(timezone.utc)
    original_last_run = (now - timedelta(hours=2)).isoformat()
    task = _make_task(
        condition_code="output({ wake: true })",
        approved=True,
        last_run=original_last_run,
    )

    scheduler._condition_runner = AsyncMock(side_effect=RuntimeError("runner crashed"))

    await scheduler._fire(task, now)

    assert scheduler._on_fire.call_count == 0
    assert task.last_run == original_last_run  # NOT advanced
    assert task._condition_retry_count == 1


# --- No condition_runner wired ---

@pytest.mark.asyncio
async def test_no_condition_runner_treated_as_failure(scheduler):
    """If no condition_runner is wired, the task fails."""
    now = datetime.now(timezone.utc)
    original_last_run = (now - timedelta(hours=2)).isoformat()
    task = _make_task(
        condition_code="output({ wake: true })",
        approved=True,
        last_run=original_last_run,
    )

    scheduler._condition_runner = None

    await scheduler._fire(task, now)

    assert scheduler._on_fire.call_count == 0
    assert task.last_run == original_last_run  # NOT advanced
    assert task._condition_retry_count == 1


# --- AC.1: no condition_code fires as before ---

@pytest.mark.asyncio
async def test_no_condition_code_fires_normally(scheduler):
    """Task without condition_code wakes the agent directly."""
    now = datetime.now(timezone.utc)
    task = _make_task()  # no condition_code

    await scheduler._fire(task, now)

    assert scheduler._on_fire.call_count == 1
    assert task.last_run == now.isoformat()

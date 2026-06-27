"""Tests for PR 12: Fix scheduler — advance last_run after callback success."""

import asyncio

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.scheduler.scheduler import Scheduler, ScheduledTask


@pytest.fixture
async def scheduler_with_store(tmp_path):
    """Create a scheduler backed by a real DocumentStore."""
    from src.store.store import Store

    store = Store(path=tmp_path / "test.db")
    await store.initialize()
    s = Scheduler(store=store.documents)
    s._on_fire = AsyncMock(return_value="ok")
    yield s, store
    await store.close()


@pytest.fixture
async def scheduler(scheduler_with_store):
    s, _store = scheduler_with_store
    return s


@pytest.mark.asyncio
async def test_last_run_advanced_on_success(scheduler):
    """last_run should be advanced after callback succeeds."""
    now = datetime.now(timezone.utc)
    task = ScheduledTask(name="test", schedule="interval:1h", prompt="hello")
    task.last_run = (now - timedelta(hours=2)).isoformat()

    await scheduler._fire(task, now)

    assert task.last_run == now.isoformat()
    assert task._retry_count == 0


@pytest.mark.asyncio
async def test_last_run_not_advanced_on_failure(scheduler):
    """last_run should NOT be advanced when callback fails."""
    now = datetime.now(timezone.utc)
    original_last_run = (now - timedelta(hours=2)).isoformat()
    task = ScheduledTask(name="test", schedule="interval:1h", prompt="hello")
    task.last_run = original_last_run

    scheduler._on_fire = AsyncMock(side_effect=RuntimeError("boom"))

    await scheduler._fire(task, now)

    assert task.last_run == original_last_run  # unchanged
    assert task._retry_count == 1


@pytest.mark.asyncio
async def test_last_run_advanced_after_3_failures(scheduler):
    """After 3 consecutive failures, last_run should be advanced."""
    now = datetime.now(timezone.utc)
    original_last_run = (now - timedelta(hours=2)).isoformat()
    task = ScheduledTask(name="test", schedule="interval:1h", prompt="hello")
    task.last_run = original_last_run

    scheduler._on_fire = AsyncMock(side_effect=RuntimeError("boom"))

    # Fire 3 times — each should fail without advancing
    for i in range(3):
        await scheduler._fire(task, now)

    # After 3 failures, last_run should be advanced
    assert task.last_run == now.isoformat()
    assert task._retry_count == 0  # reset after advancing


@pytest.mark.asyncio
async def test_retry_count_resets_on_success(scheduler):
    """Retry count should reset to 0 after a success."""
    now = datetime.now(timezone.utc)
    task = ScheduledTask(name="test", schedule="interval:1h", prompt="hello")
    task.last_run = (now - timedelta(hours=2)).isoformat()
    task._retry_count = 2  # had some failures

    await scheduler._fire(task, now)

    assert task._retry_count == 0
    assert task.last_run == now.isoformat()


def test_is_due_no_write_side_effect():
    """_is_due should not modify task.last_run when it's None."""
    scheduler = Scheduler(store=MagicMock())
    task = ScheduledTask(name="test", schedule="interval:1h", prompt="hello")
    task.last_run = None  # never run

    now = datetime.now(timezone.utc)
    result = scheduler._is_due(task, now)

    # Should return True (due now) without writing to last_run
    assert result is True
    assert task.last_run is None  # unchanged


@pytest.mark.asyncio
async def test_slow_task_does_not_block_others(scheduler_with_store):
    """A slow on_fire callback should not starve other due tasks.

    On serial _tick, the slow task (first in dict order) blocks the fast
    task indefinitely. With concurrent firing + a per-fire timeout, the
    fast task completes within a bounded time.
    """
    s, _store = scheduler_with_store
    s._fire_timeout = 0.5  # short timeout for test

    fast_fired = asyncio.Event()

    async def on_fire(name, prompt):
        if name == "slow":
            await asyncio.sleep(10)  # would block indefinitely without timeout
        elif name == "fast":
            fast_fired.set()
        return "ok"

    s._on_fire = on_fire

    now = datetime.now(timezone.utc)
    task_slow = ScheduledTask(name="slow", schedule="1h", prompt="slow")
    task_slow.last_run = (now - timedelta(hours=2)).isoformat()
    task_fast = ScheduledTask(name="fast", schedule="1h", prompt="fast")
    task_fast.last_run = (now - timedelta(hours=2)).isoformat()

    # slow is first so serial firing would block fast
    s._tasks = {"slow": task_slow, "fast": task_fast}

    await asyncio.wait_for(s._tick(), timeout=3.0)

    assert fast_fired.is_set(), (
        "fast task should have fired despite slow task blocking"
    )

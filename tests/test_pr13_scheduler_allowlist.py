"""Tests for PR 13: Restrict update_task to an allowlist of fields."""

import pytest
from unittest.mock import MagicMock

from src.scheduler.scheduler import Scheduler, ScheduledTask


@pytest.fixture
async def scheduler_with_task(tmp_path):
    """Create a scheduler with a test task backed by a real DocumentStore."""
    from src.store.store import Store

    store = Store(path=tmp_path / "test.db")
    await store.initialize()
    s = Scheduler(store=store.documents)
    task = ScheduledTask(name="test", schedule="interval:1h", prompt="hello")
    s._tasks["test"] = task
    yield s, task
    await store.close()


@pytest.mark.asyncio
async def test_update_task_accepts_allowed_fields(scheduler_with_task):
    """update_task should accept schedule, prompt, enabled, last_run."""
    s, task = scheduler_with_task

    result = await s.update_task("test", prompt="new prompt", enabled=False)

    assert task.prompt == "new prompt"
    assert task.enabled is False
    assert "updated" in result


@pytest.mark.asyncio
async def test_update_task_rejects_unknown_fields(scheduler_with_task):
    """update_task should not modify internal attributes via unknown fields."""
    s, task = scheduler_with_task

    result = await s.update_task("test", _config="malicious")

    assert task._config != "malicious"
    assert "ignored" in result.lower()


@pytest.mark.asyncio
async def test_update_task_skips_none_values(scheduler_with_task):
    """update_task should skip fields with None values."""
    s, task = scheduler_with_task
    original_prompt = task.prompt

    await s.update_task("test", prompt=None)

    assert task.prompt == original_prompt


@pytest.mark.asyncio
async def test_update_task_not_found(tmp_path):
    """update_task should return not found for unknown task."""
    from src.store.store import Store

    store = Store(path=tmp_path / "test.db")
    await store.initialize()
    s = Scheduler(store=store.documents)

    result = await s.update_task("nonexistent", prompt="test")

    assert "not found" in result.lower()
    await store.close()


@pytest.mark.asyncio
async def test_update_task_does_not_clobber_internal_attrs(scheduler_with_task):
    """update_task should not allow clobbering internal attributes."""
    s, task = scheduler_with_task

    await s.update_task(
        "test",
        _config="evil",
        _tasks="evil",
    )

    assert task._config != "evil"
    assert not hasattr(task, "_tasks")

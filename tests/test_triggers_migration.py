"""Tests for schedule storage migration from JSON to DocumentStore."""

import json
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


def _write_legacy(path: Path, tasks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks, indent=2) + "\n")


@pytest.mark.asyncio
async def test_migration_imports_tasks(store, tmp_path):
    """schedules.json entries are migrated to DocumentStore and file renamed."""
    legacy = tmp_path / "schedules.json"
    _write_legacy(legacy, [
        {
            "name": "task1",
            "schedule": "1h",
            "prompt": "do thing 1",
            "enabled": True,
        },
        {
            "name": "task2",
            "schedule": "daily@9:00",
            "prompt": "do thing 2",
            "enabled": False,
        },
    ])

    sched = Scheduler(store=store.documents, legacy_path=legacy)
    await sched.load_tasks()

    # Both tasks loaded
    assert len(sched.list_tasks()) == 2
    t1 = sched.get_task("task1")
    t2 = sched.get_task("task2")
    assert t1 is not None and t2 is not None

    # New fields have defaults
    assert t1.condition_code is None
    assert t1.requires_net is False
    assert t1.secrets == []
    assert t1.approved is False

    # Legacy file renamed
    assert not legacy.exists()
    assert (tmp_path / "schedules.json.migrated").exists()


@pytest.mark.asyncio
async def test_migration_idempotent(store, tmp_path):
    """Re-running with already-migrated file does not error or duplicate."""
    legacy = tmp_path / "schedules.json"
    _write_legacy(legacy, [
        {"name": "task1", "schedule": "1h", "prompt": "p1", "enabled": True},
    ])

    sched = Scheduler(store=store.documents, legacy_path=legacy)
    await sched.load_tasks()
    assert len(sched.list_tasks()) == 1

    # The file is renamed, so a second load should be a no-op
    assert not legacy.exists()
    sched2 = Scheduler(store=store.documents, legacy_path=legacy)
    await sched2.load_tasks()
    assert len(sched2.list_tasks()) == 1  # no duplicate


@pytest.mark.asyncio
async def test_no_legacy_file(store, tmp_path):
    """No legacy file → no error, empty task list."""
    legacy = tmp_path / "schedules.json"  # doesn't exist
    sched = Scheduler(store=store.documents, legacy_path=legacy)
    await sched.load_tasks()
    assert len(sched.list_tasks()) == 0


@pytest.mark.asyncio
async def test_round_trip_persisted_fields(store, tmp_path):
    """Tasks with condition fields round-trip through DocumentStore."""
    sched = Scheduler(store=store.documents, legacy_path=None)

    task = ScheduledTask(
        name="trigger1",
        schedule="30m",
        prompt="check stuff",
        condition_code="output({ wake: true })",
        requires_net=True,
        secrets=["API_KEY"],
        approved=True,
    )
    await sched.create_task(task)

    # Reload from store
    sched2 = Scheduler(store=store.documents, legacy_path=None)
    await sched2.load_tasks()

    loaded = sched2.get_task("trigger1")
    assert loaded is not None
    assert loaded.condition_code == "output({ wake: true })"
    assert loaded.requires_net is True
    assert loaded.secrets == ["API_KEY"]
    assert loaded.approved is True

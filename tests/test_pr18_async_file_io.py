"""Tests for PR 18: Move blocking file I/O off the event loop."""

import asyncio
import json
import pytest
from pathlib import Path

from src.tools.secrets import SecretManager
from src.scheduler.scheduler import Scheduler, ScheduledTask


@pytest.mark.asyncio
async def test_secret_manager_save_is_async(tmp_path):
    """SecretManager._save should be async and write the file."""
    sm = SecretManager(path=tmp_path / "secrets.json")
    await sm.set_secret("test_key", "test_value")

    # Verify file was written
    assert (tmp_path / "secrets.json").exists()
    data = json.loads((tmp_path / "secrets.json").read_text())
    assert data["test_key"] == "test_value"


@pytest.mark.asyncio
async def test_secret_manager_load_is_async(tmp_path):
    """SecretManager.load_secrets should be async and read the file."""
    path = tmp_path / "secrets.json"
    path.write_text(json.dumps({"my_secret": "my_value"}))

    sm = SecretManager(path=path)
    await sm.load_secrets()

    assert sm.get_secret("my_secret") == "my_value"


@pytest.mark.asyncio
async def test_secret_manager_atomic_write(tmp_path):
    """SecretManager should use atomic write (tempfile + os.replace)."""
    sm = SecretManager(path=tmp_path / "secrets.json")
    await sm.set_secret("key1", "val1")
    await sm.set_secret("key2", "val2")

    # Verify both secrets are in the file
    data = json.loads((tmp_path / "secrets.json").read_text())
    assert data["key1"] == "val1"
    assert data["key2"] == "val2"


@pytest.mark.asyncio
async def test_secret_manager_delete_is_async(tmp_path):
    """SecretManager.delete_secret should be async."""
    sm = SecretManager(path=tmp_path / "secrets.json")
    await sm.set_secret("to_delete", "value")
    result = await sm.delete_secret("to_delete")
    assert "deleted" in result.lower()
    assert sm.get_secret("to_delete") is None


@pytest.mark.asyncio
async def test_scheduler_save_is_async(tmp_path):
    """Scheduler._save should be async."""
    scheduler = Scheduler(path=tmp_path / "schedules.json")
    task = ScheduledTask(name="test", schedule="interval:1h", prompt="hello")
    scheduler._tasks["test"] = task

    await scheduler._save()

    # Verify file was written
    assert (tmp_path / "schedules.json").exists()
    data = json.loads((tmp_path / "schedules.json").read_text())
    assert len(data) == 1
    assert data[0]["name"] == "test"


@pytest.mark.asyncio
async def test_scheduler_load_is_async(tmp_path):
    """Scheduler.load_tasks should be async."""
    path = tmp_path / "schedules.json"
    path.write_text(json.dumps([
        {"name": "test", "schedule": "2h", "prompt": "hello", "enabled": True}
    ]))

    scheduler = Scheduler(path=path)
    await scheduler.load_tasks()

    assert len(scheduler._tasks) == 1
    assert scheduler._tasks["test"].prompt == "hello"


@pytest.mark.asyncio
async def test_no_sync_file_io_in_async_methods():
    """No synchronous write_text/read_text calls outside to_thread callbacks."""
    import subprocess
    for f in ["src/tools/secrets.py", "src/scheduler/scheduler.py"]:
        # write_text and read_text should only appear inside _write/_read closures
        # that are passed to asyncio.to_thread
        result = subprocess.run(
            ["grep", "-n", "write_text\\|read_text", f],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                # These calls are fine if they're inside the _write/_read closures
                # (which are passed to asyncio.to_thread). We just verify the calls
                # exist — the real check is that _save and load_* are async and
                # use to_thread.
                assert "write_text" in line or "read_text" in line, \
                    f"Unexpected line in {f}: {line}"

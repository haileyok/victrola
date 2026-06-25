"""Tests for PR 3: Track and cancel TUI background tasks."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


def test_app_stores_task_refs():
    """VictrolaApp should initialize _scheduler_task and _bot_task as None."""
    from src.tui.app import VictrolaApp

    app = VictrolaApp(agent=MagicMock(), executor=MagicMock())
    assert hasattr(app, "_scheduler_task")
    assert hasattr(app, "_bot_task")
    assert app._scheduler_task is None
    assert app._bot_task is None


@pytest.mark.asyncio
async def test_on_unmount_cancels_tasks():
    """on_unmount should cancel and await both background tasks."""
    from src.tui.app import VictrolaApp

    app = VictrolaApp(agent=MagicMock(), executor=MagicMock())

    async def _long_running():
        await asyncio.sleep(100)

    app._scheduler_task = asyncio.create_task(_long_running())
    app._bot_task = asyncio.create_task(_long_running())

    app.agent.aclose = AsyncMock()

    await app.on_unmount()

    assert app._scheduler_task.cancelled()
    assert app._bot_task.cancelled()
    app.agent.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_on_unmount_handles_completed_tasks():
    """on_unmount should handle already-completed tasks gracefully."""
    from src.tui.app import VictrolaApp

    app = VictrolaApp(agent=MagicMock(), executor=MagicMock())

    async def _quick():
        pass

    app._scheduler_task = asyncio.create_task(_quick())
    app._bot_task = asyncio.create_task(_quick())

    await asyncio.sleep(0.01)

    app.agent.aclose = AsyncMock()
    await app.on_unmount()

    app.agent.aclose.assert_called_once()

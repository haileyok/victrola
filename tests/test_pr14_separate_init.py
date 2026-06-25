"""Tests for PR 14: Separate try/except blocks per subsystem in initialize()."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry, ToolContext


@pytest.mark.asyncio
async def test_secrets_failure_does_not_block_scheduler(tmp_path):
    """A failure in SecretManager should not prevent Scheduler from initializing."""
    with patch("src.config.CONFIG") as mock_config:
        mock_config.data_dir = str(tmp_path)

        ctx = ToolContext()
        executor = ToolExecutor(registry=ToolRegistry(), ctx=ctx)

        # Mock SecretManager.load_secrets to raise
        with patch("src.tools.secrets.SecretManager.load_secrets", side_effect=RuntimeError("secrets broken")):
            await executor.initialize()

    # Scheduler should still be initialized
    assert executor._scheduler is not None
    # Secret manager may be set but failed to load
    assert executor._secret_manager is not None


@pytest.mark.asyncio
async def test_scheduler_failure_does_not_block_secrets(tmp_path):
    """A failure in Scheduler.load_tasks should not prevent SecretManager from initializing."""
    with patch("src.config.CONFIG") as mock_config:
        mock_config.data_dir = str(tmp_path)

        ctx = ToolContext()
        executor = ToolExecutor(registry=ToolRegistry(), ctx=ctx)

        # Mock Scheduler.load_tasks to raise
        with patch("src.scheduler.scheduler.Scheduler.load_tasks", side_effect=RuntimeError("scheduler broken")):
            await executor.initialize()

    # Secret manager should still be initialized
    assert executor._secret_manager is not None
    # Scheduler may be set but failed to load
    assert executor._scheduler is not None


@pytest.mark.asyncio
async def test_custom_tool_manager_failure_does_not_block_others(tmp_path):
    """A failure in CustomToolManager should not prevent secrets or scheduler."""
    with patch("src.config.CONFIG") as mock_config:
        mock_config.data_dir = str(tmp_path)

        ctx = ToolContext()
        executor = ToolExecutor(registry=ToolRegistry(), ctx=ctx)

        with patch("src.tools.custom.CustomToolManager.load_tools", side_effect=RuntimeError("custom tools broken")):
            await executor.initialize()

    assert executor._secret_manager is not None
    assert executor._scheduler is not None

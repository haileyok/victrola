"""Tests for PR 17: Fix Deno process output handling."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry, ToolContext


def test_no_deprecated_get_event_loop():
    """No asyncio.get_event_loop() calls should remain in executor.py."""
    import subprocess
    result = subprocess.run(
        ["grep", "-n", "get_event_loop", "src/tools/executor.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, f"Found get_event_loop:\n{result.stdout}"


@pytest.mark.asyncio
async def test_nonzero_exit_code_treated_as_error():
    """A process that exits with code != 0 and empty stdout should be an error."""
    process = MagicMock()
    process.stdin = MagicMock()
    process.stdout = MagicMock()
    process.stderr = MagicMock()
    process.returncode = 1
    process.pid = 12345

    # Simulate: empty stdout on first readline, then process exits
    process.stdout.readline = AsyncMock(return_value=b"")
    process.wait = AsyncMock()
    process.stderr.read = AsyncMock(return_value=b"some error text")

    executor = ToolExecutor(registry=ToolRegistry(), ctx=ToolContext())

    result = await executor._process_deno_output(process)

    assert result["success"] is False
    assert "error" in result
    # Should mention the exit code
    assert "code 1" in result["error"] or "some error text" in result["error"]


@pytest.mark.asyncio
async def test_zero_exit_code_with_empty_stdout_is_clean():
    """A process that exits with code 0 and empty stdout is a clean exit."""
    process = MagicMock()
    process.stdin = MagicMock()
    process.stdout = MagicMock()
    process.stderr = MagicMock()
    process.returncode = 0
    process.pid = 12345

    process.stdout.readline = AsyncMock(return_value=b"")
    process.wait = AsyncMock()
    process.stderr.read = AsyncMock(return_value=b"")

    executor = ToolExecutor(registry=ToolRegistry(), ctx=ToolContext())

    result = await executor._process_deno_output(process)

    assert result["success"] is True
    assert "error" not in result


@pytest.mark.asyncio
async def test_stderr_capped():
    """Stderr output should be capped at MAX_STDERR_SIZE."""
    process = MagicMock()
    process.stdin = MagicMock()
    process.stdout = MagicMock()
    process.stderr = MagicMock()
    process.returncode = 1  # Non-zero so stderr becomes the error
    process.pid = 12345

    process.stdout.readline = AsyncMock(return_value=b"")
    process.wait = AsyncMock()

    # Generate 100KB of stderr across multiple reads
    chunk1 = b"x" * 65536  # exactly MAX_STDERR_SIZE
    chunk2 = b"y" * 40000  # exceeds the cap
    reads = [chunk1, chunk2, b""]
    read_idx = 0
    async def mock_read(size):
        nonlocal read_idx
        if read_idx >= len(reads):
            return b""
        r = reads[read_idx]
        read_idx += 1
        return r
    process.stderr.read = mock_read

    executor = ToolExecutor(registry=ToolRegistry(), ctx=ToolContext())

    result = await executor._process_deno_output(process)

    # The stderr should be truncated
    assert "error" in result
    assert "truncated" in result["error"]


@pytest.mark.asyncio
async def test_stderr_drained_concurrently_no_deadlock():
    """_process_deno_output must drain stderr concurrently.

    Without concurrent stderr draining, a process that fills its stderr
    pipe buffer deadlocks: it can't write more to stderr (pipe full), so
    it can't write to stdout, and the stdout readline loop blocks until
    the read timeout.

    This fake encodes that contract: stdout.readline() blocks until
    stderr.read() has been called. On current code, stderr is only read
    AFTER the stdout loop, so readline blocks for the full read timeout.
    """
    stderr_drained = asyncio.Event()

    class _Stdout:
        async def readline(self):
            await stderr_drained.wait()
            return b""  # EOF

    class _Stderr:
        async def read(self, n=-1):
            stderr_drained.set()
            return b""  # EOF

    process = MagicMock()
    process.stdin = MagicMock()
    process.stdout = _Stdout()
    process.stderr = _Stderr()
    process.returncode = 0
    process.pid = 12345
    process.wait = AsyncMock(return_value=0)

    executor = ToolExecutor(registry=ToolRegistry(), ctx=ToolContext())

    # Should complete quickly (< 5s). On current code this blocks for
    # the full read timeout (~30s) because stderr is never drained.
    await asyncio.wait_for(
        executor._process_deno_output(process),
        timeout=5.0,
    )

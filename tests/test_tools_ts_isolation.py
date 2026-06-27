"""Tests for Bug 01: per-execution tool-stub isolation.

`execute_code` used to write a single shared ``deno/tools.ts`` at the start of
every call and have the generated wrapper ``import ... from "./tools.ts"``.
Because one ``ToolExecutor`` is shared across web, Discord, Signal, and the
scheduler (which fires due tasks via ``asyncio.gather``), concurrent
executions raced on that one file: a call could read a half-written stub
(TypeScript parse error) or a stub written by another call with a different
MCP-tool set (calling a stub that doesn't exist).

The fix gives each execution its own stub file, so no other execution can
mutate the stubs a given run imports.
"""

import asyncio
import re

import pytest

from src.tools.executor import DENO_DIR, ToolExecutor
from src.tools.registry import Tool, ToolContext, ToolRegistry

_IMPORT_RE = re.compile(r'import \* as tools from "\./([^"]+)"')


def _register(registry: ToolRegistry, name: str) -> None:
    async def handler(ctx, **kwargs):
        return None

    registry.register(
        Tool(name=name, description="d", parameters=[], handler=handler)
    )


def _imported_stub(script_path: str) -> tuple[str, str]:
    """Return (stub_filename, stub_contents) imported by a wrapper script."""
    text = (DENO_DIR / script_path).read_text() if not script_path.startswith("/") \
        else open(script_path).read()
    match = _IMPORT_RE.search(text)
    assert match, f"no tools import found in wrapper:\n{text}"
    stub_name = match.group(1)
    stub_text = (DENO_DIR / stub_name).read_text()
    return stub_name, stub_text


@pytest.mark.asyncio
async def test_execute_code_imports_unique_stub_not_shared_file():
    """Each call imports its own stub file, never the shared ``tools.ts``."""
    registry = ToolRegistry()
    _register(registry, "ns.alpha")
    executor = ToolExecutor(registry=registry, ctx=ToolContext())

    seen: list[str] = []

    async def fake_run(script_path: str):
        stub_name, stub_text = _imported_stub(script_path)
        seen.append(stub_name)
        assert "ns.alpha" in stub_text
        return {"success": True, "debug": []}

    executor._run_deno = fake_run  # type: ignore[assignment]

    await executor.execute_code("output(1)")
    await executor.execute_code("output(2)")

    assert seen[0] != "tools.ts", "wrapper imported the shared tools.ts"
    assert seen[0] != seen[1], "two executions shared one stub file"


@pytest.mark.asyncio
async def test_concurrent_executions_see_isolated_stub_snapshots():
    """A run must import the registry snapshot from its own write.

    With a shared stub file, a second execution that registers another tool and
    rewrites the stub between the first run's write and its read would corrupt
    the first run's view. Per-execution stubs prevent that.
    """
    registry = ToolRegistry()
    _register(registry, "ns.alpha")
    executor = ToolExecutor(registry=registry, ctx=ToolContext())

    first_entered = asyncio.Event()
    second_done = asyncio.Event()
    captured: dict[str, str] = {}

    async def fake_run(script_path: str):
        text = open(script_path).read()
        if "SECOND" in text:
            _, stub_text = _imported_stub(script_path)
            captured["second"] = stub_text
            second_done.set()
            return {"success": True, "debug": []}
        # first execution: block until the second has written + read its stub
        first_entered.set()
        await asyncio.wait_for(second_done.wait(), timeout=2.0)
        _, stub_text = _imported_stub(script_path)
        captured["first"] = stub_text
        return {"success": True, "debug": []}

    executor._run_deno = fake_run  # type: ignore[assignment]

    first = asyncio.create_task(executor.execute_code("/*FIRST*/ output(1)"))
    await asyncio.wait_for(first_entered.wait(), timeout=2.0)

    # A racing execution registers a new tool and runs while `first` is mid-flight.
    _register(registry, "ns.beta")
    await executor.execute_code("/*SECOND*/ output(2)")
    await first

    assert "ns.alpha" in captured["first"]
    assert "ns.beta" not in captured["first"], (
        "first execution saw a tool registered by a concurrent execution"
    )
    assert "ns.beta" in captured["second"]

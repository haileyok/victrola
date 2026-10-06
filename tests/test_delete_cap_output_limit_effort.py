"""memory.delete cap, large Deno output lines, memory.delete return value, reasoning_effort."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.agent.agent import OpenAICompatibleClient
from src.agent.llm import SubAgentLLM
from src.config import CONFIG
from src.store.store import Store
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolContext, ToolParameter, ToolRegistry


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(CONFIG, "workspace_dir", str(ws))
    return ws


def _executor(registry: ToolRegistry) -> ToolExecutor:
    return ToolExecutor(registry=registry, ctx=ToolContext())


# --- memory.delete cap (real Deno) -------------------------------------------


async def test_memory_delete_capped_per_execution(workspace):
    deleted: list[int] = []
    reg = ToolRegistry()

    @reg.tool(name="memory.delete", description="d", parameters=[ToolParameter(name="id", type="number", description="id")])
    async def fake_delete(ctx, id: int) -> str:
        deleted.append(int(id))
        return f"Deleted memory entry (id={id})."

    code = """
const results = [];
for (const id of [1, 2, 3, 4, 5]) {
  try { results.push(await tools.memory.delete(id)); }
  catch (e) { results.push("ERR: " + String(e.message ?? e)); }
}
output(results);
"""
    r = await _executor(reg).execute_code(code)
    assert r["success"] is True, r
    out = r["output"] if isinstance(r["output"], list) and isinstance(r["output"][0], str) else r["output"][0]
    assert deleted == [1, 2, 3]
    assert sum(1 for x in out if x.startswith("ERR:")) == 2
    assert "limited to 3 calls per execution" in out[3]


async def test_cap_resets_between_executions(workspace):
    deleted: list[int] = []
    reg = ToolRegistry()

    @reg.tool(name="memory.delete", description="d", parameters=[ToolParameter(name="id", type="number", description="id")])
    async def fake_delete(ctx, id: int) -> str:
        deleted.append(int(id))
        return "ok"

    ex = _executor(reg)
    for _ in range(2):
        await ex.execute_code("for (const id of [1,2,3]) await tools.memory.delete(id); output('done');")
    assert len(deleted) == 6


# --- large output lines (real Deno) ------------------------------------------


async def test_tool_result_and_output_over_64kib(workspace):
    reg = ToolRegistry()
    big = "x" * 300_000

    @reg.tool(name="probe.big", description="returns a large string")
    async def big_result(ctx) -> str:
        return big

    code = "const s = await tools.probe.big(); output({ n: s.length, echo: s });"
    r = await _executor(reg).execute_code(code)
    assert r["success"] is True, r
    out = r["output"] if isinstance(r["output"], dict) else r["output"][0]
    assert out["n"] == 300_000 and len(out["echo"]) == 300_000


async def test_tool_call_args_over_64kib(workspace):
    reg = ToolRegistry()
    got: list[int] = []

    @reg.tool(name="probe.take", description="takes a large string", parameters=[ToolParameter(name="s", type="string", description="s")])
    async def take(ctx, s: str) -> int:
        got.append(len(s))
        return len(s)

    r = await _executor(reg).execute_code("output(await tools.probe.take('y'.repeat(200000)));")
    assert r["success"] is True, r
    assert got == [200_000]


# --- memory.delete returns what it deleted ------------------------------------


async def test_memory_delete_returns_deleted_content(tmp_path: Path):
    from src.tools.definitions.memory import memory_delete

    store = Store(path=tmp_path / "t.db")
    await store.initialize()
    try:
        e = await store.memory.add_entry("factual", "test-scope", "the content to keep an eye on")
        ctx = ToolContext(store=store)
        msg = await memory_delete(ctx, e["id"])
        assert "the content to keep an eye on" in msg and "test-scope" in msg
        assert await store.memory.get_entry(e["id"]) is None
        assert "not found" in await memory_delete(ctx, e["id"])
    finally:
        await store.close()


# --- reasoning_effort ----------------------------------------------------------

CHAT = {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]}


def _recorder(seen: list[dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=CHAT)

    return httpx.MockTransport(handler)


@pytest.mark.parametrize("effort, expected", [("low", "low"), (None, None)])
async def test_main_client_reasoning_effort(effort, expected):
    seen: list[dict[str, Any]] = []
    c = OpenAICompatibleClient(api_key="k", model_name="m", endpoint="https://x/v1", reasoning_effort=effort)
    c._http = httpx.AsyncClient(transport=_recorder(seen))
    await c.complete([{"role": "user", "content": "hi"}], system="s")
    assert seen[0].get("reasoning_effort") == expected
    await c.aclose()


async def test_sub_client_reasoning_effort(monkeypatch):
    seen: list[dict[str, Any]] = []
    real = httpx.AsyncClient
    monkeypatch.setattr("src.agent.llm.httpx.AsyncClient", lambda **kw: real(transport=_recorder(seen), **kw))
    await SubAgentLLM(api="openapi", model="m", api_key="k", endpoint="https://x/v1", reasoning_effort="low").complete("hi")
    assert seen[0]["reasoning_effort"] == "low"


def test_build_services_reasoning_effort_wiring(monkeypatch):
    import main

    for k, v in dict(
        model_api="openapi", model_endpoint="https://x/v1", model_api_key="k",
        model_auth_header_name="", model_auth_header_value="", model_reasoning_effort="low",
        sub_model_api="openapi", sub_model_endpoint="https://x/v1", sub_model_api_key="",
        sub_model_auth_header_name="", sub_model_auth_header_value="", sub_model_reasoning_effort="",
        exa_api_key="",
    ).items():
        monkeypatch.setattr(CONFIG, k, v)
    executor, agent = main.build_services(None, None, None, None)
    assert agent.client._reasoning_effort == "low"
    assert executor.ctx.llm_client._reasoning_effort == "low"  # reused by the sub-agent

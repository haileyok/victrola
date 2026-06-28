"""Tests for workspace feature: Deno permissions, path injection, and web API."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from src.config import CONFIG
from src.tools.registry import ToolRegistry, ToolContext
from src.tools.executor import ToolExecutor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_workspace(tmp_path: Path, monkeypatch) -> Path:
    """Create a temp workspace directory and patch CONFIG to point at it."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(CONFIG, "workspace_dir", str(workspace))
    return workspace


@pytest.fixture
async def executor_with_workspace(temp_workspace: Path) -> ToolExecutor:
    """A ToolExecutor with a temp workspace and mock context."""
    ctx = ToolContext()
    registry = ToolRegistry()
    executor = ToolExecutor(registry=registry, ctx=ctx)
    executor._tool_definition = None
    yield executor


@pytest.fixture
def workspace_app(temp_workspace: Path):
    """Create a FastAPI TestClient with the workspace router."""
    from src.web.app import create_app

    agent = MagicMock()
    executor = MagicMock()
    conversation_manager = MagicMock()
    app = create_app(agent, executor, conversation_manager)
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# Deno security tests (actual subprocess execution)
# ---------------------------------------------------------------------------


async def test_deno_can_write_file_inside_workspace(executor_with_workspace, temp_workspace):
    """Deno can write a file inside the workspace."""
    result = await executor_with_workspace.execute_code(
        'await Deno.writeTextFile(WORKSPACE + "/test_write.txt", "hello workspace");\noutput({ ok: true });'
    )
    assert result["success"], f"Expected success, got: {result}"
    assert (temp_workspace / "test_write.txt").read_text() == "hello workspace"


async def test_deno_can_read_file_from_workspace(executor_with_workspace, temp_workspace):
    """Deno can read a file from the workspace."""
    (temp_workspace / "read_test.txt").write_text("readable content")
    result = await executor_with_workspace.execute_code(
        'const content = await Deno.readTextFile(WORKSPACE + "/read_test.txt");\noutput({ content });'
    )
    assert result["success"], f"Expected success, got: {result}"
    assert result["output"]["content"] == "readable content"


async def test_deno_can_create_subdirectories(executor_with_workspace, temp_workspace):
    """Deno can create subdirectories inside the workspace."""
    result = await executor_with_workspace.execute_code(
        'await Deno.mkdir(WORKSPACE + "/lib/sub", { recursive: true });\n'
        'await Deno.writeTextFile(WORKSPACE + "/lib/sub/mod.ts", "export const x = 1;");\n'
        'output({ ok: true });'
    )
    assert result["success"], f"Expected success, got: {result}"
    assert (temp_workspace / "lib" / "sub" / "mod.ts").exists()
    assert (temp_workspace / "lib" / "sub" / "mod.ts").read_text() == "export const x = 1;"


async def test_deno_cannot_write_outside_workspace(executor_with_workspace):
    """Deno CANNOT write outside the workspace (e.g. /tmp/)."""
    result = await executor_with_workspace.execute_code(
        'try {\n'
        '  await Deno.writeTextFile("/tmp/victrola_escape_test.txt", "escaped");\n'
        '  output({ escaped: true });\n'
        '} catch (e) {\n'
        '  output({ escaped: false, error: e instanceof Error ? e.message : String(e) });\n'
        '}'
    )
    assert result["success"], f"Code should run, got: {result}"
    output = result["output"]
    assert output["escaped"] is False, "Should not be able to write to /tmp"
    # Deno should report a permission error
    err_lower = output["error"].lower()
    assert "permission" in err_lower or "notcapable" in err_lower or "denied" in err_lower or "write access" in err_lower or "allow-write" in err_lower


async def test_deno_cannot_write_via_path_traversal(executor_with_workspace, temp_workspace):
    """Deno CANNOT write via path traversal (../../escape)."""
    result = await executor_with_workspace.execute_code(
        'try {\n'
        '  await Deno.writeTextFile(WORKSPACE + "/../../escape_test.txt", "escaped");\n'
        '  output({ escaped: true });\n'
        '} catch (e) {\n'
        '  output({ escaped: false, error: e instanceof Error ? e.message : String(e) });\n'
        '}'
    )
    assert result["success"], f"Code should run, got: {result}"
    output = result["output"]
    assert output["escaped"] is False, "Should not be able to write via traversal"
    # The file should not exist outside the workspace
    assert not (temp_workspace.parent.parent / "escape_test.txt").exists()


async def test_deno_cannot_create_symlinks(executor_with_workspace, temp_workspace):
    """Deno CANNOT create symlinks inside the workspace."""
    (temp_workspace / "target.txt").write_text("target")
    result = await executor_with_workspace.execute_code(
        'try {\n'
        '  await Deno.symlink(WORKSPACE + "/target.txt", WORKSPACE + "/link.txt");\n'
        '  output({ created: true });\n'
        '} catch (e) {\n'
        '  output({ created: false, error: e instanceof Error ? e.message : String(e) });\n'
        '}'
    )
    assert result["success"], f"Code should run, got: {result}"
    output = result["output"]
    assert output["created"] is False, "Should not be able to create symlinks"
    err_lower = output["error"].lower()
    assert "permission" in err_lower or "notcapable" in err_lower or "denied" in err_lower or "write access" in err_lower or "allow-write" in err_lower
    # The symlink should not have been created
    assert not (temp_workspace / "link.txt").exists() or not (temp_workspace / "link.txt").is_symlink()


async def test_deno_can_dynamically_import_workspace_module(executor_with_workspace, temp_workspace):
    """Deno can dynamically import a module from the workspace."""
    # Write a module to the workspace
    (temp_workspace / "mathmod.ts").write_text(
        "export function double(n: number): number {\n"
        "  return n * 2;\n"
        "}\n"
    )
    result = await executor_with_workspace.execute_code(
        'const mod = await import(WORKSPACE + "/mathmod.ts");\n'
        'const result = mod.double(21);\n'
        'output({ result });'
    )
    assert result["success"], f"Expected success, got: {result}"
    assert result["output"]["result"] == 42


# ---------------------------------------------------------------------------
# Preamble / WORKSPACE constant tests
# ---------------------------------------------------------------------------


async def test_workspace_constant_in_execute_code(executor_with_workspace, temp_workspace):
    """The WORKSPACE constant is correctly set in the execute_code preamble."""
    result = await executor_with_workspace.execute_code(
        'output({ workspace: WORKSPACE });'
    )
    assert result["success"], f"Expected success, got: {result}"
    expected = str(temp_workspace.resolve())
    assert result["output"]["workspace"] == expected


async def test_workspace_constant_in_custom_tool_code(executor_with_workspace, temp_workspace):
    """The WORKSPACE constant is correctly set in the custom tool preamble."""
    result = await executor_with_workspace.execute_custom_tool_code(
        code='output({ workspace: WORKSPACE, params: params });',
        params={"test": True},
    )
    assert result["success"], f"Expected success, got: {result}"
    expected = str(temp_workspace.resolve())
    assert result["output"]["workspace"] == expected
    assert result["output"]["params"]["test"] is True


async def test_condition_code_has_no_workspace_access(executor_with_workspace, temp_workspace):
    """Condition code does NOT get workspace access — no WORKSPACE constant and writes fail."""
    # The WORKSPACE constant should not be defined — referencing it should be a ReferenceError
    result = await executor_with_workspace.execute_condition_code(
        code='try {\n'
        '  output({ workspace: WORKSPACE });\n'
        '} catch (e) {\n'
        '  output({ error: e instanceof Error ? e.message : String(e) });\n'
        '}'
    )
    assert result["success"] is False or "error" in result.get("output", {}), \
        f"Expected error or failure, got: {result}"


async def test_condition_code_cannot_write_files(executor_with_workspace, temp_workspace):
    """Condition code attempting Deno.writeTextFile fails with a permission error."""
    result = await executor_with_workspace.execute_condition_code(
        code='try {\n'
        '  await Deno.writeTextFile("/tmp/condition_escape.txt", "data");\n'
        '  output({ written: true });\n'
        '} catch (e) {\n'
        '  output({ written: false, error: e instanceof Error ? e.message : String(e) });\n'
        '}'
    )
    # The code runs but the write should fail
    output = result.get("output", {})
    if isinstance(output, dict) and "written" in output:
        assert output["written"] is False, "Condition code should not be able to write files"
    else:
        # If the process errored out entirely, that's also acceptable
        assert not result["success"] or "error" in result, \
            f"Expected write failure, got: {result}"


async def test_bare_execute_code_still_runs(executor_with_workspace):
    """A bare execute_code call (output('hi')) still runs with the new permission setup."""
    result = await executor_with_workspace.execute_code(
        "output('hello world');"
    )
    assert result["success"], f"Expected success, got: {result}"
    assert result["output"] == "hello world"


# ---------------------------------------------------------------------------
# Web API tests
# ---------------------------------------------------------------------------


def test_web_list_workspace_root(workspace_app, temp_workspace):
    """Web API directory listing works at the workspace root."""
    (temp_workspace / "file1.txt").write_text("content1")
    (temp_workspace / "subdir").mkdir()

    resp = workspace_app.get("/api/workspace")
    assert resp.status_code == 200
    data = resp.json()
    assert "entries" in data
    names = [e["name"] for e in data["entries"]]
    assert "file1.txt" in names
    assert "subdir" in names
    assert data["total_size_bytes"] >= 0
    assert data["max_size_bytes"] > 0


def test_web_list_workspace_subdirectory(workspace_app, temp_workspace):
    """Web API can list subdirectories."""
    (temp_workspace / "projects").mkdir()
    (temp_workspace / "projects" / "app.ts").write_text("export const x = 1;")

    resp = workspace_app.get("/api/workspace", params={"path": "projects"})
    assert resp.status_code == 200
    data = resp.json()
    names = [e["name"] for e in data["entries"]]
    assert "app.ts" in names


def test_web_read_workspace_file(workspace_app, temp_workspace):
    """Web API file reading works for text files."""
    (temp_workspace / "readable.ts").write_text("export const value = 42;")

    resp = workspace_app.get("/api/workspace/file", params={"path": "readable.ts"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"] == "export const value = 42;"
    assert data["size"] > 0


def test_web_delete_workspace_file(workspace_app, temp_workspace):
    """Web API file deletion works."""
    (temp_workspace / "deletable.txt").write_text("delete me")

    resp = workspace_app.delete("/api/workspace/file", params={"path": "deletable.txt"})
    assert resp.status_code == 200
    assert resp.json()["deleted"] == "deletable.txt"
    assert not (temp_workspace / "deletable.txt").exists()


def test_web_delete_workspace_directory(workspace_app, temp_workspace):
    """Web API can delete directories recursively."""
    (temp_workspace / "to-remove").mkdir()
    (temp_workspace / "to-remove" / "inner.txt").write_text("inner")

    resp = workspace_app.delete("/api/workspace/file", params={"path": "to-remove"})
    assert resp.status_code == 200
    assert not (temp_workspace / "to-remove").exists()


def test_web_create_workspace_dir(workspace_app, temp_workspace):
    """Web API directory creation works."""
    resp = workspace_app.post("/api/workspace/dir", params={"path": "new-dir/sub"})
    assert resp.status_code == 200
    assert (temp_workspace / "new-dir" / "sub").is_dir()


def test_web_rejects_path_traversal_list(workspace_app):
    """Web API rejects path traversal on list endpoint."""
    resp = workspace_app.get("/api/workspace", params={"path": "../../etc/passwd"})
    assert resp.status_code == 403


def test_web_rejects_path_traversal_read(workspace_app):
    """Web API rejects path traversal on file read endpoint."""
    resp = workspace_app.get("/api/workspace/file", params={"path": "../../etc/passwd"})
    assert resp.status_code == 403


def test_web_rejects_path_traversal_delete(workspace_app):
    """Web API rejects path traversal on delete endpoint."""
    resp = workspace_app.delete("/api/workspace/file", params={"path": "../../etc/passwd"})
    assert resp.status_code == 403


def test_web_rejects_path_traversal_mkdir(workspace_app):
    """Web API rejects path traversal on mkdir endpoint."""
    resp = workspace_app.post("/api/workspace/dir", params={"path": "../../etc/evil"})
    assert resp.status_code == 403


def test_web_rejects_deleting_workspace_root(workspace_app):
    """Web API rejects deleting the workspace root (path='')."""
    resp = workspace_app.delete("/api/workspace/file", params={"path": ""})
    assert resp.status_code == 400


def test_web_404_for_missing_path(workspace_app):
    """Web API returns 404 for a non-existent path."""
    resp = workspace_app.get("/api/workspace", params={"path": "nonexistent"})
    assert resp.status_code == 404


def test_web_csrf_protection_post(workspace_app):
    """Cross-origin POST to /api/workspace/dir is rejected (no valid Origin)."""
    resp = workspace_app.post(
        "/api/workspace/dir",
        params={"path": "csrf-test"},
        headers={"Origin": "https://evil.com"},
    )
    assert resp.status_code == 403


def test_web_csrf_protection_delete(workspace_app):
    """Cross-origin DELETE to /api/workspace/file is rejected."""
    resp = workspace_app.delete(
        "/api/workspace/file",
        params={"path": "csrf-test"},
        headers={"Origin": "https://evil.com"},
    )
    assert resp.status_code == 403


def test_web_same_origin_post_allowed(workspace_app, temp_workspace):
    """Same-origin POST to /api/workspace/dir is allowed."""
    resp = workspace_app.post(
        "/api/workspace/dir",
        params={"path": "allowed-dir"},
        headers={"Origin": "http://localhost:8000"},
    )
    assert resp.status_code == 200
    assert (temp_workspace / "allowed-dir").is_dir()


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_config_has_workspace_fields():
    """Config should have workspace_dir and workspace_max_size_mb fields."""
    assert hasattr(CONFIG, "workspace_dir")
    assert hasattr(CONFIG, "workspace_max_size_mb")
    assert CONFIG.workspace_dir == "data/workspace"
    assert CONFIG.workspace_max_size_mb == 1024


def test_config_rejects_comma_in_workspace_dir():
    """Config validator should reject workspace_dir containing commas."""
    from pydantic import ValidationError
    from src.config import Config

    with pytest.raises(ValidationError):
        Config(workspace_dir="data/workspace,/tmp")


def test_resolve_workspace_rejects_comma(tmp_path, monkeypatch):
    """_resolve_workspace should raise on comma-containing paths (defense-in-depth)."""
    comma_dir = tmp_path / "ws,evil"
    comma_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(CONFIG, "workspace_dir", str(comma_dir))

    from src.tools.executor import _resolve_workspace

    with pytest.raises(ValueError, match="comma"):
        _resolve_workspace()


def test_web_read_large_file_does_not_load_full_content(workspace_app, temp_workspace):
    """Reading a file larger than MAX_READ_SIZE should not load the full file."""
    from src.web.routers.workspace import MAX_READ_SIZE

    # Create a file larger than the read cap
    large_content = "x" * (MAX_READ_SIZE + 4096)
    (temp_workspace / "large.txt").write_text(large_content)

    resp = workspace_app.get("/api/workspace/file", params={"path": "large.txt"})
    assert resp.status_code == 200
    data = resp.json()
    # The returned content should be truncated, not the full file
    assert "truncated" in data["content"]
    assert len(data["content"]) < len(large_content)
    # The reported size should be the real file size
    assert data["size"] == len(large_content)


# ---------------------------------------------------------------------------
# Symlink-escape regression tests
#
# Deno authorizes a read/write THROUGH a symlink based on the link's location,
# not its target, so a pre-existing symlink inside the workspace would turn
# --allow-write=<workspace> into a write-outside primitive. The agent can't
# create symlinks (covered above), but the workspace is persistent, so the
# executor refuses to run while one exists and the web API refuses to follow it.
# ---------------------------------------------------------------------------


async def test_deno_refuses_run_when_workspace_has_symlink(
    executor_with_workspace, temp_workspace, tmp_path
):
    """A pre-existing symlink in the workspace blocks execute_code (would
    otherwise let scoped writes escape the workspace)."""
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("ORIGINAL")
    (temp_workspace / "escape").symlink_to(outside)  # planted, points outside

    result = await executor_with_workspace.execute_code(
        'await Deno.writeTextFile(WORKSPACE + "/escape", "CLOBBERED");\n'
        "output({ ok: true });"
    )
    assert result["success"] is False, f"expected refusal, got: {result}"
    assert "symlink" in result["error"].lower()
    # The outside file must be untouched.
    assert outside.read_text() == "ORIGINAL"


async def test_deno_refuses_run_for_inside_pointing_symlink(
    executor_with_workspace, temp_workspace
):
    """The guard rejects ANY symlink, even one pointing inside the workspace —
    symlinks aren't a supported artifact and are the escape vector."""
    (temp_workspace / "real.txt").write_text("x")
    (temp_workspace / "inside_link").symlink_to(temp_workspace / "real.txt")

    result = await executor_with_workspace.execute_code("output({ ok: true });")
    assert result["success"] is False
    assert "symlink" in result["error"].lower()


async def test_custom_tool_refuses_run_when_workspace_has_symlink(
    executor_with_workspace, temp_workspace, tmp_path
):
    """The custom-tool path enforces the same symlink guard."""
    outside = tmp_path / "ct_outside.txt"
    outside.write_text("ORIG")
    (temp_workspace / "ln").symlink_to(outside)

    result = await executor_with_workspace.execute_custom_tool_code(
        'await Deno.writeTextFile(WORKSPACE + "/ln", "X");\noutput({ ok: true });',
        params={},
    )
    assert result["success"] is False
    assert "symlink" in result["error"].lower()
    assert outside.read_text() == "ORIG"


async def test_condition_code_runs_despite_workspace_symlink(
    executor_with_workspace, temp_workspace, tmp_path
):
    """Condition code gets no workspace access, so the symlink guard does not
    apply and it still runs (the guard is scoped to workspace-granting paths)."""
    (temp_workspace / "ln").symlink_to(tmp_path / "whatever")  # dangling is fine
    result = await executor_with_workspace.execute_condition_code("output({ ok: true });")
    assert result["success"] is True


async def test_workspace_over_quota_refuses_execution(
    executor_with_workspace, temp_workspace, monkeypatch
):
    """Execution is refused when the workspace already exceeds its size quota."""
    monkeypatch.setattr(CONFIG, "workspace_max_size_mb", 1)  # 1 MB cap
    (temp_workspace / "big.bin").write_bytes(b"\0" * (2 * 1024 * 1024))  # 2 MB

    result = await executor_with_workspace.execute_code("output({ ok: true });")
    assert result["success"] is False
    assert "size limit" in result["error"].lower()


def test_web_read_rejects_symlink(workspace_app, temp_workspace, tmp_path):
    """The read endpoint refuses to follow a symlink out of the workspace."""
    outside = tmp_path / "secret.txt"
    outside.write_text("SECRET")
    (temp_workspace / "ln").symlink_to(outside)

    resp = workspace_app.get("/api/workspace/file", params={"path": "ln"})
    assert resp.status_code == 403


def test_web_delete_rejects_symlink(workspace_app, temp_workspace, tmp_path):
    """The delete endpoint refuses a symlink path and leaves the target intact."""
    outside = tmp_path / "secret2.txt"
    outside.write_text("SECRET")
    (temp_workspace / "ln2").symlink_to(outside)

    resp = workspace_app.delete("/api/workspace/file", params={"path": "ln2"})
    assert resp.status_code == 403
    assert outside.read_text() == "SECRET"  # untouched


def test_web_list_skips_symlinks(workspace_app, temp_workspace, tmp_path):
    """Listing skips symlink entries so it can't leak metadata about files
    outside the workspace."""
    outside = tmp_path / "exists.txt"
    outside.write_text("data")
    (temp_workspace / "real.txt").write_text("hi")
    (temp_workspace / "ln3").symlink_to(outside)  # points at an existing outside file

    resp = workspace_app.get("/api/workspace", params={"path": ""})
    assert resp.status_code == 200
    names = [e["name"] for e in resp.json()["entries"]]
    assert "real.txt" in names
    assert "ln3" not in names


# ---------------------------------------------------------------------------
# Hardlink-escape regression tests
#
# A hardlink is a regular file (no symlink, O_NOFOLLOW doesn't help). If its
# inode also has a link OUTSIDE the workspace, writing it modifies / reading it
# discloses out-of-workspace data. The agent cannot create such a link (Deno
# denies it), but a pre-existing one (operator/backup/other process) must be
# rejected — symmetric with the symlink guard.
# ---------------------------------------------------------------------------


async def test_deno_refuses_run_with_external_hardlink(
    executor_with_workspace, temp_workspace, tmp_path
):
    """A hardlink to an outside file blocks execution."""
    outside = tmp_path / "hl_outside.txt"
    outside.write_text("ORIGINAL")
    os.link(outside, temp_workspace / "hl")  # hardlink into the workspace

    result = await executor_with_workspace.execute_code(
        'await Deno.writeTextFile(WORKSPACE + "/hl", "CLOBBERED");\noutput({ ok: true });'
    )
    assert result["success"] is False, f"expected refusal, got: {result}"
    assert "hardlink" in result["error"].lower()
    assert outside.read_text() == "ORIGINAL"


async def test_deno_allows_internal_hardlink(executor_with_workspace, temp_workspace):
    """A hardlink whose links are all INSIDE the workspace is fine — the guard
    must not brick the agent on a self-created internal hardlink."""
    (temp_workspace / "a.txt").write_text("data")
    os.link(temp_workspace / "a.txt", temp_workspace / "b.txt")  # both inside

    result = await executor_with_workspace.execute_code("output({ ok: true });")
    assert result["success"] is True, f"internal hardlink should be allowed: {result}"


async def test_deno_cannot_create_escaping_hardlink(
    executor_with_workspace, temp_workspace, tmp_path
):
    """The agent itself cannot Deno.link an outside file into the workspace
    (Deno denies read on the outside source) — locks in the runtime behavior."""
    src = tmp_path / "link_src.txt"
    src.write_text("x")
    code = (
        "try {\n"
        f"  await Deno.link({json.dumps(str(src))}, WORKSPACE + \"/x\");\n"
        "  output({ created: true });\n"
        "} catch (e) {\n"
        "  output({ created: false, error: e instanceof Error ? e.message : String(e) });\n"
        "}"
    )
    result = await executor_with_workspace.execute_code(code)
    assert result["success"], f"code should run, got: {result}"
    assert result["output"]["created"] is False
    err = result["output"]["error"].lower()
    assert any(k in err for k in ("permission", "notcapable", "denied", "read access", "allow-read"))


def test_web_read_rejects_hardlink(workspace_app, temp_workspace, tmp_path):
    """The read endpoint refuses a multi-linked file (possible hardlink-out)."""
    outside = tmp_path / "hl_secret.txt"
    outside.write_text("SECRET")
    os.link(outside, temp_workspace / "hl")

    resp = workspace_app.get("/api/workspace/file", params={"path": "hl"})
    assert resp.status_code == 404
    assert outside.read_text() == "SECRET"

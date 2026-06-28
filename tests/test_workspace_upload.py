"""Tests for files.upload_workspace_file."""
from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from src.config import CONFIG
from src.tools.definitions import workspace_upload as wu
from src.tools.definitions.workspace_upload import (
    _read_workspace_file,
    _safe_name,
    upload_workspace_file,
)
from src.tools.registry import ToolContext

PDF = b"%PDF-1.4 hello"


@pytest.fixture
def temp_workspace(tmp_path: Path, monkeypatch):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setattr(CONFIG, "workspace_dir", str(ws))
    return ws


class _FakeMCP:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def call_tool(self, server, tool, params):
        self.calls.append((server, tool, params))
        return self.result


def _ctx(mcp):
    ctx = ToolContext()
    ctx._mcp_manager = mcp
    return ctx


# -------------------- safe read --------------------
def test_read_regular_file(temp_workspace):
    (temp_workspace / "a.pdf").write_bytes(PDF)
    data, name = _read_workspace_file("a.pdf")
    assert data == PDF and name == "a.pdf"


@pytest.mark.parametrize("bad", ["", "   ", "../outside", "/etc/passwd", "a/../../x"])
def test_read_rejects_bad_paths(temp_workspace, bad):
    with pytest.raises(ValueError):
        _read_workspace_file(bad)


def test_read_rejects_symlink_component(temp_workspace, tmp_path):
    outside = tmp_path / "secret"
    outside.mkdir()
    (outside / "f.pdf").write_bytes(b"SECRET")
    os.symlink(outside, temp_workspace / "link")
    with pytest.raises(ValueError):
        _read_workspace_file("link/f.pdf")


def test_read_rejects_final_symlink(temp_workspace, tmp_path):
    outside = tmp_path / "s.pdf"
    outside.write_bytes(b"SECRET")
    os.symlink(outside, temp_workspace / "x.pdf")
    with pytest.raises(ValueError):
        _read_workspace_file("x.pdf")


def test_read_rejects_hardlink(temp_workspace, tmp_path):
    outside = tmp_path / "h.pdf"
    outside.write_bytes(b"SECRET")
    os.link(outside, temp_workspace / "x.pdf")
    with pytest.raises(ValueError):
        _read_workspace_file("x.pdf")


def test_read_rejects_directory(temp_workspace):
    (temp_workspace / "d").mkdir()
    with pytest.raises(ValueError):
        _read_workspace_file("d")


def test_read_rejects_fifo_without_hanging(temp_workspace):
    # O_NONBLOCK must keep opening a FIFO from blocking on a writer.
    os.mkfifo(temp_workspace / "pipe")
    with pytest.raises(ValueError):
        _read_workspace_file("pipe")


def test_read_rejects_oversize(temp_workspace, monkeypatch):
    monkeypatch.setattr(wu, "MAX_UPLOAD_BYTES", 8)
    (temp_workspace / "big.bin").write_bytes(b"x" * 64)
    with pytest.raises(ValueError):
        _read_workspace_file("big.bin")


def test_safe_name():
    assert _safe_name("a b.pdf") == "a_b.pdf"
    assert _safe_name("../../etc/passwd") == "passwd"
    assert _safe_name("") == "file"


# -------------------- upload --------------------
async def test_upload_success(temp_workspace):
    (temp_workspace / "a.pdf").write_bytes(PDF)
    mcp = _FakeMCP({"uploaded": True, "filename": "a.pdf"})
    out = await upload_workspace_file(_ctx(mcp), "a.pdf", dest="Note")
    assert out["uploaded"] is True
    assert out["size_bytes"] == len(PDF)
    server, tool, params = mcp.calls[0]
    assert server == "supernote" and tool == "upload_file"
    assert params["filename"] == "a.pdf" and params["dest"] == "Note"
    assert base64.b64decode(params["content_base64"]) == PDF


async def test_upload_passes_through_mcp_error(temp_workspace):
    (temp_workspace / "a.pdf").write_bytes(PDF)
    mcp = _FakeMCP({"error": "folder 'Inbox' not found"})
    out = await upload_workspace_file(_ctx(mcp), "a.pdf", dest="Inbox")
    assert "error" in out and "folder" in out["error"]


async def test_upload_no_mcp_manager(temp_workspace):
    (temp_workspace / "a.pdf").write_bytes(PDF)
    out = await upload_workspace_file(_ctx(None), "a.pdf")
    assert "error" in out and "MCP manager" in out["error"]


async def test_upload_bad_path_returns_error(temp_workspace):
    out = await upload_workspace_file(_ctx(_FakeMCP({})), "../escape.pdf")
    assert "error" in out and "outside the workspace" in out["error"]


async def test_upload_symlink_sends_nothing(temp_workspace, tmp_path):
    outside = tmp_path / "secret.pdf"
    outside.write_bytes(b"SECRET")
    os.symlink(outside, temp_workspace / "x.pdf")
    mcp = _FakeMCP({"uploaded": True})
    out = await upload_workspace_file(_ctx(mcp), "x.pdf")
    assert "error" in out
    assert mcp.calls == []  # outside bytes were never sent


async def test_upload_oversize_sends_nothing(temp_workspace, monkeypatch):
    monkeypatch.setattr(wu, "MAX_UPLOAD_BYTES", 8)
    (temp_workspace / "big.bin").write_bytes(b"x" * 64)
    mcp = _FakeMCP({"uploaded": True})
    out = await upload_workspace_file(_ctx(mcp), "big.bin")
    assert "error" in out
    assert mcp.calls == []

"""Tests for web.save_url_as_pdf (download-or-render a URL to the workspace)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import CONFIG
from src.tools.definitions import web_pdf
from src.tools.definitions.web_pdf import (
    SsrfError,
    _check_public_url,
    _name_from_url,
    _render_pdf,
    _safe_pdf_name,
    save_url_as_pdf,
)
from src.tools.registry import ToolContext

PDF_BYTES = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<<>>\n%%EOF\n"


# --------------------------------------------------------------------------- #
# SSRF policy (network-free: IP literals resolve to themselves)
# --------------------------------------------------------------------------- #
BLOCKED = [
    "http://127.0.0.1/x",                      # loopback
    "http://169.254.169.254/latest",           # link-local / metadata
    "http://10.0.0.5/x",                        # RFC1918
    "http://192.168.1.1/x",                     # RFC1918
    "http://100.104.167.50:19072/x",            # tailnet (CGNAT)
    "http://[::ffff:127.0.0.1]/x",              # IPv4-mapped loopback
    "http://[64:ff9b::a9fe:a9fe]/x",            # NAT64 -> 169.254.169.254
    "http://[2002:7f00:1::]/x",                 # 6to4 -> 127.0.0.1
    "http://[::1]/x",                            # IPv6 loopback
    "http://224.0.0.1/x",                        # multicast
    "http://0.0.0.0/x",                          # unspecified
    "ftp://example.com/x",                       # bad scheme
    "http:///nohost",                            # no host
]
ALLOWED = ["http://1.1.1.1/x", "https://8.8.8.8/x.pdf"]


@pytest.mark.parametrize("url", BLOCKED)
def test_check_public_url_blocks(url):
    with pytest.raises(SsrfError):
        _check_public_url(url)


@pytest.mark.parametrize("url", ALLOWED)
def test_check_public_url_allows(url):
    _check_public_url(url)  # must not raise


def test_name_helpers():
    assert _name_from_url("https://h.example/a/b/report%20v2.html?x=1") == "report v2.html"
    assert _safe_pdf_name("report v2.html") == "report_v2.pdf"
    assert _safe_pdf_name("paper.pdf") == "paper.pdf"
    assert _safe_pdf_name("../../etc/passwd") == "passwd.pdf"
    assert _safe_pdf_name("") == "page.pdf"
    assert _safe_pdf_name("..") == "page.pdf"


# --------------------------------------------------------------------------- #
# Fake httpx client for the download path
# --------------------------------------------------------------------------- #
class _FakeStream:
    def __init__(self, body=b"", content_type="application/octet-stream", location=None):
        self._body = body
        self.headers = {"content-type": content_type}
        self.is_redirect = location is not None
        if location:
            self.headers["location"] = location

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        yield self._body


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)

    def stream(self, method, url, **kwargs):
        return self._responses.pop(0)


def _ctx(responses):
    return ToolContext(http_client=_FakeClient(responses))


@pytest.fixture
def temp_workspace(tmp_path: Path, monkeypatch):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setattr(CONFIG, "workspace_dir", str(ws))
    return ws


async def test_download_pdf_branch(temp_workspace):
    ctx = _ctx([_FakeStream(PDF_BYTES, "application/pdf")])
    out = await save_url_as_pdf(ctx, "http://1.1.1.1/paper.pdf")
    assert out["kind"] == "downloaded"
    assert out["path"] == "paper.pdf"
    assert (temp_workspace / "paper.pdf").read_bytes() == PDF_BYTES


async def test_download_pdf_by_magic_even_with_wrong_content_type(temp_workspace):
    ctx = _ctx([_FakeStream(PDF_BYTES, "application/octet-stream")])
    out = await save_url_as_pdf(ctx, "http://1.1.1.1/paper")
    assert out["kind"] == "downloaded"  # sniffed %PDF- magic


async def test_redirect_to_private_is_refused(temp_workspace):
    ctx = _ctx([_FakeStream(location="http://127.0.0.1/secret.pdf")])
    out = await save_url_as_pdf(ctx, "http://1.1.1.1/redir")
    assert "error" in out
    assert "public address" in out["error"]


async def test_render_branch_mocked(temp_workspace, monkeypatch):
    """Non-PDF content goes through the render path and is saved."""
    async def fake_fetch(ctx, url):
        return b"<html><body>hi</body></html>", "text/html; charset=utf-8", url

    async def fake_render(url):
        return b"%PDF-1.4 rendered"

    monkeypatch.setattr(web_pdf, "_fetch_guarded", fake_fetch)
    monkeypatch.setattr(web_pdf, "_render_pdf", fake_render)

    out = await save_url_as_pdf(_ctx([]), "http://1.1.1.1/article")
    assert out["kind"] == "rendered"
    assert out["path"] == "article.pdf"
    assert (temp_workspace / "article.pdf").read_bytes() == b"%PDF-1.4 rendered"


async def test_write_refuses_symlink_in_workspace(temp_workspace, monkeypatch, tmp_path):
    """A planted symlink at the destination name must not be written through."""
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"ORIGINAL")
    os.symlink(outside, temp_workspace / "x.pdf")  # planted symlink

    ctx = _ctx([_FakeStream(PDF_BYTES, "application/pdf")])
    out = await save_url_as_pdf(ctx, "http://1.1.1.1/x.pdf", filename="x.pdf")
    assert "error" in out
    assert outside.read_bytes() == b"ORIGINAL"  # not clobbered through the symlink


# --------------------------------------------------------------------------- #
# Real render smoke (skips if Playwright/Chromium isn't installed)
# --------------------------------------------------------------------------- #
async def test_render_pdf_real_smoke():
    pytest.importorskip("playwright")
    try:
        pdf = await _render_pdf("data:text/html,<h1>Hello PDF</h1>")
    except web_pdf.RenderUnavailable as e:
        pytest.skip(f"chromium not installed: {e}")
    assert pdf[:5] == b"%PDF-"
    assert len(pdf) > 200

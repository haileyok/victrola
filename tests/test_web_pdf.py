"""Tests for web.save_url_as_pdf (download-or-render a URL to the workspace)."""
from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

from src.config import CONFIG
from src.tools.definitions import web_pdf
from src.tools.definitions.web_pdf import (
    SsrfError,
    _check_public_url,
    _guard_route,
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
    "http://127.0.0.1/x",
    "http://169.254.169.254/latest",
    "http://10.0.0.5/x",
    "http://192.168.1.1/x",
    "http://100.104.167.50:19072/x",        # tailnet (CGNAT)
    "http://[::ffff:127.0.0.1]/x",          # IPv4-mapped loopback
    "http://[64:ff9b::a9fe:a9fe]/x",        # NAT64 -> 169.254.169.254
    "http://[2002:7f00:1::]/x",             # 6to4 -> 127.0.0.1
    "http://[::1]/x",
    "http://224.0.0.1/x",                    # multicast
    "http://0.0.0.0/x",                      # unspecified
    "ftp://example.com/x",                   # bad scheme
    "http:///nohost",                        # no host
]
ALLOWED = ["http://1.1.1.1/x", "https://8.8.8.8/x.pdf"]


@pytest.mark.parametrize("url", BLOCKED)
def test_check_public_url_blocks(url):
    with pytest.raises(SsrfError):
        _check_public_url(url)


@pytest.mark.parametrize("url", ALLOWED)
def test_check_public_url_allows(url):
    _check_public_url(url)


def test_name_helpers():
    assert _name_from_url("https://h.example/a/b/report%20v2.html?x=1") == "report v2.html"
    assert _safe_pdf_name("report v2.html") == "report_v2.pdf"
    assert _safe_pdf_name("paper.pdf") == "paper.pdf"
    assert _safe_pdf_name("../../etc/passwd") == "passwd.pdf"
    assert _safe_pdf_name("") == "page.pdf"
    assert _safe_pdf_name("..") == "page.pdf"


# --------------------------------------------------------------------------- #
# Fake httpx client (records calls so we can assert pinning)
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
        self.calls = []

    def stream(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return self._responses.pop(0)


def _ctx(responses):
    return ToolContext(http_client=_FakeClient(responses))


@pytest.fixture
def temp_workspace(tmp_path: Path, monkeypatch):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setattr(CONFIG, "workspace_dir", str(ws))
    return ws


# --------------------------------------------------------------------------- #
# Download path
# --------------------------------------------------------------------------- #
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


async def test_content_type_pdfx_is_not_treated_as_pdf(temp_workspace, monkeypatch):
    async def fake_render(url):
        return b"%PDF-1.4 rendered"

    monkeypatch.setattr(web_pdf, "_render_pdf", fake_render)
    ctx = _ctx([_FakeStream(b"<html>not a pdf</html>", "application/pdfx")])
    out = await save_url_as_pdf(ctx, "http://1.1.1.1/x")
    assert out["kind"] == "rendered"  # exact media-type match, not substring


async def test_redirect_to_private_is_refused(temp_workspace):
    ctx = _ctx([_FakeStream(location="http://127.0.0.1/secret.pdf")])
    out = await save_url_as_pdf(ctx, "http://1.1.1.1/redir")
    assert "error" in out and "public address" in out["error"]


async def test_download_pins_validated_ip(temp_workspace, monkeypatch):
    """The connection targets the IP we validated (closes DNS rebinding), with
    Host + SNI preserved against the real hostname."""

    def fake_gai(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", port or 80))]

    monkeypatch.setattr(web_pdf.socket, "getaddrinfo", fake_gai)
    client = _FakeClient([_FakeStream(PDF_BYTES, "application/pdf")])
    out = await save_url_as_pdf(ToolContext(http_client=client), "http://example.test/p.pdf")
    assert out["kind"] == "downloaded"
    call = client.calls[0]
    assert "93.184.216.34" in call["url"]            # connected to the validated IP
    assert call["headers"]["Host"] == "example.test"
    assert call["extensions"]["sni_hostname"] == "example.test"


# --------------------------------------------------------------------------- #
# Render branch (mocked) + route guard
# --------------------------------------------------------------------------- #
async def test_render_branch_mocked(temp_workspace, monkeypatch):
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


class _FakeRoute:
    def __init__(self, url):
        self.request = type("R", (), {"url": url})()
        self.action = None

    async def abort(self):
        self.action = "abort"

    async def continue_(self):
        self.action = "continue"


async def test_route_guard_fails_closed_on_private():
    r = _FakeRoute("http://127.0.0.1/x")
    await _guard_route(r)
    assert r.action == "abort"


async def test_route_guard_allows_public():
    r = _FakeRoute("http://1.1.1.1/x")
    await _guard_route(r)
    assert r.action == "continue"


async def test_route_guard_allows_non_http_scheme():
    r = _FakeRoute("data:text/html,<h1>hi</h1>")
    await _guard_route(r)
    assert r.action == "continue"


# --------------------------------------------------------------------------- #
# Workspace write safety (atomic replace never writes through a planted link)
# --------------------------------------------------------------------------- #
async def test_write_does_not_clobber_through_symlink(temp_workspace, tmp_path):
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"ORIGINAL")
    os.symlink(outside, temp_workspace / "x.pdf")  # planted symlink at dest

    ctx = _ctx([_FakeStream(PDF_BYTES, "application/pdf")])
    out = await save_url_as_pdf(ctx, "http://1.1.1.1/x.pdf", filename="x.pdf")
    assert "error" not in out
    assert outside.read_bytes() == b"ORIGINAL"  # not written through the symlink
    dest = temp_workspace / "x.pdf"
    assert not dest.is_symlink() and dest.read_bytes() == PDF_BYTES


async def test_write_does_not_clobber_through_hardlink(temp_workspace, tmp_path):
    outside = tmp_path / "outside_hl.pdf"
    outside.write_bytes(b"ORIGINAL")
    os.link(outside, temp_workspace / "x.pdf")  # planted hardlink at dest

    ctx = _ctx([_FakeStream(PDF_BYTES, "application/pdf")])
    out = await save_url_as_pdf(ctx, "http://1.1.1.1/x.pdf", filename="x.pdf")
    assert "error" not in out
    assert outside.read_bytes() == b"ORIGINAL"  # outside inode untouched
    assert (temp_workspace / "x.pdf").read_bytes() == PDF_BYTES


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

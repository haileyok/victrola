"""Save a web URL as a PDF in the agent workspace.

Host-side (Python) tool: the Deno sandbox has no network, so fetching/rendering
happens here and the result lands in the workspace for the agent to use.

- If the URL is already a PDF, it's downloaded as-is.
- Otherwise the page is rendered to a PDF with a headless Chromium (Playwright).

SSRF: only public (globally-routable) addresses are allowed. The entry URL and
every redirect hop are validated, and during rendering each http(s) sub-resource
request is checked too. Residual: DNS can change between validation and the
actual connection (rebinding), and the browser may follow JS/meta redirects we
don't see — so this is a strong guard, not an airtight network jail.
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import re
import socket
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit

import httpx

from src.config import CONFIG
from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter

logger = logging.getLogger(__name__)

MAX_BYTES = 100 * 1024 * 1024  # 100 MB cap on download / fetched body
MAX_REDIRECTS = 5
RENDER_TIMEOUT_MS = 30_000

# NAT64 translation prefixes (RFC 6052/8215); embedded IPv4 is in the low 32 bits.
_NAT64_PREFIXES = (
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/96"),
)


class SsrfError(Exception):
    """The URL is not allowed by the public-address policy."""


class RenderUnavailable(Exception):
    """Playwright / Chromium is not available to render a page."""


# --------------------------------------------------------------------------- #
# SSRF: public-address validation
# --------------------------------------------------------------------------- #
def _normalize_ip(addr: str) -> ipaddress._BaseAddress:
    """Collapse IPv6 forms that embed an IPv4 address (mapped / 6to4 / NAT64) to
    that IPv4 so the public check can't be bypassed via the alternate form."""
    ip = ipaddress.ip_address(addr)
    if ip.version == 6:
        if ip.ipv4_mapped is not None:
            return ip.ipv4_mapped
        if ip.sixtofour is not None:
            return ip.sixtofour
        for prefix in _NAT64_PREFIXES:
            if ip in prefix:
                return ipaddress.ip_address(int(ip) & 0xFFFFFFFF)
    return ip


def _is_public(ip: ipaddress._BaseAddress) -> bool:
    """is_global already excludes loopback/private/link-local/CGNAT(tailnet)/
    unspecified, but returns True for multicast and some reserved space."""
    return bool(ip.is_global) and not ip.is_multicast and not ip.is_reserved


def _check_public_url(url: str) -> None:
    """Raise SsrfError unless url is http(s) and resolves only to public IPs."""
    sp = urlsplit(url)
    if sp.scheme not in ("http", "https"):
        raise SsrfError("only http and https URLs are allowed")
    host = sp.hostname
    if not host:
        raise SsrfError("URL has no host")
    try:
        port = sp.port
    except ValueError:
        raise SsrfError("URL has an invalid port")
    default_port = 443 if sp.scheme == "https" else 80
    try:
        infos = socket.getaddrinfo(host, port or default_port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise SsrfError(f"cannot resolve host {host!r}: {exc}")
    if not infos:
        raise SsrfError(f"no addresses for host {host!r}")
    for info in infos:
        ip = _normalize_ip(info[4][0])
        if not _is_public(ip):
            raise SsrfError(f"{host!r} -> {ip} is not a public address")


# --------------------------------------------------------------------------- #
# Fetch (download path) with per-hop revalidation
# --------------------------------------------------------------------------- #
async def _fetch_guarded(ctx: ToolContext, url: str) -> tuple[bytes, str, str]:
    """Return (body, content_type, final_url). Redirects are followed manually
    so each hop is re-validated; the body is size-capped."""
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        _check_public_url(current)
        async with ctx.http_client.stream(
            "GET", current, follow_redirects=False, timeout=60.0
        ) as resp:
            if resp.is_redirect:
                location = resp.headers.get("location")
                if not location:
                    raise SsrfError("redirect without a Location header")
                current = urljoin(current, location)
                continue
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            buf = bytearray()
            async for chunk in resp.aiter_bytes():
                buf.extend(chunk)
                if len(buf) > MAX_BYTES:
                    raise SsrfError(f"response exceeds {MAX_BYTES} bytes")
            return bytes(buf), ctype, current
    raise SsrfError(f"too many redirects (> {MAX_REDIRECTS})")


# --------------------------------------------------------------------------- #
# Render (page -> PDF) with sub-resource guarding
# --------------------------------------------------------------------------- #
async def _guard_route(route: Any) -> None:
    """Abort http(s) sub-resource requests to non-public hosts during render."""
    req_url = route.request.url
    if urlsplit(req_url).scheme in ("http", "https"):
        try:
            await asyncio.to_thread(_check_public_url, req_url)
        except SsrfError:
            await route.abort()
            return
        except Exception:  # noqa: BLE001 - never break a page on a resolver hiccup
            logger.warning("save_url_as_pdf: route guard resolver error for %s", req_url)
    await route.continue_()


async def _render_pdf(url: str) -> bytes:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RenderUnavailable(
            "rendering needs Playwright: `pip install playwright` then "
            "`playwright install chromium`"
        ) from exc

    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
        except Exception as exc:  # noqa: BLE001
            raise RenderUnavailable(
                f"could not launch Chromium (run `playwright install chromium`): {exc}"
            ) from exc
        try:
            page = await browser.new_page()
            await page.route("**/*", _guard_route)
            await page.goto(url, wait_until="load", timeout=RENDER_TIMEOUT_MS)
            return await page.pdf(format="A4", print_background=True)
        finally:
            await browser.close()


# --------------------------------------------------------------------------- #
# Filenames + workspace write
# --------------------------------------------------------------------------- #
def _name_from_url(url: str) -> str:
    sp = urlsplit(url)
    tail = unquote(sp.path.rsplit("/", 1)[-1]) if sp.path else ""
    return tail or (sp.hostname or "page")


def _safe_pdf_name(name: str) -> str:
    base = os.path.basename((name or "").replace("\\", "/")).strip()
    stem = os.path.splitext(base)[0]
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem).strip("._") or "page"
    return f"{stem}.pdf"


def _write_to_workspace(name: str, data: bytes) -> str:
    workspace = Path(CONFIG.workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    dest = workspace / name
    # O_NOFOLLOW: never write THROUGH a (planted) symlink out of the workspace.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    fd = os.open(dest, flags, 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return name


# --------------------------------------------------------------------------- #
# Tool
# --------------------------------------------------------------------------- #
@TOOL_REGISTRY.tool(
    name="web.save_url_as_pdf",
    description=(
        "Fetch a web URL and save it as a PDF in the workspace. If the URL is "
        "already a PDF it is downloaded as-is; otherwise the page is rendered to "
        "a PDF with a headless browser. Only public web addresses are allowed "
        "(internal/loopback/private/tailnet hosts are refused). Returns the "
        "workspace-relative path — read it with Deno via WORKSPACE + '/' + path, "
        "or hand it to an upload tool (e.g. to send it to a device)."
    ),
    parameters=[
        ToolParameter(
            name="url",
            type="string",
            description="The http(s) URL of the page or PDF to save.",
        ),
        ToolParameter(
            name="filename",
            type="string",
            description=(
                "Optional name to save as; a .pdf extension is enforced. "
                "Defaults to a name derived from the URL."
            ),
            required=False,
            default=None,
        ),
    ],
)
async def save_url_as_pdf(
    ctx: ToolContext, url: str, filename: str | None = None
) -> dict[str, Any]:
    if not url:
        return {"error": "url is required"}

    try:
        data, ctype, final_url = await _fetch_guarded(ctx, url)
    except SsrfError as exc:
        return {"error": f"refused: {exc}"}
    except httpx.HTTPError as exc:
        return {"error": f"fetch failed: {exc}"}

    is_pdf = "application/pdf" in ctype.split(";")[0].strip().lower() or data[:5] == b"%PDF-"

    if is_pdf:
        pdf_bytes, kind = data, "downloaded"
    else:
        try:
            pdf_bytes = await _render_pdf(final_url)
        except RenderUnavailable as exc:
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"render failed: {exc}"}
        kind = "rendered"
        if len(pdf_bytes) > MAX_BYTES:
            return {"error": f"rendered PDF exceeds {MAX_BYTES} bytes"}

    name = _safe_pdf_name(filename or _name_from_url(final_url))
    try:
        saved = _write_to_workspace(name, pdf_bytes)
    except OSError as exc:
        return {"error": f"could not write {name!r} to the workspace: {exc}"}

    logger.info("save_url_as_pdf: %s -> %s (%d bytes, %s)", url, saved, len(pdf_bytes), kind)
    return {
        "path": saved,
        "kind": kind,
        "size_bytes": len(pdf_bytes),
        "source_url": final_url,
    }

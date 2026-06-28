"""Save a web URL as a PDF in the agent workspace.

Host-side (Python) tool: the Deno sandbox has no network, so fetching/rendering
happens here and the result lands in the workspace for the agent to use.

- If the URL is already a PDF, it's downloaded as-is.
- Otherwise the page is rendered to a PDF with a headless Chromium (Playwright).

## SSRF model (read before deploying)

Only public (globally-routable) addresses are allowed.

* **Download path** — closed in code: the host is resolved, every IP is checked,
  and the connection is *pinned* to the validated IP (Host header + TLS SNI keep
  cert verification against the real hostname). Each redirect hop is re-checked.
  This defeats DNS rebinding for downloads.

* **Render path** — a headless browser does its own DNS, follows redirects, and
  loads sub-resources we can't fully pin from here. The in-process checks
  (entry URL + a fail-closed per-request guard) are *best-effort only*. The
  **authoritative** control for rendering is **host egress filtering**: the
  process running victrola MUST be denied outbound access to loopback,
  link-local (169.254/16 incl. cloud metadata), RFC1918, CGNAT/tailnet
  (100.64/10), and unique-local ranges. Example (nftables, adjust to taste):

      table inet egress {
        chain out {
          type filter hook output priority 0;
          ip daddr { 127.0.0.0/8, 169.254.0.0/16, 10.0.0.0/8, 172.16.0.0/12,
                     192.168.0.0/16, 100.64.0.0/10 } drop
          ip6 daddr { ::1, fe80::/10, fc00::/7 } drop
        }
      }

  Without that egress policy the render path is rebinding-exploitable.

Chromium's own process sandbox is kept ON by default; set
``WEB_PDF_CHROMIUM_NO_SANDBOX=1`` only if the host can't otherwise launch it
(prefer running victrola as a non-root user / in a container instead).
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
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit
from uuid import uuid4

import httpx

from src.config import CONFIG
from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter

logger = logging.getLogger(__name__)

MAX_BYTES = 100 * 1024 * 1024  # 100 MB cap on download / rendered PDF
MAX_REDIRECTS = 5
RENDER_TIMEOUT_MS = 30_000
MAX_CONCURRENT_RENDERS = 2  # bound concurrent Chromium launches (memory/CPU)

# NAT64 translation prefixes (RFC 6052/8215); embedded IPv4 is in the low 32 bits.
_NAT64_PREFIXES = (
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/96"),
)

_render_sem = asyncio.Semaphore(MAX_CONCURRENT_RENDERS)


class SsrfError(Exception):
    """The URL is not allowed by the public-address policy."""


class RenderUnavailable(Exception):
    """Playwright / Chromium is not available to render a page."""


# --------------------------------------------------------------------------- #
# SSRF: public-address validation + connection pinning
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


def _resolve_pinned(url: str) -> tuple[str, str, int | None, str]:
    """Validate url and return (scheme, host, port, pinned_ip).

    Raises SsrfError unless it's http(s) and *every* resolved address is public.
    The pinned_ip is the address the caller must connect to, so the IP that was
    validated is the IP that's used (closes DNS rebinding)."""
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
    pinned: str | None = None
    for info in infos:
        ip = _normalize_ip(info[4][0])
        if not _is_public(ip):
            raise SsrfError(f"{host!r} -> {ip} is not a public address")
        if pinned is None:
            pinned = str(ip)
    assert pinned is not None
    return sp.scheme, host, port, pinned


def _check_public_url(url: str) -> None:
    """Validate-only wrapper (raises SsrfError); used for best-effort checks."""
    _resolve_pinned(url)


# --------------------------------------------------------------------------- #
# Download path (pinned, per-hop revalidation)
# --------------------------------------------------------------------------- #
async def _fetch_guarded(ctx: ToolContext, url: str) -> tuple[bytes, str, str]:
    """Return (body, content_type, final_url). Connects to the validated IP at
    every hop (pinned), follows redirects manually, and caps the body size."""
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        scheme, host, port, ip = _resolve_pinned(current)
        sp = urlsplit(current)
        ip_host = f"[{ip}]" if ":" in ip else ip
        ip_netloc = f"{ip_host}:{port}" if port else ip_host
        ip_url = urlunsplit((scheme, ip_netloc, sp.path or "/", sp.query, ""))
        host_header = f"{host}:{port}" if port else host
        async with ctx.http_client.stream(
            "GET",
            ip_url,
            headers={"Host": host_header},
            extensions={"sni_hostname": host},  # SNI + cert check vs the hostname
            follow_redirects=False,
            timeout=60.0,
        ) as resp:
            if resp.is_redirect:
                location = resp.headers.get("location")
                if not location:
                    raise SsrfError("redirect without a Location header")
                # Resolve relative to the real-host URL, then revalidate next hop.
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
# Render path (best-effort guard; host egress filtering is authoritative)
# --------------------------------------------------------------------------- #
async def _guard_route(route: Any) -> None:
    """Best-effort per-request guard during render. Fails CLOSED: any http(s)
    request that isn't provably public is aborted (a dropped sub-resource just
    means a missing image, not a failed render). The real control is egress
    filtering — this can't see browser-followed redirects."""
    req_url = route.request.url
    if urlsplit(req_url).scheme in ("http", "https"):
        try:
            await asyncio.to_thread(_check_public_url, req_url)
        except Exception:  # noqa: BLE001 - SsrfError OR resolver error -> abort
            await route.abort()
            return
    await route.continue_()


async def _render_pdf(url: str) -> bytes:
    # NOTE: callers pass a URL already validated by _fetch_guarded; the browser
    # re-resolves on its own, so egress filtering (not this code) is the real
    # SSRF control here.
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RenderUnavailable(
            "rendering needs Playwright: `pip install playwright` then "
            "`playwright install chromium`"
        ) from exc

    launch_args = ["--disable-dev-shm-usage"]
    if os.environ.get("WEB_PDF_CHROMIUM_NO_SANDBOX"):
        launch_args.append("--no-sandbox")

    async with _render_sem:  # bound concurrent Chromium launches
        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(args=launch_args)
            except Exception as exc:  # noqa: BLE001
                raise RenderUnavailable(
                    "could not launch Chromium (run `playwright install chromium`; "
                    "on locked-down Linux you may need WEB_PDF_CHROMIUM_NO_SANDBOX=1 "
                    f"and a non-root/container setup): {exc}"
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
    """Write `data` to <workspace>/<name> safely.

    Writes to a fresh temp file (O_EXCL|O_NOFOLLOW => guaranteed new regular
    file) then atomically renames it onto `name`. os.replace swaps the directory
    ENTRY, so even if `name` is already a symlink or a hardlink to a file
    OUTSIDE the workspace, that outside inode is never written through.
    """
    workspace = Path(CONFIG.workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    dest = workspace / name
    tmp = workspace / f".{name}.{os.getpid()}.{uuid4().hex}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
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

    media = ctype.split(";")[0].strip().lower()
    is_pdf = media == "application/pdf" or data[:5] == b"%PDF-"

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

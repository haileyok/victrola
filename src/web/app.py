"""FastAPI app factory — assembles routers, serves static frontend."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import CONFIG
from src.web.routers import (
    chat,
    mcp,
    memory,
    schedules,
    secrets,
    sessions,
    status,
    system_prompt,
    tools,
    workspace,
)

# Methods that can change server state — these require Origin validation
# to prevent CSRF attacks from malicious cross-origin pages.
_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

# Hostnames allowed for same-origin requests. Always includes loopback so local
# access works. The configured bind host is added, plus any extra hostnames the
# operator lists in WEB_ALLOWED_HOSTS (e.g. a Tailscale host) for cases where
# the bind is 0.0.0.0 but browsers reach the server via a non-loopback name.
# Lowercased because urlparse().hostname always lowercases.
_EXTRA_HOSTS = {
    h.strip().lower() for h in CONFIG.web_allowed_hosts.split(",") if h.strip()
}
_ALLOWED_HOSTS = {
    "localhost", "127.0.0.1", "::1", CONFIG.web_host.lower()
} | _EXTRA_HOSTS


def _host_header_hostname(host: str | None) -> str | None:
    """Extract the hostname from a Host header value, or None if malformed.

    Handles ports and bracketed IPv6 literals (``[::1]:8000`` -> ``::1``),
    which a naive ``split(":")`` gets wrong.
    """
    if not host:
        return None
    try:
        return urlparse(f"//{host}").hostname
    except ValueError:
        return None


class CsrfMiddleware(BaseHTTPMiddleware):
    """Reject requests from untrusted hosts and cross-origin state changes.

    Two checks, both against _ALLOWED_HOSTS:

    1. Host header, on every request (DNS rebinding defense). A malicious
       website can re-point its own domain at this server's address; the
       browser then treats the server as same-origin with the attacker's page
       and lets it *read* responses (sessions, memory, workspace files). The
       Host header still carries the attacker's domain in that case, so
       rejecting unknown Host values blocks it.

    2. Origin header, on unsafe methods (CSRF defense). A malicious website the
       operator visits could otherwise submit cross-origin POSTs to e.g.
       /api/tools/<name>/approve. If Origin is present, its hostname must be
       allowed. Requests without Origin (non-browser clients) pass.

    We use urlparse to avoid prefix-matching bypasses like http://localhost.evil.com.
    """

    async def dispatch(self, request: Request, call_next):
        if _host_header_hostname(request.headers.get("host")) not in _ALLOWED_HOSTS:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": "Host not allowed. Add this hostname to "
                    "WEB_ALLOWED_HOSTS if you access Victrola through it."
                },
            )
        if request.method in _UNSAFE_METHODS:
            origin = request.headers.get("origin")
            if origin:
                parsed = urlparse(origin)
                if parsed.hostname not in _ALLOWED_HOSTS:
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "Cross-origin requests not allowed"},
                    )
        return await call_next(request)


def create_app(agent, executor, conversation_manager) -> FastAPI:
    app = FastAPI(title="Victrola")
    app.state.agent = agent
    app.state.executor = executor
    app.state.conversation_manager = conversation_manager

    app.add_middleware(CsrfMiddleware)

    # mount API routers
    app.include_router(status.router, prefix="/api")
    app.include_router(sessions.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(tools.router, prefix="/api")
    app.include_router(secrets.router, prefix="/api")
    app.include_router(schedules.router, prefix="/api")
    app.include_router(mcp.router, prefix="/api")
    app.include_router(system_prompt.router, prefix="/api")
    app.include_router(memory.router, prefix="/api")
    app.include_router(workspace.router, prefix="/api")

    static_dir = Path(__file__).parent / "static"

    if static_dir.exists():
        # serve static assets (JS/CSS bundles) at /assets
        assets_dir = static_dir / "assets"
        if assets_dir.exists():
            app.mount(
                "/assets",
                StaticFiles(directory=str(assets_dir)),
                name="assets",
            )

        # SPA catch-all: client-side routes like /sessions/:id won't match
        # static files on refresh/direct-nav, so serve index.html as fallback.
        @app.get("/{path:path}")
        async def spa_fallback(path: str):
            if path.startswith("api/"):
                raise HTTPException(404)
            # try to serve a real static file first (favicon, etc.)
            # resolve and verify the file is inside static_dir to prevent
            # path traversal via encoded ../ segments
            file_path = (static_dir / path).resolve()
            try:
                file_path.relative_to(static_dir.resolve())
            except ValueError:
                raise HTTPException(404)
            if path and file_path.is_file():
                return FileResponse(str(file_path))
            # fall back to index.html for client-side routing
            index = static_dir / "index.html"
            if index.exists():
                return FileResponse(str(index))
            raise HTTPException(404, "Frontend index.html not found")
    else:
        @app.get("/{path:path}")
        async def spa_fallback(path: str):
            if path.startswith("api/"):
                raise HTTPException(404)
            raise HTTPException(
                404,
                "Frontend not built. Run: cd web && npm run build",
            )

    return app

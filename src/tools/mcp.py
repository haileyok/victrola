"""MCP (Model Context Protocol) client support.

Connects to external MCP servers, discovers their tools, and registers
approved tools into the TOOL_REGISTRY so the agent can call them via
the standard `tools.<server>.<method>()` pipeline inside execute_code.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import AnyUrl

from src.config import CONFIG
from src.tools.registry import Tool, ToolParameter

if TYPE_CHECKING:
    from src.store.store import Store
    from src.tools.registry import ToolRegistry
    from src.tools.secrets import SecretManager

logger = logging.getLogger(__name__)

MCP_RKEY_PREFIX = "mcpserver:"

# Existing tool namespaces that MCP server names must not collide with
_RESERVED_NAMESPACES = {
    "memory",
    "web",
    "notify",
    "scheduler",
    "summarize",
    "image",
    "custom_tools",
}

# TypeScript/JavaScript reserved words that must not be used as identifiers
_TS_RESERVED_WORDS = frozenset({
    "break", "case", "catch", "class", "const", "continue", "debugger", "default",
    "delete", "do", "else", "enum", "export", "extends", "false", "finally", "for",
    "function", "if", "import", "in", "instanceof", "new", "null", "return", "super",
    "switch", "this", "throw", "true", "try", "typeof", "var", "void", "while", "with",
    "yield", "let", "static", "await", "async", "implements", "interface", "package",
    "private", "protected", "public", "type", "as", "from", "of", "namespace",
})

# Valid server name: must be a valid TypeScript identifier (letter/underscore start, alphanumeric/underscore body)
_SERVER_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")

# Sanitize discovered tool names to valid TS identifiers
_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9_]")

# Bounded timeout for connecting to an MCP server
_CONNECT_TIMEOUT = 30.0

# Bounded timeout for individual tool calls and list_tools
_CALL_TIMEOUT = 60.0

# Bounded timeout for auto-reconnection (disconnect + reconnect).
# Bounds the full OAuth flow (client registration, token exchange, callback).
_RECONNECT_TIMEOUT = 20.0

# Best-effort timeout for closing an old transport during reconnect.
# If cleanup doesn't finish in this time, the background task is abandoned.
_DETACHED_DISCONNECT_TIMEOUT = 10.0


async def _detached_disconnect(conn: "MCPConnection") -> None:
    """Close a connection in the background with a best-effort timeout.

    Prevents a hung transport close (cancellation-resistant aclose) from
    blocking the caller. If the close doesn't finish in time, the task
    is abandoned — the stale connection leaks but the caller proceeds.
    """
    try:
        await asyncio.wait_for(conn.disconnect(), timeout=_DETACHED_DISCONNECT_TIMEOUT)
    except Exception:
        logger.warning("Detached disconnect failed (abandoned)", exc_info=True)


@dataclass
class MCPTool:
    """Per-tool metadata discovered from an MCP server, with approval state."""

    name: str  # original MCP tool name (may contain dots, etc.)
    description: str
    input_schema: dict[str, Any]
    approved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "approved": self.approved,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPTool":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            input_schema=data.get("input_schema", {}),
            approved=data.get("approved", False),
        )


@dataclass
class MCPServerConfig:
    """Persisted server configuration + discovered tools."""

    name: str
    transport: Literal["sse", "stdio", "streamable_http"]
    url: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    auth_type: Literal["none", "bearer", "oauth"] = "none"
    auth_token_secret: str | None = None
    env_secrets: list[str] = field(default_factory=list)
    enabled: bool = True
    tools: list[MCPTool] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "transport": self.transport,
            "url": self.url,
            "command": self.command,
            "args": self.args,
            "auth_type": self.auth_type,
            "auth_token_secret": self.auth_token_secret,
            "env_secrets": self.env_secrets,
            "enabled": self.enabled,
            "tools": [t.to_dict() for t in self.tools],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPServerConfig":
        tools_data = data.get("tools", [])
        return cls(
            name=data["name"],
            transport=data.get("transport", "sse"),
            url=data.get("url"),
            command=data.get("command"),
            args=data.get("args", []),
            auth_type=data.get("auth_type", data.get("authType", "none")),
            auth_token_secret=data.get("auth_token_secret"),
            env_secrets=data.get("env_secrets", []),
            enabled=data.get("enabled", True),
            tools=[MCPTool.from_dict(t) for t in tools_data],
        )


class MCPConnection:
    """Manages a single MCP server connection lifecycle.

    Uses an AsyncExitStack to hold the transport + session context managers
    open persistently across tool calls.
    """

    def __init__(
        self, config: MCPServerConfig, secret_manager: SecretManager | None,
        oauth_provider: Any | None = None,
    ) -> None:
        self._config = config
        self._secret_manager = secret_manager
        self._session: Any | None = None
        self._stack: contextlib.AsyncExitStack | None = None
        self._oauth_provider = oauth_provider

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        self._stack = contextlib.AsyncExitStack()

        try:
            # resolve auth based on type
            headers: dict[str, str] = {}
            auth: Any = None

            if self._config.auth_type == "bearer" and self._config.auth_token_secret and self._secret_manager:
                token = self._secret_manager.get_secret(self._config.auth_token_secret)
                if token:
                    headers["Authorization"] = f"Bearer {token}"
            elif self._config.auth_type == "oauth" and self._oauth_provider is not None:
                # OAuthClientProvider extends httpx.Auth, pass it directly
                auth = self._oauth_provider

            if self._config.transport == "sse":
                if not self._config.url:
                    raise ValueError(f"SSE transport requires URL for server '{self._config.name}'")
                from mcp.client.sse import sse_client

                read, write = await self._stack.enter_async_context(
                    sse_client(
                        self._config.url,
                        headers=headers or None,
                        auth=auth,
                        timeout=30.0,
                    )
                )
            elif self._config.transport == "streamable_http":
                if not self._config.url:
                    raise ValueError(f"Streamable HTTP transport requires URL for server '{self._config.name}'")
                from mcp.client.streamable_http import streamablehttp_client

                result = await self._stack.enter_async_context(
                    streamablehttp_client(
                        self._config.url,
                        headers=headers or None,
                        auth=auth,
                        timeout=30.0,
                        sse_read_timeout=float(CONFIG.mcp_sse_read_timeout_seconds),
                    )
                )
                # streamablehttp_client returns (read, write, get_session_id)
                read, write = result[0], result[1]
            elif self._config.transport == "stdio":
                if not self._config.command:
                    raise ValueError(
                        f"stdio transport requires command for server '{self._config.name}'"
                    )
                from mcp.client.stdio import StdioServerParameters, stdio_client

                env: dict[str, str] = {}
                if self._secret_manager:
                    for secret_name in self._config.env_secrets:
                        val = self._secret_manager.get_secret(secret_name)
                        if val:
                            env[secret_name.upper()] = val

                params = StdioServerParameters(
                    command=self._config.command,
                    args=self._config.args,
                    env=env or None,
                )
                read, write = await self._stack.enter_async_context(stdio_client(params))
            else:
                raise ValueError(f"Unknown transport: {self._config.transport}")

            from mcp.client.session import ClientSession

            self._session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            await asyncio.wait_for(self._session.initialize(), timeout=_CONNECT_TIMEOUT)
            logger.info("Connected to MCP server '%s'", self._config.name)
        except BaseException:
            # Clean up the exit stack if any step failed — otherwise
            # transports/subprocesses entered into the stack would leak.
            # Bound the cleanup so a hung transport close can't block the
            # caller indefinitely (e.g. during wait_for cancellation in
            # auto-reconnect). If the transport's aclose() is truly
            # cancellation-resistant, the wait_for itself may delay — this
            # is an asyncio limitation. The session/stack are nulled
            # regardless so the connection is marked dead.
            try:
                await asyncio.wait_for(
                    self.disconnect(), timeout=_DETACHED_DISCONNECT_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.warning("disconnect() timed out during connect cleanup — abandoning transport")
                self._session = None
                self._stack = None
            except Exception:
                logger.warning("Error during connect cleanup disconnect", exc_info=True)
            raise

    async def disconnect(self) -> None:
        """Close the connection."""
        self._session = None
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:
                logger.warning("Error closing MCP connection", exc_info=True)
            self._stack = None

    async def list_tools(self) -> list[Any]:
        """List tools from the MCP server."""
        if self._session is None:
            raise RuntimeError("Not connected")
        result = await asyncio.wait_for(
            self._session.list_tools(), timeout=_CALL_TIMEOUT
        )
        return result.tools

    async def send_ping(self) -> None:
        """Send a health-check ping. Raises if the connection is dead."""
        session = self._session
        if session is None:
            raise RuntimeError("Not connected")
        await asyncio.wait_for(session.send_ping(), timeout=15.0)

    async def call_tool(self, name: str, params: dict[str, Any]) -> Any:
        """Call a tool on the MCP server and convert the result."""
        if self._session is None:
            raise RuntimeError("Not connected")

        result = await asyncio.wait_for(
            self._session.call_tool(name, params), timeout=_CALL_TIMEOUT
        )

        # convert CallToolResult to a plain Python value
        if result.isError:
            texts = [
                c.text for c in (result.content or []) if hasattr(c, "text")
            ]
            return {"error": " ".join(texts) if texts else "MCP tool returned an error"}

        items: list[Any] = []
        for content in result.content or []:
            if hasattr(content, "text"):
                items.append(content.text)
            elif hasattr(content, "data"):
                items.append(
                    {
                        "type": "image",
                        "data": content.data,
                        "mimeType": getattr(content, "mimeType", "image/png"),
                    }
                )
            else:
                items.append(str(content))

        # single text item → return string directly
        if len(items) == 1:
            return items[0]
        if len(items) > 1:
            return {"content": items}
        return None


class MCPOAuthTokenStorage:
    """Persists OAuth tokens per-server using the DocumentStore.

    Implements the MCP SDK's TokenStorage protocol so it can be passed
    to OAuthClientProvider. Tokens are stored as JSON documents with
    the rkey prefix 'mcptoken:'.
    """

    def __init__(self, store: Store, server_name: str) -> None:
        self._store = store
        self._server_name = server_name
        self._rkey = f"mcptoken:{server_name}"

    async def get_tokens(self) -> Any:
        from mcp.shared.auth import OAuthToken

        if self._store.documents is None:
            return None
        try:
            doc = await self._store.documents.get(self._rkey)
            data = json.loads(doc["content"])
            return OAuthToken.model_validate(data)
        except Exception:
            return None

    async def set_tokens(self, tokens: Any) -> None:
        if self._store.documents is None:
            raise RuntimeError("DocumentStore is not initialized")
        content = json.dumps(tokens.model_dump())
        try:
            await self._store.documents.update(self._rkey, content)
        except Exception:
            from src.store.store import StoreNotFound

            try:
                await self._store.documents.create(self._rkey, content)
            except Exception:
                logger.warning("Failed to store OAuth tokens for '%s'", self._server_name, exc_info=True)

    async def get_client_info(self) -> Any:
        return None

    async def set_client_info(self, client_info: Any) -> None:
        pass  # We don't persist client info — it's regenerated each time


class MCPManager:
    """Manages all MCP servers — parallel to CustomToolManager."""

    def __init__(
        self,
        store: Store,
        secret_manager: SecretManager | None,
        registry: ToolRegistry,
    ) -> None:
        self._store = store
        self._secret_manager = secret_manager
        self._registry = registry
        self._servers: dict[str, MCPServerConfig] = {}
        self._connections: dict[str, MCPConnection] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._health_task: asyncio.Task | None = None

    def _get_lock(self, name: str) -> asyncio.Lock:
        """Get or create a per-server lock for serializing operations."""
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]

    # -- loading from store --

    async def load_servers(self) -> None:
        """Load all MCP server configs from the DocumentStore."""
        self._servers.clear()
        cursor: str | None = None

        if self._store.documents is None:
            raise RuntimeError("DocumentStore is not initialized")

        while True:
            resp = await self._store.documents.list(limit=100, cursor=cursor)
            documents = resp.get("documents", [])

            for doc in documents:
                rkey = doc.get("rkey", "")
                if not rkey.startswith(MCP_RKEY_PREFIX):
                    continue
                content = doc.get("content", "")
                if not content:
                    continue
                try:
                    data = json.loads(content)
                    config = MCPServerConfig.from_dict(data)
                    self._servers[config.name] = config
                except Exception as e:
                    logger.warning("Failed to parse MCP server config %s: %s", rkey, e)

            cursor = resp.get("cursor")
            if not cursor or not documents:
                break

        logger.info("Loaded %d MCP server(s)", len(self._servers))

    # -- connection lifecycle --

    async def connect_server(self, name: str) -> None:
        """Connect to a server, discover tools, and register approved tools."""
        async with self._get_lock(name):
            await self._connect_server_impl(name)
        self.start_health_monitor(float(CONFIG.mcp_health_check_interval_seconds))

    async def _connect_server_impl(self, name: str, *, auto: bool = False) -> None:
        config = self._servers.get(name)
        if config is None:
            raise ValueError(f"MCP server '{name}' not found")

        # disconnect existing connection if any
        if name in self._connections:
            await self._disconnect_server_impl(name)

        # build OAuth provider if needed
        oauth_provider = None
        if config.auth_type == "oauth" and config.url:
            callback_timeout = 5.0 if auto else 300.0
            oauth_provider = self._create_oauth_provider(
                name, config.url, callback_timeout=callback_timeout
            )

        conn = MCPConnection(config, self._secret_manager, oauth_provider=oauth_provider)
        await conn.connect()
        self._connections[name] = conn

        try:
            # discover tools (updates config.tools)
            await self._discover_tools_impl(name)

            # register any previously-approved tools
            for tool in config.tools:
                if tool.approved:
                    self._register_tool(name, tool)
        except BaseException:
            # Clean up the connection if discovery or registration failed
            # so we don't leave a live session with no tools registered.
            # Use detached cleanup so a hung transport close can't block
            # the caller (same pattern as auto-reconnect).
            self._cleanup_connection(name)
            raise

    def _cleanup_connection(self, name: str) -> None:
        """Unregister tools, pop connection, and schedule background cleanup.

        Synchronous tool unregistration + non-blocking transport close.
        Caller must hold the per-server lock (or be in a context where no
        other task can touch this server).
        """
        config = self._servers.get(name)
        if config:
            for tool in config.tools:
                if tool.approved:
                    self._registry.unregister(
                        f"{name}.{self._sanitize_tool_name(tool.name)}"
                    )
        conn = self._connections.pop(name, None)
        if conn:
            conn._session = None
            asyncio.create_task(_detached_disconnect(conn))

    async def disconnect_server(self, name: str) -> None:
        """Disconnect a server and unregister its tools."""
        async with self._get_lock(name):
            await self._disconnect_server_impl(name)

    async def _disconnect_server_impl(self, name: str) -> None:
        # unregister all approved tools for this server
        config = self._servers.get(name)
        if config:
            for tool in config.tools:
                if tool.approved:
                    self._registry.unregister(f"{name}.{self._sanitize_tool_name(tool.name)}")

        conn = self._connections.pop(name, None)
        if conn:
            await conn.disconnect()

    def start_health_monitor(self, interval: float) -> None:
        """Start the background health check loop. Safe to call if already running."""
        if self._health_task is not None and not self._health_task.done():
            return
        if interval <= 0:
            return
        self._health_task = asyncio.create_task(self._health_check_loop(interval))

    async def stop_health_monitor(self) -> None:
        """Cancel the health check loop and await its completion."""
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning("Health monitor task raised on cancel", exc_info=True)
        self._health_task = None

    async def _health_check_loop(self, interval: float) -> None:
        """Periodically ping each connected server and reconnect dead ones."""
        while True:
            await asyncio.sleep(interval)
            # Top-level guard: a single iteration failure must not kill the loop.
            try:
                for name in list(self._connections.keys()):
                    try:
                        await asyncio.wait_for(
                            self._health_check_server(name), timeout=20.0
                        )
                    except Exception as e:
                        logger.warning("MCP health check failed for '%s': %s", name, e)
                        await self._reconnect_server(name)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Unexpected error in health check loop", exc_info=True)

    async def _health_check_server(self, name: str) -> None:
        """Ping a single server to verify the connection is alive.

        Uses a short try-acquire on the per-server lock so a long-running
        tool call doesn't block the health check — it just skips this iteration.
        """
        lock = self._get_lock(name)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=1.0)
        except asyncio.TimeoutError:
            logger.debug("Skipping health check for '%s' — lock held by active call", name)
            return
        try:
            conn = self._connections.get(name)
            if conn is None or not conn.is_connected:
                return
            await conn.send_ping()
        finally:
            lock.release()

    async def _reconnect_server(self, name: str) -> bool:
        """Disconnect and reconnect a server under its lock.
        Uses auto=True (short OAuth callback timeout). Returns True on success.

        The entire disconnect+connect sequence is bounded by _RECONNECT_TIMEOUT.
        On timeout, the connection is popped from _connections without awaiting
        cleanup — the cancelled task's cleanup runs independently and the stale
        connection is left for GC.
        """
        async with self._get_lock(name):
            # Guard: a manual disconnect or delete may have already removed
            # the connection between the health check failure and here.
            if name not in self._connections:
                return False
            config = self._servers.get(name)
            if config is None or not config.enabled:
                return False
            try:
                await asyncio.wait_for(
                    self._reconnect_server_impl(name),
                    timeout=_RECONNECT_TIMEOUT,
                )
                return True
            except asyncio.TimeoutError:
                logger.warning("Auto-reconnect timed out for '%s'", name)
                # Pop and schedule background cleanup. The wait_for cancellation
                # may still be in progress; don't await disconnect synchronously.
                conn = self._connections.pop(name, None)
                if conn:
                    conn._session = None
                    asyncio.create_task(_detached_disconnect(conn))
                return False
            except Exception as e:
                logger.warning("Auto-reconnect failed for '%s': %s", name, e)
                if name in self._connections:
                    await self._disconnect_server_impl(name)
                return False

    async def _reconnect_server_impl(self, name: str) -> None:
        """Disconnect and reconnect. Caller must hold the per-server lock.

        The old connection is popped immediately and its cleanup is scheduled
        as a background task with a best-effort timeout. This prevents a hung
        transport close from blocking the reconnect while still attempting
        proper resource cleanup.
        """
        conn = self._connections.pop(name, None)
        if conn:
            conn._session = None
            asyncio.create_task(_detached_disconnect(conn))
        await self._connect_server_impl(name, auto=True)

    async def connect_all(self) -> None:
        """Connect to all enabled servers. Non-fatal on failure."""
        for name, config in list(self._servers.items()):
            if not config.enabled:
                continue
            # Skip OAuth servers — they need manual authorization via the web UI.
            # Even if tokens are stored, they may be expired and the refresh flow
            # can block for minutes waiting for a callback that won't come during startup.
            if config.auth_type == "oauth":
                logger.info(
                    "Skipping OAuth MCP server '%s' on startup — "
                    "connect manually via the web UI", name
                )
                continue
            try:
                await asyncio.wait_for(self.connect_server(name), timeout=_CONNECT_TIMEOUT)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(
                    "Failed to connect to MCP server '%s': %s", name, e
                )
        self.start_health_monitor(float(CONFIG.mcp_health_check_interval_seconds))

    async def disconnect_all(self) -> None:
        """Disconnect all connected servers."""
        await self.stop_health_monitor()
        for name in list(self._connections.keys()):
            await self.disconnect_server(name)

    # -- tool discovery --

    async def discover_tools(self, name: str) -> None:
        """Re-discover tools from a connected server. Persists updated config."""
        async with self._get_lock(name):
            await self._discover_tools_impl(name)

    async def _discover_tools_impl(self, name: str) -> None:
        config = self._servers.get(name)
        if config is None:
            raise ValueError(f"MCP server '{name}' not found")

        conn = self._connections.get(name)
        if conn is None or not conn.is_connected:
            raise RuntimeError(f"MCP server '{name}' is not connected")

        server_tools = await conn.list_tools()

        # build a lookup of existing tools by original name to preserve approval
        existing = {t.name: t for t in config.tools}
        new_tools: list[MCPTool] = []
        seen_sanitized: dict[str, str] = {}  # sanitized_name -> original_name

        for st in server_tools:
            # MCP SDK Tool has: name, description, inputSchema
            tool_name = st.name
            description = st.description or ""
            input_schema = st.inputSchema if st.inputSchema else {}

            # check for sanitized name collisions
            sanitized = self._sanitize_tool_name(tool_name)
            if sanitized in seen_sanitized:
                logger.warning(
                    "MCP server '%s': tool '%s' collides with '%s' after sanitization "
                    "(both map to '%s') — skipping duplicate",
                    name, tool_name, seen_sanitized[sanitized], sanitized,
                )
                continue
            seen_sanitized[sanitized] = tool_name

            if tool_name in existing:
                # preserve approval state
                old = existing[tool_name]
                new_tool = MCPTool(
                    name=tool_name,
                    description=description,
                    input_schema=input_schema,
                    approved=old.approved,
                )
                # re-register if approved and anything registry-facing changed
                if old.approved and (
                    old.input_schema != input_schema
                    or old.description != description
                ):
                    # unregister old first to avoid stale handler
                    self._registry.unregister(f"{name}.{sanitized}")
                    self._register_tool(name, new_tool)
            else:
                new_tool = MCPTool(
                    name=tool_name,
                    description=description,
                    input_schema=input_schema,
                    approved=False,
                )
            new_tools.append(new_tool)

        # for tools that disappeared from the server or were skipped due to collision
        retained_names = {t.name for t in new_tools}
        for old_tool in config.tools:
            if old_tool.name not in retained_names and old_tool.approved:
                self._registry.unregister(
                    f"{name}.{self._sanitize_tool_name(old_tool.name)}"
                )

        config.tools = new_tools

        # persist updated config
        await self._persist_server(name)

        logger.info(
            "Discovered %d tool(s) from MCP server '%s'",
            len(new_tools),
            name,
        )

    # -- approval / revocation --

    async def approve_tool(self, server_name: str, tool_name: str) -> str:
        """Approve a tool — registers it in the registry."""
        async with self._get_lock(server_name):
            return await self._approve_tool_impl(server_name, tool_name)

    async def _approve_tool_impl(self, server_name: str, tool_name: str) -> str:
        config = self._servers.get(server_name)
        if config is None:
            return f"MCP server '{server_name}' not found."

        # find tool by original name
        tool = None
        for t in config.tools:
            if t.name == tool_name:
                tool = t
                break
        if tool is None:
            return f"Tool '{tool_name}' not found on server '{server_name}'."

        tool.approved = True
        await self._persist_server(server_name)

        self._register_tool(server_name, tool)
        return f"MCP tool '{server_name}.{self._sanitize_tool_name(tool_name)}' approved."

    async def revoke_tool(self, server_name: str, tool_name: str) -> str:
        """Revoke a tool — unregisters it from the registry."""
        async with self._get_lock(server_name):
            return await self._revoke_tool_impl(server_name, tool_name)

    async def _revoke_tool_impl(self, server_name: str, tool_name: str) -> str:
        config = self._servers.get(server_name)
        if config is None:
            return f"MCP server '{server_name}' not found."

        tool = None
        for t in config.tools:
            if t.name == tool_name:
                tool = t
                break
        if tool is None:
            return f"Tool '{tool_name}' not found on server '{server_name}'."

        tool.approved = False
        await self._persist_server(server_name)

        sanitized = self._sanitize_tool_name(tool_name)
        self._registry.unregister(f"{server_name}.{sanitized}")
        return f"MCP tool '{server_name}.{sanitized}' approval revoked."

    # -- tool execution --

    async def call_tool(
        self, server_name: str, tool_name: str, params: dict[str, Any]
    ) -> Any:
        """Proxy a tool call to the MCP server.

        Holds the per-server lock for the entire call so the health monitor
        can't tear down the session mid-call. If the call fails, attempts one
        bounded reconnection and retries.
        """
        async with self._get_lock(server_name):
            conn = self._connections.get(server_name)
            if conn is None or not conn.is_connected:
                return {
                    "error": f"MCP server '{server_name}' is not connected. "
                    "The operator can reconnect it via the web UI."
                }
            try:
                return await conn.call_tool(tool_name, params)
            except Exception as e:
                logger.warning("MCP tool call failed: %s.%s: %s", server_name, tool_name, e)
                # Attempt one bounded reconnection + retry. The full disconnect
                # + connect (including OAuth round-trips) is bounded by
                # _RECONNECT_TIMEOUT so the agent isn't blocked indefinitely.
                try:
                    await asyncio.wait_for(
                        self._reconnect_server_impl(server_name),
                        timeout=_RECONNECT_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    # Pop and schedule background cleanup. The wait_for
                    # cancellation may still be in progress; don't await
                    # disconnect synchronously.
                    conn = self._connections.pop(server_name, None)
                    if conn:
                        conn._session = None
                        asyncio.create_task(_detached_disconnect(conn))
                    return {"error": f"MCP tool call failed: {e}. Auto-reconnect timed out."}
                except Exception as reconnect_err:
                    if server_name in self._connections:
                        await self._disconnect_server_impl(server_name)
                    return {"error": f"MCP tool call failed: {e}. Reconnect failed: {reconnect_err}"}
                conn = self._connections.get(server_name)
                if conn is None or not conn.is_connected:
                    return {"error": f"MCP tool call failed: {e}. Reconnect produced no active connection."}
                try:
                    return await conn.call_tool(tool_name, params)
                except Exception as retry_err:
                    return {"error": f"MCP tool call failed: {e}. Reconnected but retry also failed: {retry_err}"}

    # -- server CRUD --

    async def create_server(self, config: MCPServerConfig) -> str:
        """Create a new MCP server config."""
        # validate server name
        if not _SERVER_NAME_PATTERN.match(config.name):
            return (
                f"Error: server name '{config.name}' is invalid. "
                "Allowed: letter or underscore start, then letters, digits, or underscore (1-64 chars)"
            )
        if config.name in _RESERVED_NAMESPACES:
            return f"Error: server name '{config.name}' is a reserved namespace."
        if config.name in _TS_RESERVED_WORDS:
            return f"Error: server name '{config.name}' is a TypeScript reserved word."
        if config.name in self._servers:
            return f"Error: server '{config.name}' already exists."

        # validate transport-specific fields
        if config.transport == "sse" and not config.url:
            return "Error: SSE transport requires a URL."
        if config.transport == "stdio" and not config.command:
            return "Error: stdio transport requires a command."

        # backward compat: if auth_token_secret is set but auth_type is "none",
        # upgrade to "bearer"
        if config.auth_token_secret and config.auth_type == "none":
            config.auth_type = "bearer"

        # OAuth only works with SSE or streamable_http transport
        if config.auth_type == "oauth" and config.transport not in ("sse", "streamable_http"):
            return "Error: OAuth auth_type requires SSE or streamable_http transport."

        rkey = f"{MCP_RKEY_PREFIX}{config.name}"
        content = json.dumps(config.to_dict())
        if self._store.documents is None:
            raise RuntimeError("DocumentStore is not initialized")

        try:
            await self._store.documents.create(rkey, content)
        except Exception as e:
            from src.store.store import StoreConflict

            if isinstance(e, StoreConflict):
                return f"Error: server '{config.name}' already exists."
            raise

        self._servers[config.name] = config
        return f"MCP server '{config.name}' created."

    async def update_server(self, name: str, **fields: Any) -> str:
        """Update an existing server config. Reconnects if connection-affecting fields change."""
        async with self._get_lock(name):
            config = self._servers.get(name)
            if config is None:
                return f"MCP server '{name}' not found."

            connection_fields = {"transport", "url", "command", "args", "auth_token_secret", "env_secrets"}
            needs_reconnect = False
            was_connected = name in self._connections

            for key, value in fields.items():
                if value is None:
                    continue
                if key in connection_fields and getattr(config, key) != value:
                    needs_reconnect = True
                if hasattr(config, key):
                    setattr(config, key, value)

            await self._persist_server(name)

            if needs_reconnect and was_connected:
                # disconnect old connection, then reconnect under the same lock
                if name in self._connections:
                    await self._disconnect_server_impl(name)
                try:
                    await self._connect_server_impl(name)
                except Exception as e:
                    logger.warning("Failed to reconnect MCP server '%s': %s", name, e)
                    return f"MCP server '{name}' updated, but reconnection failed: {e}"

        return f"MCP server '{name}' updated."

    async def delete_server(self, name: str) -> str:
        """Delete a server — disconnects and unregisters tools first."""
        async with self._get_lock(name):
            config = self._servers.get(name)
            if config is None:
                return f"MCP server '{name}' not found."

            # disconnect (also unregisters tools)
            if name in self._connections:
                await self._disconnect_server_impl(name)
            else:
                # even if not connected, unregister any approved tools
                for tool in config.tools:
                    if tool.approved:
                        self._registry.unregister(
                            f"{name}.{self._sanitize_tool_name(tool.name)}"
                        )

            rkey = f"{MCP_RKEY_PREFIX}{name}"
            if self._store.documents is None:
                raise RuntimeError("DocumentStore is not initialized")
            try:
                await self._store.documents.delete(rkey)
            except Exception:
                logger.exception("Failed to delete MCP server doc %s", rkey)

            self._servers.pop(name, None)
        return f"MCP server '{name}' deleted."

    # -- read-only accessors --

    def list_servers(self) -> list[MCPServerConfig]:
        return list(self._servers.values())

    def get_server(self, name: str) -> MCPServerConfig | None:
        return self._servers.get(name)

    def is_connected(self, name: str) -> bool:
        conn = self._connections.get(name)
        return conn is not None and conn.is_connected

    def get_oauth_status(self, name: str) -> str:
        """Return OAuth authorization status for a server.

        Returns: 'not_configured' | 'not_authorized' | 'authorized' | 'unknown'
        """
        config = self._servers.get(name)
        if config is None or config.auth_type != "oauth":
            return "not_configured"

        # check if we have stored tokens
        if self._store.documents is None:
            return "not_authorized"
        import asyncio

        async def _check() -> str:
            rkey = f"mcptoken:{name}"
            try:
                await self._store.documents.get(rkey)
                return "authorized"
            except Exception:
                return "not_authorized"

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context — can't easily run async here.
                # The web layer should call the async version instead.
                return "unknown"
            return asyncio.run(_check())
        except RuntimeError:
            return "unknown"

    async def get_oauth_status_async(self, name: str) -> str:
        """Async version of get_oauth_status."""
        config = self._servers.get(name)
        if config is None or config.auth_type != "oauth":
            return "not_configured"

        if self._store.documents is None:
            return "not_authorized"

        rkey = f"mcptoken:{name}"
        try:
            await self._store.documents.get(rkey)
            return "authorized"
        except Exception:
            return "not_authorized"

    async def clear_oauth_tokens(self, name: str) -> str:
        """Delete stored OAuth tokens for a server (re-authorize)."""
        config = self._servers.get(name)
        if config is None:
            return f"MCP server '{name}' not found."

        rkey = f"mcptoken:{name}"
        if self._store.documents is None:
            raise RuntimeError("DocumentStore is not initialized")
        try:
            await self._store.documents.delete(rkey)
        except Exception:
            pass  # not found is fine
        return f"OAuth tokens cleared for '{name}'."

    # -- internals --

    def _create_oauth_provider(
        self, server_name: str, server_url: str, *, callback_timeout: float = 300.0
    ) -> Any:
        """Create an OAuthClientProvider for a server.

        Uses a paste-back flow instead of a local callback server so it works
        even when Victrola runs on a remote server. The redirect_handler stores
        the consent URL for the web UI. The callback_handler waits for the
        operator to paste the full redirect URL back via the web API.
        """
        from mcp.client.auth import OAuthClientProvider
        from mcp.shared.auth import OAuthClientMetadata

        storage = MCPOAuthTokenStorage(self._store, server_name)

        # Use a simple redirect URI — the page will show the code to paste
        client_metadata = OAuthClientMetadata(
            client_name="Victrola",
            redirect_uris=[AnyUrl("http://localhost:8989/callback")],  # type: ignore
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            scope=None,
        )

        # Per-server state for the paste-back flow
        if not hasattr(self, "_oauth_state"):
            self._oauth_state: dict[str, dict[str, Any]] = {}
        self._oauth_state[server_name] = {
            "consent_url": None,
            "pending_callback": None,
        }

        async def redirect_handler(url: str) -> None:
            self._oauth_state[server_name]["consent_url"] = url
            logger.info("OAuth consent URL for '%s': %s", server_name, url)

        async def callback_handler() -> tuple[str, str | None]:
            """Wait for the operator to paste the redirect URL via the web API."""
            # Create a future that will be resolved when the operator
            # submits the redirect URL via the /oauth/callback endpoint
            loop = asyncio.get_event_loop()
            future = loop.create_future()
            self._oauth_state[server_name]["pending_callback"] = future

            logger.info("Waiting for OAuth callback paste for '%s'...", server_name)

            try:
                # Wait up to callback_timeout for the operator to complete the flow
                callback_url = await asyncio.wait_for(future, timeout=callback_timeout)
            except asyncio.TimeoutError:
                self._oauth_state[server_name]["pending_callback"] = None
                raise RuntimeError(
                    f"OAuth callback timed out — no response within {callback_timeout}s"
                )

            self._oauth_state[server_name]["pending_callback"] = None

            # Parse the callback URL to extract code and state
            import urllib.parse

            parsed = urllib.parse.urlparse(callback_url)
            params = urllib.parse.parse_qs(parsed.query)

            if "error" in params:
                raise RuntimeError(f"OAuth authorization error: {params['error'][0]}")

            code = params.get("code", [None])[0]
            state = params.get("state", [None])[0]

            if not code:
                raise RuntimeError("OAuth callback URL did not contain an authorization code")

            return code, state

        provider = OAuthClientProvider(
            server_url=server_url,
            client_metadata=client_metadata,
            storage=storage,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
        )
        return provider

    def get_oauth_consent_url(self, name: str) -> str | None:
        """Return the OAuth consent URL if one was generated during connect."""
        state = getattr(self, "_oauth_state", {}).get(name, {})
        return state.get("consent_url")

    async def submit_oauth_callback(self, name: str, redirect_url: str) -> str:
        """Submit the OAuth redirect URL pasted by the operator.

        This resolves the pending callback handler and allows the OAuth flow
        to continue.
        """
        state = getattr(self, "_oauth_state", {}).get(name, {})
        future = state.get("pending_callback")
        if future is None or future.done():
            return f"No pending OAuth callback for '{name}'. Start a connection first."

        future.set_result(redirect_url)
        return f"OAuth callback submitted for '{name}'."

    def _register_tool(self, server_name: str, mcp_tool: MCPTool) -> None:
        """Register an MCP tool in the TOOL_REGISTRY."""
        sanitized = self._sanitize_tool_name(mcp_tool.name)
        full_name = f"{server_name}.{sanitized}"
        params, param_map = self._schema_to_parameters(mcp_tool.input_schema)
        manager = self
        original_tool_name = mcp_tool.name

        async def handler(ctx: Any, **kwargs: Any) -> Any:
            # Map sanitized param names back to the original schema keys
            original_params = {param_map.get(k, k): v for k, v in kwargs.items()}
            return await manager.call_tool(server_name, original_tool_name, original_params)

        self._registry.register(
            Tool(
                name=full_name,
                description=mcp_tool.description,
                parameters=params,
                handler=handler,
                source="mcp",
            )
        )

    @staticmethod
    def _sanitize_tool_name(name: str) -> str:
        """Replace non-identifier characters with underscore, ensure valid start, avoid reserved words."""
        sanitized = _SANITIZE_PATTERN.sub("_", name)
        if not sanitized:
            sanitized = "_"
        if not sanitized[0].isalpha() and sanitized[0] != "_":
            sanitized = "_" + sanitized
        if sanitized in _TS_RESERVED_WORDS:
            sanitized += "_"
        return sanitized

    @staticmethod
    def _schema_to_parameters(schema: dict[str, Any]) -> tuple[list[ToolParameter], dict[str, str]]:
        """Convert a JSON Schema to ToolParameter list + param name mapping.

        Returns (params, mapping) where mapping is sanitized_name -> original_name.
        Sanitized names are made unique with numeric suffixes if collisions occur.
        """
        if not isinstance(schema, dict):
            return [], {}

        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        type_map = {
            "string": "string",
            "number": "number",
            "integer": "number",
            "boolean": "boolean",
            "object": "object",
            "array": "array",
        }
        params: list[ToolParameter] = []
        param_map: dict[str, str] = {}
        used_names: set[str] = set()

        for prop_name, prop in props.items():
            if not isinstance(prop, dict):
                prop = {}
            sanitized_name = MCPManager._sanitize_tool_name(prop_name)
            # disambiguate if collision
            if sanitized_name in used_names:
                i = 2
                while f"{sanitized_name}_{i}" in used_names:
                    i += 1
                sanitized_name = f"{sanitized_name}_{i}"
            used_names.add(sanitized_name)
            param_map[sanitized_name] = prop_name

            # normalize type — handle union types, missing type, etc.
            raw_type = prop.get("type", "string")
            if isinstance(raw_type, list):
                raw_type = raw_type[0] if raw_type else "string"

            params.append(
                ToolParameter(
                    name=sanitized_name,
                    type=type_map.get(raw_type, "string"),
                    description=prop.get("description", ""),
                    required=prop_name in required,
                )
            )
        return params, param_map

    async def _persist_server(self, name: str) -> None:
        """Persist a server config to the DocumentStore."""
        config = self._servers.get(name)
        if config is None:
            return
        rkey = f"{MCP_RKEY_PREFIX}{name}"
        content = json.dumps(config.to_dict())
        if self._store.documents is None:
            raise RuntimeError("DocumentStore is not initialized")
        await self._store.documents.update(rkey, content)

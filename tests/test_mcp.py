"""Tests for MCP server integration."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.mcp import (
    MCPConnection,
    MCPManager,
    MCPServerConfig,
    MCPTool,
    MCP_RKEY_PREFIX,
)
from src.tools.registry import Tool, ToolParameter, ToolContext, ToolRegistry


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_unregister_removes_tool_and_returns_true():
    """unregister() should remove a tool and return True when it existed."""
    reg = ToolRegistry()

    async def handler(ctx, **kw):
        pass

    reg.register(Tool(name="test.tool", description="x", parameters=[], handler=handler))
    assert reg.unregister("test.tool") is True
    assert reg.get("test.tool") is None


def test_unregister_returns_false_for_missing():
    """unregister() should return False when the tool doesn't exist."""
    reg = ToolRegistry()
    assert reg.unregister("nonexistent") is False


# ---------------------------------------------------------------------------
# Dataclass serialization tests
# ---------------------------------------------------------------------------


def test_mcp_tool_serialization():
    """MCPTool round-trips through to_dict / from_dict."""
    tool = MCPTool(
        name="search_mail",
        description="Search mail",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        approved=True,
    )
    d = tool.to_dict()
    tool2 = MCPTool.from_dict(d)
    assert tool2.name == "search_mail"
    assert tool2.description == "Search mail"
    assert tool2.approved is True
    assert tool2.input_schema == tool.input_schema


def test_mcp_server_config_serialization():
    """MCPServerConfig round-trips through to_dict / from_dict."""
    cfg = MCPServerConfig(
        name="fastmail",
        transport="sse",
        url="https://api.fastmail.com/mcp",
        auth_token_secret="FASTMAIL_API_TOKEN",
        tools=[MCPTool(name="search", description="Search", input_schema={})],
    )
    d = cfg.to_dict()
    cfg2 = MCPServerConfig.from_dict(d)
    assert cfg2.name == "fastmail"
    assert cfg2.transport == "sse"
    assert cfg2.url == "https://api.fastmail.com/mcp"
    assert cfg2.auth_token_secret == "FASTMAIL_API_TOKEN"
    assert len(cfg2.tools) == 1
    assert cfg2.tools[0].name == "search"


def test_mcp_server_config_defaults():
    """MCPServerConfig uses field(default_factory) for list fields."""
    cfg = MCPServerConfig(name="test", transport="sse", url="https://example.com")
    assert cfg.args == []
    assert cfg.env_secrets == []
    assert cfg.tools == []
    assert cfg.enabled is True


# ---------------------------------------------------------------------------
# Schema conversion tests
# ---------------------------------------------------------------------------


def test_schema_to_parameters_simple_types():
    """_schema_to_parameters converts JSON Schema to ToolParameter list."""
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results"},
            "ratio": {"type": "number", "description": "Ratio"},
            "verbose": {"type": "boolean", "description": "Verbose"},
            "filter": {"type": "object", "description": "Filter obj"},
            "tags": {"type": "array", "description": "Tags"},
        },
        "required": ["query", "limit"],
    }
    params, param_map = MCPManager._schema_to_parameters(schema)
    assert len(params) == 6

    by_name = {p.name: p for p in params}
    assert by_name["query"].type == "string"
    assert by_name["query"].required is True
    assert by_name["limit"].type == "number"
    assert by_name["limit"].required is True
    assert by_name["ratio"].type == "number"
    assert by_name["ratio"].required is False
    assert by_name["verbose"].type == "boolean"
    assert by_name["filter"].type == "object"
    assert by_name["tags"].type == "array"
    # param_map should map sanitized -> original (all same here since no special chars)
    assert param_map["query"] == "query"


def test_schema_to_parameters_empty():
    """_schema_to_parameters returns ([], {}) for empty schema."""
    params, param_map = MCPManager._schema_to_parameters({})
    assert params == []
    assert param_map == {}
    params, param_map = MCPManager._schema_to_parameters({"type": "object"})
    assert params == []
    assert param_map == {}


def test_schema_to_parameters_unknown_type_defaults_string():
    """Unknown JSON Schema type falls back to 'string'."""
    schema = {
        "type": "object",
        "properties": {
            "custom": {"type": "weird-type", "description": "Unknown"},
        },
    }
    params, _ = MCPManager._schema_to_parameters(schema)
    assert params[0].type == "string"


def test_schema_to_parameters_sanitizes_non_identifier_names():
    """Property names with non-identifier chars are sanitized, with a mapping back to original."""
    schema = {
        "type": "object",
        "properties": {
            "foo-bar": {"type": "string", "description": "Hyphenated"},
            "x.y": {"type": "string", "description": "Dotted"},
        },
    }
    params, param_map = MCPManager._schema_to_parameters(schema)
    assert len(params) == 2
    names = {p.name for p in params}
    assert "foo_bar" in names
    assert "x_y" in names
    # mapping preserves original names for the MCP server call
    assert param_map["foo_bar"] == "foo-bar"
    assert param_map["x_y"] == "x.y"


# ---------------------------------------------------------------------------
# Tool name sanitization tests
# ---------------------------------------------------------------------------


def test_sanitize_tool_name():
    """_sanitize_tool_name replaces non-identifier chars with underscore, ensures valid start."""
    assert MCPManager._sanitize_tool_name("search_mail") == "search_mail"
    assert MCPManager._sanitize_tool_name("search.mail") == "search_mail"
    assert MCPManager._sanitize_tool_name("foo-bar") == "foo_bar"
    assert MCPManager._sanitize_tool_name("foo bar") == "foo_bar"
    # leading digit gets underscore prefix
    assert MCPManager._sanitize_tool_name("123abc") == "_123abc"


# ---------------------------------------------------------------------------
# MCPManager CRUD tests (with real Store)
# ---------------------------------------------------------------------------


async def _make_manager(tmp_path) -> MCPManager:
    """Create an MCPManager backed by a real SQLite store."""
    from src.store.store import Store

    store = Store(path=tmp_path / "store.db")
    await store.initialize()
    registry = ToolRegistry()
    return MCPManager(store=store, secret_manager=None, registry=registry), store


@pytest.mark.asyncio
async def test_create_server(tmp_path):
    """create_server stores config and makes it visible via list_servers."""
    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="testserver", transport="sse", url="https://example.com/mcp")
    result = await manager.create_server(config)
    assert "created" in result
    assert len(manager.list_servers()) == 1
    await store.close()


@pytest.mark.asyncio
async def test_create_server_rejects_reserved_name(tmp_path):
    """create_server rejects names that collide with built-in namespaces."""
    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="memory", transport="sse", url="https://example.com")
    result = await manager.create_server(config)
    assert "reserved" in result.lower()
    assert len(manager.list_servers()) == 0
    await store.close()


@pytest.mark.asyncio
async def test_create_server_rejects_invalid_name(tmp_path):
    """create_server rejects names that are invalid TS identifiers."""
    manager, store = await _make_manager(tmp_path)
    # hyphens are not valid TS identifier characters
    config = MCPServerConfig(name="bad-name", transport="sse", url="https://example.com")
    result = await manager.create_server(config)
    assert "invalid" in result.lower()
    # leading digits are not valid
    config2 = MCPServerConfig(name="123abc", transport="sse", url="https://example.com")
    result2 = await manager.create_server(config2)
    assert "invalid" in result2.lower()
    await store.close()


@pytest.mark.asyncio
async def test_create_server_rejects_duplicate(tmp_path):
    """create_server rejects duplicate server names."""
    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="dup", transport="sse", url="https://example.com")
    await manager.create_server(config)
    result = await manager.create_server(config)
    assert "already exists" in result
    await store.close()


@pytest.mark.asyncio
async def test_delete_server(tmp_path):
    """delete_server removes config from store and memory."""
    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="todelete", transport="sse", url="https://example.com")
    await manager.create_server(config)
    assert len(manager.list_servers()) == 1

    result = await manager.delete_server("todelete")
    assert "deleted" in result
    assert len(manager.list_servers()) == 0
    await store.close()


@pytest.mark.asyncio
async def test_load_servers(tmp_path):
    """load_servers reads configs from DocumentStore on startup."""
    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(
        name="loaded",
        transport="sse",
        url="https://example.com",
        tools=[MCPTool(name="tool1", description="d", input_schema={}, approved=True)],
    )
    await manager.create_server(config)

    # create a fresh manager and load from store
    registry2 = ToolRegistry()
    manager2 = MCPManager(store=store, secret_manager=None, registry=registry2)
    await manager2.load_servers()

    servers = manager2.list_servers()
    assert len(servers) == 1
    assert servers[0].name == "loaded"
    assert len(servers[0].tools) == 1
    assert servers[0].tools[0].approved is True
    await store.close()


# ---------------------------------------------------------------------------
# Approval / revocation tests (with mocked connection)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approve_tool_registers_in_registry(tmp_path):
    """approve_tool registers the tool in the TOOL_REGISTRY."""
    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="srv", transport="sse", url="https://example.com")
    config.tools = [
        MCPTool(
            name="search",
            description="Search stuff",
            input_schema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )
    ]
    await manager.create_server(config)

    result = await manager.approve_tool("srv", "search")
    assert "approved" in result

    # tool should be in registry
    tool = manager._registry.get("srv.search")
    assert tool is not None
    assert tool.description == "Search stuff"
    assert len(tool.parameters) == 1
    assert tool.parameters[0].name == "q"

    # tool should appear in generated TS types
    ts = manager._registry.generate_typescript_types()
    assert "srv" in ts
    assert "search" in ts

    # tool should appear in documentation
    docs = manager._registry.generate_tool_documentation()
    assert "srv.search" in docs
    await store.close()


@pytest.mark.asyncio
async def test_revoke_tool_unregisters_from_registry(tmp_path):
    """revoke_tool removes the tool from the TOOL_REGISTRY."""
    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="srv", transport="sse", url="https://example.com")
    config.tools = [
        MCPTool(name="search", description="Search", input_schema={})
    ]
    await manager.create_server(config)

    await manager.approve_tool("srv", "search")
    assert manager._registry.get("srv.search") is not None

    result = await manager.revoke_tool("srv", "search")
    assert "revoked" in result
    assert manager._registry.get("srv.search") is None
    await store.close()


@pytest.mark.asyncio
async def test_call_tool_not_connected(tmp_path):
    """call_tool returns error dict when server is not connected."""
    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="srv", transport="sse", url="https://example.com")
    await manager.create_server(config)

    result = await manager.call_tool("srv", "search", {"q": "test"})
    assert isinstance(result, dict)
    assert "error" in result
    assert "not connected" in result["error"]
    await store.close()


# ---------------------------------------------------------------------------
# Tool naming tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_naming_uses_server_dot_tool(tmp_path):
    """Approved MCP tools are registered as <server>.<tool> in the registry."""
    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="fastmail", transport="sse", url="https://example.com")
    config.tools = [
        MCPTool(name="search_mail", description="Search", input_schema={})
    ]
    await manager.create_server(config)

    await manager.approve_tool("fastmail", "search_mail")
    tool = manager._registry.get("fastmail.search_mail")
    assert tool is not None
    await store.close()


@pytest.mark.asyncio
async def test_tool_naming_sanitizes_dots(tmp_path):
    """Tool names with dots are sanitized for registry but original name preserved for calls."""
    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="srv", transport="sse", url="https://example.com")
    config.tools = [
        MCPTool(name="search.mail", description="Search", input_schema={})
    ]
    await manager.create_server(config)

    await manager.approve_tool("srv", "search.mail")
    # registered as sanitized name
    assert manager._registry.get("srv.search_mail") is not None
    # original name is NOT registered
    assert manager._registry.get("srv.search.mail") is None
    await store.close()


# ---------------------------------------------------------------------------
# ToolContext / ToolExecutor integration tests
# ---------------------------------------------------------------------------


def test_tool_context_has_mcp_manager_property():
    """ToolContext exposes mcp_manager property (None by default)."""
    ctx = ToolContext()
    assert ctx.mcp_manager is None


def test_executor_has_mcp_manager_property():
    """ToolExecutor exposes mcp_manager public property."""
    from src.tools.executor import ToolExecutor

    ctx = ToolContext()
    executor = ToolExecutor(registry=ToolRegistry(), ctx=ctx)
    assert hasattr(executor, "mcp_manager")
    assert executor.mcp_manager is None


# ---------------------------------------------------------------------------
# Web API tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_list_servers(tmp_path):
    """GET /api/mcp/servers returns summaries with correct counts."""
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from src.web.dependencies import get_executor
    from src.web.routers import mcp as mcp_router

    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="webtest", transport="sse", url="https://example.com")
    config.tools = [
        MCPTool(name="t1", description="d1", input_schema={}, approved=True),
        MCPTool(name="t2", description="d2", input_schema={}, approved=False),
    ]
    await manager.create_server(config)

    # mock executor
    executor = MagicMock()
    executor.mcp_manager = manager

    app = FastAPI()
    app.state.executor = executor
    app.dependency_overrides[get_executor] = lambda: executor
    app.include_router(mcp_router.router, prefix="/api")

    client = TestClient(app)
    resp = client.get("/api/mcp/servers")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "webtest"
    assert data[0]["transport"] == "sse"
    assert data[0]["tools_total"] == 2
    assert data[0]["tools_approved"] == 1
    await store.close()


@pytest.mark.asyncio
async def test_web_create_server(tmp_path):
    """POST /api/mcp/servers creates a server and returns summary."""
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from src.web.dependencies import get_executor
    from src.web.routers import mcp as mcp_router

    manager, store = await _make_manager(tmp_path)

    executor = MagicMock()
    executor.mcp_manager = manager

    app = FastAPI()
    app.state.executor = executor
    app.dependency_overrides[get_executor] = lambda: executor
    app.include_router(mcp_router.router, prefix="/api")

    client = TestClient(app)
    resp = client.post("/api/mcp/servers", json={
        "name": "apitest",
        "transport": "sse",
        "url": "https://example.com/mcp",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "apitest"
    assert data["connected"] is False
    await store.close()


@pytest.mark.asyncio
async def test_web_approve_tool(tmp_path):
    """POST /api/mcp/servers/{name}/tools/{tool}/approve registers the tool."""
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from src.web.dependencies import get_executor
    from src.web.routers import mcp as mcp_router

    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="webtest", transport="sse", url="https://example.com")
    config.tools = [
        MCPTool(name="tool1", description="d1", input_schema={})
    ]
    await manager.create_server(config)

    executor = MagicMock()
    executor.mcp_manager = manager

    app = FastAPI()
    app.state.executor = executor
    app.dependency_overrides[get_executor] = lambda: executor
    app.include_router(mcp_router.router, prefix="/api")

    client = TestClient(app)
    resp = client.post(
        "/api/mcp/servers/webtest/tools/approve",
        json={"tool_name": "tool1"},
    )
    assert resp.status_code == 200
    assert "approved" in resp.json()["message"]

    # verify tool was registered
    assert manager._registry.get("webtest.tool1") is not None
    await store.close()


# ---------------------------------------------------------------------------
# Security: TS injection prevention
# ---------------------------------------------------------------------------


def test_ts_injection_prevention_in_description():
    """Tool descriptions with */ must not break TS comment generation."""
    from src.tools.registry import TOOL_REGISTRY, Tool

    async def handler(ctx, **kw):
        pass

    # register a tool with a malicious description
    malicious_desc = '*/ }; evil(); export const x = { /*'
    TOOL_REGISTRY.register(
        Tool(
            name="inj.test",
            description=malicious_desc,
            parameters=[],
            handler=handler,
        )
    )

    ts = TOOL_REGISTRY.generate_typescript_types()
    # The */ must have been neutralized — no raw */ in the output
    assert "*/ };" not in ts
    # clean up
    TOOL_REGISTRY.unregister("inj.test")


def test_ts_injection_prevention_in_tool_name():
    """Tool names with quotes must not break the callTool string literal."""
    from src.tools.registry import TOOL_REGISTRY, Tool

    async def handler(ctx, **kw):
        pass

    TOOL_REGISTRY.register(
        Tool(
            name='weird";name',
            description="test",
            parameters=[],
            handler=handler,
        )
    )

    ts = TOOL_REGISTRY.generate_typescript_types()
    # The quote must have been escaped — no raw unescaped " inside the callTool string
    assert 'callTool("weird\\";name"' in ts
    TOOL_REGISTRY.unregister('weird";name')


@pytest.mark.asyncio
async def test_connect_failure_cleans_up_exit_stack(tmp_path):
    """If MCPConnection.connect() fails, the exit stack must be closed (no leak)."""
    from pathlib import Path
    from src.store.store import Store

    store = Store(path=tmp_path / "store.db")
    await store.initialize()
    registry = ToolRegistry()
    manager = MCPManager(store=store, secret_manager=None, registry=registry)

    config = MCPServerConfig(name="failtest", transport="sse", url="https://invalid.example.invalid/mcp")
    await manager.create_server(config)

    # Attempt to connect — should fail and clean up
    try:
        await manager.connect_server("failtest")
    except Exception:
        pass  # expected to fail

    # Connection should not be stored
    assert "failtest" not in manager._connections
    await store.close()


# ---------------------------------------------------------------------------
# Reserved word validation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_server_rejects_ts_reserved_word(tmp_path):
    """create_server rejects TypeScript reserved words as server names."""
    manager, store = await _make_manager(tmp_path)
    for bad_name in ["class", "default", "await", "function", "import"]:
        config = MCPServerConfig(name=bad_name, transport="sse", url="https://example.com")
        result = await manager.create_server(config)
        assert "reserved word" in result.lower(), f"Should reject '{bad_name}'"
    await store.close()


def test_sanitize_reserved_word_gets_suffix():
    """_sanitize_tool_name appends _ to reserved words."""
    assert MCPManager._sanitize_tool_name("class") == "class_"
    assert MCPManager._sanitize_tool_name("await") == "await_"
    assert MCPManager._sanitize_tool_name("function") == "function_"


# ---------------------------------------------------------------------------
# Param collision detection tests
# ---------------------------------------------------------------------------


def test_schema_param_collision_disambiguated():
    """Properties that sanitize to the same name get numeric suffixes."""
    schema = {
        "type": "object",
        "properties": {
            "foo-bar": {"type": "string"},
            "foo_bar": {"type": "string"},
            "foo.bar": {"type": "string"},
        },
    }
    params, param_map = MCPManager._schema_to_parameters(schema)
    names = [p.name for p in params]
    # all unique
    assert len(names) == len(set(names))
    # first one keeps base name, others get suffixes
    assert "foo_bar" in names
    assert "foo_bar_2" in names
    assert "foo_bar_3" in names
    # all map back to originals
    assert len(param_map) == 3
    originals = set(param_map.values())
    assert originals == {"foo-bar", "foo_bar", "foo.bar"}


# ---------------------------------------------------------------------------
# Union type / non-dict schema tests
# ---------------------------------------------------------------------------


def test_schema_union_type_handled():
    """Union types like ['string', 'null'] don't crash _schema_to_parameters."""
    schema = {
        "type": "object",
        "properties": {
            "optional_str": {"type": ["string", "null"], "description": "Maybe string"},
        },
    }
    params, _ = MCPManager._schema_to_parameters(schema)
    assert len(params) == 1
    assert params[0].type == "string"  # first type in union


def test_schema_missing_type_defaults_string():
    """Properties with no type field default to 'string'."""
    schema = {
        "type": "object",
        "properties": {
            "notype": {"description": "No type field"},
        },
    }
    params, _ = MCPManager._schema_to_parameters(schema)
    assert params[0].type == "string"


def test_schema_non_dict_property_handled():
    """Non-dict property schemas don't crash."""
    schema = {
        "type": "object",
        "properties": {
            "weird": True,  # boolean schema (JSON Schema allows this)
        },
    }
    params, _ = MCPManager._schema_to_parameters(schema)
    assert len(params) == 1
    assert params[0].type == "string"


# ---------------------------------------------------------------------------
# Markdown sanitization tests
# ---------------------------------------------------------------------------


def test_markdown_sanitization_strips_headings():
    """_sanitize_md strips leading # on every line and neutralizes code fences."""
    from src.tools.registry import ToolRegistry, Tool

    reg = ToolRegistry()

    async def handler(ctx, **kw):
        pass

    reg.register(
        Tool(
            name="md.test",
            description="## Fake Heading\n```evil```\nsafe text\n### Another heading",
            parameters=[],
            handler=handler,
        )
    )

    docs = reg.generate_tool_documentation()
    # Should not contain raw ## headings or ```
    assert "## Fake Heading" not in docs
    assert "### Another heading" not in docs
    assert "```" not in docs
    reg.unregister("md.test")


@pytest.mark.asyncio
async def test_connect_server_success_with_mocked_connection():
    """connect_server completes without deadlock when connect + discover succeed."""
    from pathlib import Path
    from src.store.store import Store
    from unittest.mock import AsyncMock, MagicMock, patch

    store = Store(path=Path("/tmp/test_mcp_connect_success.db"))
    await store.initialize()
    registry = ToolRegistry()
    manager = MCPManager(store=store, secret_manager=None, registry=registry)

    config = MCPServerConfig(name="mocksrv", transport="sse", url="https://example.com")
    await manager.create_server(config)

    # Mock MCPConnection so connect succeeds and list_tools returns a tool
    mock_tool = MagicMock()
    mock_tool.name = "search"
    mock_tool.description = "Search"
    mock_tool.inputSchema = {"type": "object", "properties": {"q": {"type": "string"}}}

    mock_conn = MagicMock()
    mock_conn.is_connected = True
    mock_conn.connect = AsyncMock()
    mock_conn.list_tools = AsyncMock(return_value=[mock_tool])
    mock_conn.disconnect = AsyncMock()

    with patch("src.tools.mcp.MCPConnection", return_value=mock_conn):
        await manager.connect_server("mocksrv")

    # connection should be stored
    assert "mocksrv" in manager._connections
    # tool should be discovered
    config = manager.get_server("mocksrv")
    assert len(config.tools) == 1
    assert config.tools[0].name == "search"
    assert config.tools[0].approved is False  # pending

    await manager.disconnect_server("mocksrv")
    await store.close()
    import os
    os.unlink("/tmp/test_mcp_connect_success.db")


@pytest.mark.asyncio
async def test_connect_server_discovery_failure_cleans_up():
    """connect_server cleans up connection if discover_tools fails after connect."""
    from pathlib import Path
    from src.store.store import Store
    from unittest.mock import AsyncMock, MagicMock, patch

    store = Store(path=Path("/tmp/test_mcp_discover_fail.db"))
    await store.initialize()
    registry = ToolRegistry()
    manager = MCPManager(store=store, secret_manager=None, registry=registry)

    config = MCPServerConfig(name="failsrv", transport="sse", url="https://example.com")
    await manager.create_server(config)

    mock_conn = MagicMock()
    mock_conn.is_connected = True
    mock_conn.connect = AsyncMock()
    mock_conn.list_tools = AsyncMock(side_effect=RuntimeError("discover failed"))
    mock_conn.disconnect = AsyncMock()

    with patch("src.tools.mcp.MCPConnection", return_value=mock_conn):
        try:
            await manager.connect_server("failsrv")
            assert False, "Should have raised"
        except RuntimeError:
            pass  # expected

    # connection should be cleaned up
    assert "failsrv" not in manager._connections
    await store.close()
    import os
    os.unlink("/tmp/test_mcp_discover_fail.db")


def test_schema_top_level_boolean():
    """A top-level boolean schema (valid JSON Schema) returns empty params."""
    params, param_map = MCPManager._schema_to_parameters(True)  # type: ignore
    assert params == []
    assert param_map == {}

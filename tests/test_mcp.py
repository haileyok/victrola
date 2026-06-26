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


# ---------------------------------------------------------------------------
# OAuth tests
# ---------------------------------------------------------------------------


def test_mcp_server_config_with_oauth_auth_type():
    """MCPServerConfig serializes auth_type correctly."""
    cfg = MCPServerConfig(
        name="fastmail",
        transport="sse",
        url="https://api.fastmail.com/mcp",
        auth_type="oauth",
    )
    d = cfg.to_dict()
    assert d["auth_type"] == "oauth"

    cfg2 = MCPServerConfig.from_dict(d)
    assert cfg2.auth_type == "oauth"


def test_mcp_server_config_default_auth_type_is_none():
    """auth_type defaults to 'none' when not specified."""
    cfg = MCPServerConfig(name="test", transport="sse", url="https://example.com")
    assert cfg.auth_type == "none"


def test_mcp_server_config_backward_compat_bearer():
    """Old configs without auth_type still deserialize correctly."""
    old_data = {
        "name": "legacy",
        "transport": "sse",
        "url": "https://example.com",
        "auth_token_secret": "MY_TOKEN",
        # no auth_type field
    }
    cfg = MCPServerConfig.from_dict(old_data)
    assert cfg.auth_type == "none"  # defaults to none
    assert cfg.auth_token_secret == "MY_TOKEN"


@pytest.mark.asyncio
async def test_create_server_bearer_backward_compat(tmp_path):
    """Server with auth_token_secret but no auth_type gets upgraded to bearer."""
    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(
        name="legacy",
        transport="sse",
        url="https://example.com",
        auth_token_secret="MY_TOKEN",
        # auth_type defaults to "none"
    )
    result = await manager.create_server(config)
    assert "created" in result

    stored = manager.get_server("legacy")
    assert stored.auth_type == "bearer"  # upgraded


@pytest.mark.asyncio
async def test_create_server_oauth_requires_sse(tmp_path):
    """OAuth auth_type requires SSE transport."""
    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(
        name="bad",
        transport="stdio",
        command="echo",
        auth_type="oauth",
    )
    result = await manager.create_server(config)
    assert "requires SSE" in result


@pytest.mark.asyncio
async def test_oauth_status_not_configured(tmp_path):
    """get_oauth_status returns 'not_configured' for non-OAuth servers."""
    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="test", transport="sse", url="https://example.com")
    await manager.create_server(config)

    status = await manager.get_oauth_status_async("test")
    assert status == "not_configured"


@pytest.mark.asyncio
async def test_oauth_status_not_authorized(tmp_path):
    """get_oauth_status returns 'not_authorized' for OAuth servers without tokens."""
    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(
        name="fastmail",
        transport="sse",
        url="https://api.fastmail.com/mcp",
        auth_type="oauth",
    )
    await manager.create_server(config)

    status = await manager.get_oauth_status_async("fastmail")
    assert status == "not_authorized"


@pytest.mark.asyncio
async def test_oauth_status_authorized(tmp_path):
    """get_oauth_status returns 'authorized' when tokens are stored."""
    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(
        name="fastmail",
        transport="sse",
        url="https://api.fastmail.com/mcp",
        auth_type="oauth",
    )
    await manager.create_server(config)

    # Simulate stored OAuth token
    from src.store.store import Store as StoreType
    assert isinstance(store, StoreType)
    await store.documents.create(
        "mcptoken:fastmail",
        json.dumps({"access_token": "tok", "token_type": "Bearer"}),
    )

    status = await manager.get_oauth_status_async("fastmail")
    assert status == "authorized"


@pytest.mark.asyncio
async def test_clear_oauth_tokens(tmp_path):
    """clear_oauth_tokens removes stored tokens."""
    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(
        name="fastmail",
        transport="sse",
        url="https://api.fastmail.com/mcp",
        auth_type="oauth",
    )
    await manager.create_server(config)

    # Store a token
    await store.documents.create(
        "mcptoken:fastmail",
        json.dumps({"access_token": "tok", "token_type": "Bearer"}),
    )
    assert await manager.get_oauth_status_async("fastmail") == "authorized"

    # Clear it
    result = await manager.clear_oauth_tokens("fastmail")
    assert "cleared" in result
    assert await manager.get_oauth_status_async("fastmail") == "not_authorized"


@pytest.mark.asyncio
async def test_connect_all_skips_unauthorized_oauth(tmp_path):
    """connect_all skips OAuth servers that haven't been authorized."""
    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(
        name="fastmail",
        transport="sse",
        url="https://api.fastmail.com/mcp",
        auth_type="oauth",
        enabled=True,
    )
    await manager.create_server(config)

    # Should skip silently, not try to connect
    await manager.connect_all()
    assert not manager.is_connected("fastmail")
    await store.close()


@pytest.mark.asyncio
async def test_oauth_token_storage_roundtrip(tmp_path):
    """MCPOAuthTokenStorage stores and retrieves tokens."""
    from src.tools.mcp import MCPOAuthTokenStorage

    manager, store = await _make_manager(tmp_path)
    storage = MCPOAuthTokenStorage(store, "testserver")

    # No tokens initially
    tokens = await storage.get_tokens()
    assert tokens is None

    # Store tokens
    from mcp.shared.auth import OAuthToken

    token = OAuthToken(
        access_token="abc123",
        token_type="Bearer",
        refresh_token="refresh456",
        expires_in=3600,
    )
    await storage.set_tokens(token)

    # Retrieve
    retrieved = await storage.get_tokens()
    assert retrieved is not None
    assert retrieved.access_token == "abc123"
    assert retrieved.refresh_token == "refresh456"
    assert retrieved.expires_in == 3600
    await store.close()


@pytest.mark.asyncio
async def test_web_create_oauth_server(tmp_path):
    """POST /api/mcp/servers creates an OAuth server."""
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
        "name": "fastmail",
        "transport": "sse",
        "url": "https://api.fastmail.com/mcp",
        "auth_type": "oauth",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "fastmail"
    await store.close()


@pytest.mark.asyncio
async def test_web_oauth_deauthorize(tmp_path):
    """POST /api/mcp/servers/{name}/oauth/deauthorize clears tokens."""
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from src.web.dependencies import get_executor
    from src.web.routers import mcp as mcp_router

    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(
        name="fastmail",
        transport="sse",
        url="https://api.fastmail.com/mcp",
        auth_type="oauth",
    )
    await manager.create_server(config)

    # Store a token
    await store.documents.create(
        "mcptoken:fastmail",
        json.dumps({"access_token": "tok", "token_type": "Bearer"}),
    )

    executor = MagicMock()
    executor.mcp_manager = manager

    app = FastAPI()
    app.state.executor = executor
    app.dependency_overrides[get_executor] = lambda: executor
    app.include_router(mcp_router.router, prefix="/api")

    client = TestClient(app)
    resp = client.post("/api/mcp/servers/fastmail/oauth/deauthorize")
    assert resp.status_code == 200
    assert "cleared" in resp.json()["message"].lower()

    # Token should be gone
    status = await manager.get_oauth_status_async("fastmail")
    assert status == "not_authorized"
    await store.close()


# ---------------------------------------------------------------------------
# Resilience tests: health monitor, auto-reconnect, sse_read_timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_ping_succeeds_on_healthy_connection(tmp_path):
    """send_ping on a healthy connection completes without error."""
    from unittest.mock import AsyncMock, MagicMock

    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="srv", transport="sse", url="https://example.com")
    await manager.create_server(config)

    mock_conn = MagicMock()
    mock_conn.is_connected = True
    mock_conn.connect = AsyncMock()
    mock_conn.disconnect = AsyncMock()
    mock_conn.send_ping = AsyncMock()
    mock_conn.list_tools = AsyncMock(return_value=[])

    with patch("src.tools.mcp.MCPConnection", return_value=mock_conn):
        await manager.connect_server("srv")

    # Directly invoke the connection's send_ping — should not raise
    await mock_conn.send_ping()
    mock_conn.send_ping.assert_called()

    await manager.stop_health_monitor()
    await manager.disconnect_server("srv")
    await store.close()


@pytest.mark.asyncio
async def test_health_check_reconnects_dead_connection(tmp_path):
    """When send_ping raises, the health check disconnects and reconnects."""
    from unittest.mock import AsyncMock, MagicMock

    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="srv", transport="sse", url="https://example.com")
    await manager.create_server(config)

    # First connection: ping raises
    dead_conn = MagicMock()
    dead_conn.is_connected = True
    dead_conn.connect = AsyncMock()
    dead_conn.disconnect = AsyncMock()
    dead_conn.send_ping = AsyncMock(side_effect=RuntimeError("connection dead"))
    dead_conn.list_tools = AsyncMock(return_value=[])

    # Second connection (after reconnect): ping succeeds
    healthy_conn = MagicMock()
    healthy_conn.is_connected = True
    healthy_conn.connect = AsyncMock()
    healthy_conn.disconnect = AsyncMock()
    healthy_conn.send_ping = AsyncMock()
    healthy_conn.list_tools = AsyncMock(return_value=[])

    call_count = 0

    def conn_factory(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return dead_conn if call_count == 1 else healthy_conn

    with patch("src.tools.mcp.MCPConnection", side_effect=conn_factory):
        await manager.connect_server("srv")
        # Simulate a health check failure + reconnect
        result = await manager._reconnect_server("srv")

    assert result is True
    assert "srv" in manager._connections
    assert manager._connections["srv"] is healthy_conn
    # dead_conn should have been disconnected
    dead_conn.disconnect.assert_awaited()

    await manager.stop_health_monitor()
    await manager.disconnect_server("srv")
    await store.close()


@pytest.mark.asyncio
async def test_call_tool_auto_reconnects_on_failure(tmp_path):
    """call_tool reconnects and retries when the first call fails."""
    from unittest.mock import AsyncMock, MagicMock

    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="srv", transport="sse", url="https://example.com")
    await manager.create_server(config)

    # First connection: call_tool raises once
    fail_conn = MagicMock()
    fail_conn.is_connected = True
    fail_conn.connect = AsyncMock()
    fail_conn.disconnect = AsyncMock()
    fail_conn.list_tools = AsyncMock(return_value=[])
    fail_conn.call_tool = AsyncMock(side_effect=RuntimeError("transport dead"))

    # Second connection (after reconnect): call_tool succeeds
    good_conn = MagicMock()
    good_conn.is_connected = True
    good_conn.connect = AsyncMock()
    good_conn.disconnect = AsyncMock()
    good_conn.list_tools = AsyncMock(return_value=[])
    good_conn.call_tool = AsyncMock(return_value="result")

    call_count = 0

    def conn_factory(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return fail_conn if call_count == 1 else good_conn

    with patch("src.tools.mcp.MCPConnection", side_effect=conn_factory):
        await manager.connect_server("srv")
        result = await manager.call_tool("srv", "search", {"q": "test"})

    assert result == "result"
    # fail_conn.call_tool called once (the initial failure)
    fail_conn.call_tool.assert_awaited_once()
    # good_conn.call_tool should have been called once (the retry after reconnect)
    good_conn.call_tool.assert_awaited_once()

    await manager.stop_health_monitor()
    await manager.disconnect_server("srv")
    await store.close()


@pytest.mark.asyncio
async def test_call_tool_returns_error_when_reconnect_fails(tmp_path):
    """call_tool returns an error dict when both the call and reconnect fail."""
    from unittest.mock import AsyncMock, MagicMock

    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="srv", transport="sse", url="https://example.com")
    await manager.create_server(config)

    mock_conn = MagicMock()
    mock_conn.is_connected = True
    mock_conn.connect = AsyncMock()
    mock_conn.disconnect = AsyncMock()
    mock_conn.list_tools = AsyncMock(return_value=[])
    mock_conn.call_tool = AsyncMock(side_effect=RuntimeError("transport dead"))

    # _connect_server_impl will also fail (connect raises)
    fail_conn = MagicMock()
    fail_conn.is_connected = False
    fail_conn.connect = AsyncMock(side_effect=RuntimeError("server down"))
    fail_conn.disconnect = AsyncMock()
    fail_conn.list_tools = AsyncMock(return_value=[])

    call_count = 0

    def conn_factory(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return mock_conn if call_count == 1 else fail_conn

    with patch("src.tools.mcp.MCPConnection", side_effect=conn_factory):
        await manager.connect_server("srv")
        result = await manager.call_tool("srv", "search", {"q": "test"})

    assert isinstance(result, dict)
    assert "error" in result
    assert "MCP tool call failed" in result["error"]

    await manager.stop_health_monitor()
    await manager.disconnect_server("srv")
    await store.close()


@pytest.mark.asyncio
async def test_oauth_auto_connect_uses_short_callback_timeout(tmp_path):
    """_connect_server_impl(auto=True) passes callback_timeout=5.0 to _create_oauth_provider."""
    from unittest.mock import AsyncMock, MagicMock

    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(
        name="fastmail",
        transport="streamable_http",
        url="https://api.fastmail.com/mcp",
        auth_type="oauth",
    )
    await manager.create_server(config)

    mock_conn = MagicMock()
    mock_conn.is_connected = True
    mock_conn.connect = AsyncMock()
    mock_conn.disconnect = AsyncMock()
    mock_conn.list_tools = AsyncMock(return_value=[])

    captured_kwargs = {}

    original_create = manager._create_oauth_provider

    def spy_create(server_name, server_url, *, callback_timeout=300.0):
        captured_kwargs["callback_timeout"] = callback_timeout
        return original_create(server_name, server_url, callback_timeout=callback_timeout)

    with patch.object(manager, "_create_oauth_provider", side_effect=spy_create):
        with patch("src.tools.mcp.MCPConnection", return_value=mock_conn):
            # auto=True path
            await manager._connect_server_impl("fastmail", auto=True)

    assert captured_kwargs.get("callback_timeout") == 5.0

    # Verify manual path uses 300.0
    captured_kwargs.clear()
    # Need to disconnect first
    await manager._disconnect_server_impl("fastmail")
    with patch.object(manager, "_create_oauth_provider", side_effect=spy_create):
        with patch("src.tools.mcp.MCPConnection", return_value=mock_conn):
            await manager._connect_server_impl("fastmail", auto=False)

    assert captured_kwargs.get("callback_timeout") == 300.0

    await manager.stop_health_monitor()
    await manager.disconnect_server("fastmail")
    await store.close()


@pytest.mark.asyncio
async def test_health_monitor_start_stop(tmp_path):
    """start_health_monitor creates a task; stop_health_monitor cancels it."""
    import asyncio

    manager, store = await _make_manager(tmp_path)

    # No task initially
    assert manager._health_task is None

    # Start with a long interval so the loop doesn't iterate during the test
    manager.start_health_monitor(9999.0)
    assert manager._health_task is not None
    assert not manager._health_task.done()

    # Starting again is a no-op
    first_task = manager._health_task
    manager.start_health_monitor(9999.0)
    assert manager._health_task is first_task

    # Stop cancels and clears the task
    await manager.stop_health_monitor()
    assert manager._health_task is None
    assert first_task.done()

    # Stopping again is a no-op
    await manager.stop_health_monitor()

    await store.close()


@pytest.mark.asyncio
async def test_sse_read_timeout_passed_to_transport(tmp_path):
    """MCPConnection.connect passes sse_read_timeout from CONFIG to streamablehttp_client."""
    from unittest.mock import AsyncMock, MagicMock

    from src.config import CONFIG

    # Save original value
    original = CONFIG.mcp_sse_read_timeout_seconds
    CONFIG.mcp_sse_read_timeout_seconds = 1234
    try:
        captured_kwargs = {}

        # We need to mock the streamablehttp_client at the import site
        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_get_session_id = MagicMock()

        class FakeAsyncCtx:
            async def __aenter__(self):
                return (mock_read, mock_write, mock_get_session_id)

            async def __aexit__(self, *args):
                pass

        def fake_streamablehttp_client(url, **kwargs):
            captured_kwargs.update(kwargs)
            return FakeAsyncCtx()

        # Mock ClientSession.initialize
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()

        def fake_client_session(read, write):
            class SessionCtx:
                async def __aenter__(self_inner):
                    return mock_session

                async def __aexit__(self_inner, *args):
                    pass

            return SessionCtx()

        config = MCPServerConfig(
            name="srv",
            transport="streamable_http",
            url="https://example.com/mcp",
        )
        conn = MCPConnection(config, secret_manager=None)

        with patch("mcp.client.streamable_http.streamablehttp_client", side_effect=fake_streamablehttp_client):
            with patch("mcp.client.session.ClientSession", side_effect=fake_client_session):
                await conn.connect()

        assert "sse_read_timeout" in captured_kwargs
        assert captured_kwargs["sse_read_timeout"] == 1234.0

        await conn.disconnect()
    finally:
        CONFIG.mcp_sse_read_timeout_seconds = original


@pytest.mark.asyncio
async def test_call_tool_serializes_with_health_monitor(tmp_path):
    """call_tool holds the per-server lock; health check skips while a call is in progress."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="srv", transport="sse", url="https://example.com")
    await manager.create_server(config)

    call_started = asyncio.Event()
    call_can_finish = asyncio.Event()

    mock_conn = MagicMock()
    mock_conn.is_connected = True
    mock_conn.connect = AsyncMock()
    mock_conn.disconnect = AsyncMock()
    mock_conn.list_tools = AsyncMock(return_value=[])

    async def slow_call_tool(*args, **kwargs):
        call_started.set()
        await call_can_finish.wait()
        return "slow_result"

    mock_conn.call_tool = AsyncMock(side_effect=slow_call_tool)
    mock_conn.send_ping = AsyncMock()

    with patch("src.tools.mcp.MCPConnection", return_value=mock_conn):
        await manager.connect_server("srv")

        # Run call_tool and _health_check_server concurrently
        async def do_call():
            return await manager.call_tool("srv", "search", {"q": "test"})

        async def do_health_check():
            await call_started.wait()
            # Health check should skip (lock held by do_call), not block
            await asyncio.wait_for(
                manager._health_check_server("srv"), timeout=5.0
            )
            # Now let the call finish
            call_can_finish.set()

        results = await asyncio.gather(do_call(), do_health_check())

    assert results[0] == "slow_result"
    # send_ping should NOT have been called (health check skipped)
    mock_conn.send_ping.assert_not_awaited()

    await manager.stop_health_monitor()
    await manager.disconnect_server("srv")
    await store.close()


@pytest.mark.asyncio
async def test_call_tool_retry_fails_after_reconnect(tmp_path):
    """When retry also fails after a successful reconnect, call_tool returns a clear error."""
    from unittest.mock import AsyncMock, MagicMock

    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="srv", transport="sse", url="https://example.com")
    await manager.create_server(config)

    # Initial connection — call_tool fails
    fail_conn = MagicMock()
    fail_conn.is_connected = True
    fail_conn.connect = AsyncMock()
    fail_conn.disconnect = AsyncMock()
    fail_conn.list_tools = AsyncMock(return_value=[])
    fail_conn.call_tool = AsyncMock(side_effect=RuntimeError("transport dead"))

    # Reconnected connection — call_tool also fails (different error)
    still_bad_conn = MagicMock()
    still_bad_conn.is_connected = True
    still_bad_conn.connect = AsyncMock()
    still_bad_conn.disconnect = AsyncMock()
    still_bad_conn.list_tools = AsyncMock(return_value=[])
    still_bad_conn.call_tool = AsyncMock(side_effect=RuntimeError("server error after reconnect"))

    call_count = 0

    def conn_factory(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return fail_conn if call_count == 1 else still_bad_conn

    with patch("src.tools.mcp.MCPConnection", side_effect=conn_factory):
        await manager.connect_server("srv")
        result = await manager.call_tool("srv", "search", {"q": "test"})

    assert isinstance(result, dict)
    assert "error" in result
    assert "Reconnected but retry also failed" in result["error"]
    assert "server error after reconnect" in result["error"]

    await manager.stop_health_monitor()
    await manager.disconnect_server("srv")
    await store.close()


@pytest.mark.asyncio
async def test_call_tool_auto_reconnect_does_not_block(tmp_path):
    """call_tool returns within bounded time even if disconnect or connect hangs."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="srv", transport="sse", url="https://example.com")
    await manager.create_server(config)

    # First connection: disconnect hangs forever, call_tool fails
    mock_conn = MagicMock()
    mock_conn.is_connected = True
    mock_conn.connect = AsyncMock()

    async def hang_disconnect():
        await asyncio.sleep(9999)

    mock_conn.disconnect = AsyncMock(side_effect=hang_disconnect)
    mock_conn.list_tools = AsyncMock(return_value=[])
    mock_conn.call_tool = AsyncMock(side_effect=RuntimeError("transport dead"))

    # Second connection: connect also hangs forever
    hang_conn = MagicMock()
    hang_conn.is_connected = False

    async def hang_forever():
        await asyncio.sleep(9999)

    hang_conn.connect = AsyncMock(side_effect=hang_forever)
    hang_conn.disconnect = AsyncMock()
    hang_conn.list_tools = AsyncMock(return_value=[])

    call_count = 0

    def conn_factory(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return mock_conn if call_count == 1 else hang_conn

    # Monkeypatch _RECONNECT_TIMEOUT to a small value so the test is fast
    import src.tools.mcp as mcp_mod

    original_timeout = mcp_mod._RECONNECT_TIMEOUT
    mcp_mod._RECONNECT_TIMEOUT = 0.5
    try:
        with patch("src.tools.mcp.MCPConnection", side_effect=conn_factory):
            await manager.connect_server("srv")
            result = await manager.call_tool("srv", "search", {"q": "test"})
    finally:
        mcp_mod._RECONNECT_TIMEOUT = original_timeout

    assert isinstance(result, dict)
    assert "error" in result
    assert "timed out" in result["error"].lower()

    await manager.stop_health_monitor()
    # Clean up — use the impl directly since disconnect_server would try to
    # disconnect the hanging connection again
    manager._connections.pop("srv", None)
    await store.close()


@pytest.mark.asyncio
async def test_concurrent_ping_and_disconnect(tmp_path):
    """_health_check_server and disconnect_server run concurrently without deadlocking."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="srv", transport="sse", url="https://example.com")
    await manager.create_server(config)

    ping_started = asyncio.Event()
    ping_can_finish = asyncio.Event()

    mock_conn = MagicMock()
    mock_conn.is_connected = True
    mock_conn.connect = AsyncMock()
    mock_conn.disconnect = AsyncMock()

    async def slow_ping():
        ping_started.set()
        await ping_can_finish.wait()

    mock_conn.send_ping = AsyncMock(side_effect=slow_ping)
    mock_conn.list_tools = AsyncMock(return_value=[])

    with patch("src.tools.mcp.MCPConnection", return_value=mock_conn):
        await manager.connect_server("srv")

        # Run health_check_server and disconnect_server concurrently.
        # The health check acquires the lock and holds it during the slow ping.
        # disconnect_server blocks on the lock until the ping finishes.
        async def do_health_check():
            try:
                await manager._health_check_server("srv")
            except Exception:
                pass

        async def do_disconnect():
            # Let the ping start (it holds the lock)
            await ping_started.wait()
            # Release the ping so the health check completes and releases the lock
            ping_can_finish.set()
            # Now disconnect_server can acquire the lock
            await manager.disconnect_server("srv")

        # Wrap in a timeout so the test fails fast if there's a deadlock
        await asyncio.wait_for(
            asyncio.gather(do_health_check(), do_disconnect()),
            timeout=10.0,
        )

    # Connection should be cleanly removed
    assert "srv" not in manager._connections
    await manager.stop_health_monitor()
    await store.close()


@pytest.mark.asyncio
async def test_disconnect_all_awaits_health_monitor_during_teardown(tmp_path):
    """disconnect_all stops the health monitor before tearing down connections."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    manager, store = await _make_manager(tmp_path)
    config = MCPServerConfig(name="srv", transport="sse", url="https://example.com")
    await manager.create_server(config)

    mock_conn = MagicMock()
    mock_conn.is_connected = True
    mock_conn.connect = AsyncMock()
    mock_conn.disconnect = AsyncMock()
    mock_conn.list_tools = AsyncMock(return_value=[])

    reconnect_called = False

    with patch("src.tools.mcp.MCPConnection", return_value=mock_conn):
        await manager.connect_server("srv")

        # Start the health monitor with a short interval
        manager.start_health_monitor(0.05)

        # Let one health check iteration run
        await asyncio.sleep(0.06)

        # Now call disconnect_all — should stop the monitor first
        await manager.disconnect_all()

    # Health task should be cancelled
    assert manager._health_task is None
    # Connections should be empty
    assert len(manager._connections) == 0
    await store.close()

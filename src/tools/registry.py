from collections.abc import Awaitable, Callable
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict

from src.store.store import Store


class ToolParameter(BaseModel):
    name: str
    type: Literal["string", "number", "boolean", "object", "array"]
    description: str
    required: bool = True
    default: Any = None


class Tool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    parameters: list[ToolParameter]
    handler: Callable[..., Awaitable[Any]]  # async function
    source: Literal["builtin", "mcp"] = "builtin"


class ToolContext:
    """a context that has access to various backend services that are available to deno sandboxed tools"""

    def __init__(
        self,
        exa_client: Any | None = None,
        llm_client: Any | None = None,
        http_client: Any | None = None,
        store: Store | None = None,
    ) -> None:
        self._exa_client = exa_client
        self._llm_client = llm_client
        self._http_client = http_client
        self._store = store
        self._custom_tool_manager: Any | None = None
        self._mcp_manager: Any | None = None
        self._secret_manager: Any | None = None
        self._scheduler: Any | None = None
        self._embedding_client: Any | None = None
        self._search_engine: Any | None = None

    @property
    def exa_client(self) -> Any:
        if self._exa_client is None:
            raise RuntimeError("Exa client not configured")
        return self._exa_client

    @property
    def llm_client(self) -> Any:
        if self._llm_client is None:
            raise RuntimeError("LLM client not configured")
        return self._llm_client

    @property
    def http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            raise RuntimeError("HTTP client not configured")
        return self._http_client

    @property
    def store(self) -> Store:
        if self._store is None:
            raise RuntimeError("Store not configured")
        return self._store

    @property
    def custom_tool_manager(self) -> Any:
        return self._custom_tool_manager

    @property
    def mcp_manager(self) -> Any:
        return self._mcp_manager

    @property
    def secret_manager(self) -> Any:
        return self._secret_manager

    @property
    def scheduler(self) -> Any:
        return self._scheduler

    @property
    def embedding_client(self) -> Any:
        if self._embedding_client is None:
            raise RuntimeError("EmbeddingClient not configured")
        return self._embedding_client

    @property
    def search_engine(self) -> Any:
        if self._search_engine is None:
            raise RuntimeError("SearchEngine not configured")
        return self._search_engine


class ToolRegistry:
    """a registry of all the available tools"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry. Returns True if it existed."""
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def tool(
        self,
        name: str,
        description: str,
        parameters: list[ToolParameter] | None = None,
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """the main tool decorator for tools that you create"""

        def decorator(
            func: Callable[..., Awaitable[Any]],
        ) -> Callable[..., Awaitable[Any]]:
            t = Tool(
                name=name,
                description=description,
                parameters=parameters or [],
                handler=func,
            )
            self.register(t)
            return func

        return decorator

    async def execute(self, ctx: ToolContext, name: str, params: dict[str, Any]) -> Any:
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")

        if len(params) == 1:
            param_names = {p.name for p in tool.parameters}
            val = next(iter(params.values()))
            if isinstance(val, dict) and set(val.keys()) <= param_names:  # ignore: type
                params = val  # type: ignore

        return await tool.handler(ctx, **params)

    def generate_tool_documentation(self) -> str:
        """Generate markdown tool documentation for the system prompt.

        Emits full per-tool descriptions and typed parameter lists, grouped by
        namespace. The enclosing `TOOL_DOCS_TEMPLATE` in prompt.py supplies the
        top-level heading, so this starts at `## <namespace>`.
        """
        by_namespace: dict[str, list[Tool]] = {}
        for tool in self._tools.values():
            namespace = tool.name.split(".")[0]
            by_namespace.setdefault(namespace, []).append(tool)

        lines: list[str] = []
        for namespace, tools in sorted(by_namespace.items()):
            lines.append(f"## {namespace}\n")
            for tool in sorted(tools, key=lambda t: t.name):
                lines.append(f"### {tool.name}")
                lines.append(f"{self._sanitize_md(tool.description)}\n")
                if tool.parameters:
                    lines.append("**Parameters:**")
                    for param in tool.parameters:
                        req = "" if param.required else " (optional)"
                        default = (
                            f", default: {param.default}"
                            if param.default is not None
                            else ""
                        )
                        lines.append(
                            f"- `{param.name}` ({param.type}{req}{default}): {self._sanitize_md(param.description)}"
                        )
                    lines.append("")

        return "\n".join(lines)

    def generate_builtin_tool_documentation(self) -> str:
        """Generate full markdown docs for built-in tools only.

        Same format as generate_tool_documentation() but filtered to
        source == "builtin". MCP tools are excluded — they get a compact
        catalog via generate_mcp_tool_catalog().
        """
        by_namespace: dict[str, list[Tool]] = {}
        for tool in self._tools.values():
            if tool.source != "builtin":
                continue
            namespace = tool.name.split(".")[0]
            by_namespace.setdefault(namespace, []).append(tool)

        lines: list[str] = []
        for namespace, tools in sorted(by_namespace.items()):
            lines.append(f"## {namespace}\n")
            for tool in sorted(tools, key=lambda t: t.name):
                lines.append(f"### {tool.name}")
                lines.append(f"{self._sanitize_md(tool.description)}\n")
                if tool.parameters:
                    lines.append("**Parameters:**")
                    for param in tool.parameters:
                        req = "" if param.required else " (optional)"
                        default = (
                            f", default: {param.default}"
                            if param.default is not None
                            else ""
                        )
                        lines.append(
                            f"- `{param.name}` ({param.type}{req}{default}): {self._sanitize_md(param.description)}"
                        )
                    lines.append("")

        return "\n".join(lines)

    def generate_mcp_tool_catalog(self) -> str:
        """Generate a compact one-line-per-tool catalog for MCP tools.

        Format: `server.method — short description` grouped by server.
        Descriptions are sanitized and truncated to the first line (~100 chars).
        Returns an empty string when no MCP tools are registered.
        """
        mcp_tools = [t for t in self._tools.values() if t.source == "mcp"]
        if not mcp_tools:
            return ""

        by_server: dict[str, list[Tool]] = {}
        for tool in mcp_tools:
            server = tool.name.split(".")[0]
            by_server.setdefault(server, []).append(tool)

        lines: list[str] = []
        for server, tools in sorted(by_server.items()):
            lines.append(f"## {server}")
            for tool in sorted(tools, key=lambda t: t.name):
                desc = tool.description
                if not desc:
                    short = "(no description provided)"
                else:
                    sanitized = self._sanitize_md(desc)
                    short = self._truncate_description(sanitized)
                    if not short:
                        short = "(no description provided)"
                lines.append(f"- `{tool.name}` — {short}")
            lines.append("")

        return "\n".join(lines).rstrip()

    @staticmethod
    def _truncate_description(text: str, max_chars: int = 100) -> str:
        """Truncate to the first line, then to max_chars with an ellipsis.

        Also strips leading markdown list markers (-, *, +) and blockquote
        prefixes (>) so they don't render as a double dash in the catalog
        bullet list.
        """
        first_line = text.split("\n", 1)[0].strip()
        # strip leading markdown list markers / blockquote prefixes
        for prefix in ("- ", "* ", "+ ", "> "):
            if first_line.startswith(prefix):
                first_line = first_line[len(prefix):].lstrip()
                break
        if len(first_line) <= max_chars:
            return first_line
        return first_line[:max_chars].rstrip() + "…"

    def generate_typescript_types(self) -> str:
        lines = [
            "// Auto-generated - do not edit",
            'import { callTool } from "./runtime.ts";',
            "",
        ]

        by_namespace: dict[str, list[Tool]] = {}
        for tool in self._tools.values():
            namespace = tool.name.split(".")[0]
            by_namespace.setdefault(namespace, []).append(tool)

        for namespace, tools in sorted(by_namespace.items()):
            lines.append(f"export const {namespace} = {{")
            for i, tool in enumerate(sorted(tools, key=lambda t: t.name)):
                method_name = (
                    tool.name.split(".", 1)[1] if "." in tool.name else tool.name
                )

                required_params = [p for p in tool.parameters if p.required]
                optional_params = [p for p in tool.parameters if not p.required]
                ordered_params = required_params + optional_params

                params: list[str] = []
                for param in ordered_params:
                    ts_type = self._python_type_to_ts(param.type)
                    if param.required:
                        params.append(f"{param.name}: {ts_type}")
                    else:
                        params.append(f"{param.name}?: {ts_type}")

                param_str = ", ".join(params)

                param_names = [p.name for p in tool.parameters]
                params_obj = (
                    "{ " + ", ".join(param_names) + " }" if param_names else "{}"
                )

                # Sanitize description for TS comment — strip */ to prevent
                # comment termination and code injection from untrusted input
                safe_desc = tool.description.replace("*/", "* /")
                lines.append(f"  /** {safe_desc} */")
                safe_name = tool.name.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(
                    f'  {method_name}: ({param_str}): Promise<unknown> => callTool("{safe_name}", {params_obj}),'
                )
                if i < len(tools) - 1:
                    lines.append("")

            lines.append("};")
            lines.append("")

        return "\n".join(lines)

    def _python_type_to_ts(self, py_type: str) -> str:
        mapping = {
            "string": "string",
            "number": "number",
            "boolean": "boolean",
            "object": "Record<string, unknown>",
            "array": "unknown[]",
        }
        return mapping.get(py_type, "unknown")

    def _default_to_ts(self, value: Any) -> str:
        if value is None:
            return "undefined"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return f'"{value}"'
        return str(value)

    @staticmethod
    def _sanitize_md(text: str) -> str:
        """Strip markdown control characters from untrusted text.

        Prevents heading injection and code fence injection from MCP
        tool descriptions in the system prompt docs.
        """
        lines = text.split("\n")
        sanitized_lines = []
        for line in lines:
            # strip leading # markers that could inject fake headings
            stripped = line.lstrip("#")
            sanitized_lines.append(stripped)
        result = "\n".join(sanitized_lines)
        # neutralize backtick-fenced code blocks
        return result.replace("```", "\\`\\`\\`")


TOOL_REGISTRY = ToolRegistry()

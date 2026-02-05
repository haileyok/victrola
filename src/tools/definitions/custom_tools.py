import json
import logging
from typing import Any

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter

logger = logging.getLogger(__name__)


@TOOL_REGISTRY.tool(
    name="custom_tools.list_available_secrets",
    description="List the names of secrets that have been configured by the human operator. These can be referenced in the `secrets` field when creating custom tools. Secret values are never exposed — only names are returned.",
    parameters=[],
)
async def list_available_secrets(ctx: ToolContext) -> str:
    sm = ctx.secret_manager
    if sm is None:
        return "Error: Secret manager not available."

    names = sm.list_secret_names()
    if not names:
        return "No secrets configured. Ask the human operator to add secrets via the TUI (press 's' from the session list)."

    lines = [f"Available secrets ({len(names)}):\n"]
    for name in names:
        lines.append(f"- `{name}`")
    lines.append("\nUse these names in the `secrets` array when creating custom tools. Values are injected as environment variables at runtime.")
    return "\n".join(lines)


@TOOL_REGISTRY.tool(
    name="custom_tools.create_custom_tool",
    description="""Create a new custom tool. The tool is stored locally and must be approved by the operator before it appears in the LLM's tool list.

The `parameters` field must be a valid JSON Schema object describing the tool's input parameters.
The `code` field is TypeScript source that will run in Deno. It receives a `params` object with the input parameters and has access to `output()` and `debug()` functions, plus the `tools` namespace.

Use `list_available_secrets` to see what secrets the operator has configured. Reference them by name in the `secrets` field — they'll be injected as env vars (accessed via `Deno.env.get("SECRET_NAME")`).""",
    parameters=[
        ToolParameter(
            name="name",
            type="string",
            description="Tool name (alphanumeric, underscores). Will be used as the tool name in the LLM's tool list.",
        ),
        ToolParameter(
            name="description",
            type="string",
            description="Description of what the tool does. Shown to the LLM.",
        ),
        ToolParameter(
            name="parameters",
            type="object",
            description="JSON Schema object describing the tool's input parameters.",
        ),
        ToolParameter(
            name="code",
            type="string",
            description="TypeScript source code. Has access to `params`, `output()`, `debug()`, and `tools` namespace.",
        ),
        ToolParameter(
            name="response_schema",
            type="object",
            description="Optional JSON Schema for the tool's response.",
            required=False,
        ),
        ToolParameter(
            name="secrets",
            type="array",
            description="Optional list of environment variable names to inject into the Deno process.",
            required=False,
        ),
    ],
)
async def create_custom_tool(
    ctx: ToolContext,
    name: str,
    description: str,
    parameters: dict[str, Any],
    code: str,
    response_schema: dict[str, Any] | None = None,
    secrets: list[str] | None = None,
) -> str:
    manager = ctx.custom_tool_manager
    if manager is None:
        return "Error: Custom tool manager not available."

    from src.tools.custom import CustomTool

    tool = CustomTool(
        name=name,
        description=description,
        parameters=parameters,
        code=code,
        approved=False,
        response_schema=response_schema,
        secrets=secrets or [],
    )
    return await manager.create_tool(tool)


@TOOL_REGISTRY.tool(
    name="custom_tools.update_custom_tool",
    description="Update an existing custom tool. If code or parameters change, approval will be reset.",
    parameters=[
        ToolParameter(
            name="name",
            type="string",
            description="Name of the tool to update.",
        ),
        ToolParameter(
            name="description",
            type="string",
            description="New description.",
            required=False,
        ),
        ToolParameter(
            name="parameters",
            type="object",
            description="New JSON Schema parameters.",
            required=False,
        ),
        ToolParameter(
            name="code",
            type="string",
            description="New TypeScript source code.",
            required=False,
        ),
        ToolParameter(
            name="response_schema",
            type="object",
            description="New response schema.",
            required=False,
        ),
        ToolParameter(
            name="secrets",
            type="array",
            description="New list of secret env var names.",
            required=False,
        ),
    ],
)
async def update_custom_tool(
    ctx: ToolContext,
    name: str,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    code: str | None = None,
    response_schema: dict[str, Any] | None = None,
    secrets: list[str] | None = None,
) -> str:
    manager = ctx.custom_tool_manager
    if manager is None:
        return "Error: Custom tool manager not available."

    fields: dict[str, Any] = {}
    if description is not None:
        fields["description"] = description
    if parameters is not None:
        fields["parameters"] = parameters
    if code is not None:
        fields["code"] = code
    if response_schema is not None:
        fields["response_schema"] = response_schema
    if secrets is not None:
        fields["secrets"] = secrets

    return await manager.update_tool(name, **fields)


@TOOL_REGISTRY.tool(
    name="custom_tools.delete_custom_tool",
    description="Delete a custom tool.",
    parameters=[
        ToolParameter(
            name="name",
            type="string",
            description="Name of the tool to delete.",
        ),
    ],
)
async def delete_custom_tool(ctx: ToolContext, name: str) -> str:
    manager = ctx.custom_tool_manager
    if manager is None:
        return "Error: Custom tool manager not available."
    return await manager.delete_tool(name)


@TOOL_REGISTRY.tool(
    name="custom_tools.list_custom_tools",
    description="List all custom tools with their name, description, and approval status.",
    parameters=[],
)
async def list_custom_tools(ctx: ToolContext) -> str:
    manager = ctx.custom_tool_manager
    if manager is None:
        return "Error: Custom tool manager not available."

    tools = manager.list_tools()
    if not tools:
        return "No custom tools found."

    lines = [f"Found {len(tools)} custom tool(s):\n"]
    for tool in tools:
        status = "approved" if tool.approved else "pending"
        param_keys = list(tool.parameters.get("properties", {}).keys()) if isinstance(tool.parameters, dict) else []
        params_preview = ", ".join(param_keys[:5])
        if len(param_keys) > 5:
            params_preview += ", ..."
        lines.append(f"- **{tool.name}** [{status}]: {tool.description}")
        if params_preview:
            lines.append(f"  params: ({params_preview})")
    return "\n".join(lines)


@TOOL_REGISTRY.tool(
    name="custom_tools.get_custom_tool",
    description="Get full details of a custom tool including its code.",
    parameters=[
        ToolParameter(
            name="name",
            type="string",
            description="Name of the tool to retrieve.",
        ),
    ],
)
async def get_custom_tool(ctx: ToolContext, name: str) -> str:
    manager = ctx.custom_tool_manager
    if manager is None:
        return "Error: Custom tool manager not available."

    tool = manager.get_tool(name)
    if tool is None:
        return f"Custom tool '{name}' not found."

    return json.dumps(tool.to_dict(), indent=2)


@TOOL_REGISTRY.tool(
    name="custom_tools.call_tool",
    description="Execute an approved custom tool by name with the given parameters.",
    parameters=[
        ToolParameter(
            name="name",
            type="string",
            description="Name of the custom tool to execute.",
        ),
        ToolParameter(
            name="params",
            type="object",
            description="Parameters to pass to the custom tool.",
        ),
    ],
)
async def call_tool(
    ctx: ToolContext, name: str, params: dict[str, Any]
) -> dict[str, Any]:
    manager = ctx.custom_tool_manager
    if manager is None:
        return {"success": False, "error": "Custom tool manager not available."}
    return await manager.execute_tool(name, params)

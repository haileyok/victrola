from typing import Any

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter


@TOOL_REGISTRY.tool(
    name="system.get_tool_docs",
    description="""Get full parameter documentation for one or more tools.

Returns the complete parameter list (name, type, required, description) for each
requested tool. Call this before using a tool whose parameters you don't have
in context yet — typically an MCP tool from the compact catalog. You only need
to call this once per tool per conversation — the results stay in your context.

Pass an array of full tool names (e.g. ["github.create_issue", "github.list_repos"]).
You can batch multiple tool doc requests in a single call.""",
    parameters=[
        ToolParameter(
            name="names",
            type="array",
            description="Array of full tool names (e.g. ['github.create_issue'])",
        ),
    ],
)
async def get_tool_docs(ctx: ToolContext, names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for name in names:
        tool = TOOL_REGISTRY.get(name)
        if tool is None:
            result[name] = {"error": f"Tool '{name}' not found"}
            continue

        params = []
        for p in tool.parameters:
            params.append(
                {
                    "name": p.name,
                    "type": p.type,
                    "required": p.required,
                    "description": p.description,
                }
            )

        result[name] = {
            "description": tool.description,
            "parameters": params,
        }

    return result

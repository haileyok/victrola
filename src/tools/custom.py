import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.store.store import Store
    from src.tools.executor import ToolExecutor
    from src.tools.secrets import SecretManager

logger = logging.getLogger(__name__)

TOOL_RKEY_PREFIX = "customtool:"


@dataclass
class CustomTool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    code: str
    approved: bool = False
    response_schema: dict[str, Any] | None = None
    secrets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "code": self.code,
            "approved": self.approved,
        }
        if self.response_schema:
            d["responseSchema"] = self.response_schema
        if self.secrets:
            d["secrets"] = self.secrets
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CustomTool":
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=data.get("parameters", {}),
            code=data.get("code", ""),
            approved=data.get("approved", False),
            response_schema=data.get("responseSchema", data.get("response_schema")),
            secrets=data.get("secrets", []),
        )


class CustomToolManager:
    """Manages custom tools stored in the local store and executed in Deno."""

    def __init__(
        self,
        store: "Store",
        executor: "ToolExecutor",
        secret_manager: "SecretManager | None" = None,
    ) -> None:
        self._store = store
        self._executor = executor
        self._secret_manager = secret_manager
        self._tools: dict[str, CustomTool] = {}

    async def load_tools(self) -> None:
        """Load all custom tools from the local store."""
        self._tools.clear()
        cursor: str | None = None

        while True:
            assert self._store.documents is not None
            resp = await self._store.documents.list(limit=100, cursor=cursor)
            documents = resp.get("documents", [])

            for doc in documents:
                rkey = doc.get("rkey", "")
                if not rkey.startswith(TOOL_RKEY_PREFIX):
                    continue
                content = doc.get("content", "")
                if not content:
                    continue
                try:
                    data = json.loads(content)
                    tool = CustomTool.from_dict(data)
                    self._tools[tool.name] = tool
                except Exception as e:
                    logger.warning(
                        "Failed to parse custom tool %s: %s",
                        rkey,
                        e,
                    )

            cursor = resp.get("cursor")
            if not cursor or not documents:
                break

        logger.info("Loaded %d custom tool(s)", len(self._tools))

    def get_tool(self, name: str) -> CustomTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[CustomTool]:
        return list(self._tools.values())

    def get_approved_tools(self) -> list[CustomTool]:
        return [t for t in self._tools.values() if t.approved]

    def build_tool_definitions(self) -> list[dict[str, Any]]:
        """Build Anthropic-format tool definitions for approved custom tools."""
        definitions: list[dict[str, Any]] = []
        for tool in self.get_approved_tools():
            defn: dict[str, Any] = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            definitions.append(defn)
        return definitions

    async def execute_tool(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute an approved custom tool."""
        tool = self._tools.get(name)
        if tool is None:
            return {"success": False, "error": f"Custom tool not found: {name}"}
        if not tool.approved:
            return {"success": False, "error": f"Custom tool '{name}' is not approved"}

        env = self._build_env(tool)
        return await self._executor.execute_custom_tool_code(
            code=tool.code,
            params=params,
            env=env,
            allow_net=True,
        )

    def _build_env(self, tool: CustomTool) -> dict[str, str]:
        """Build environment variables for a custom tool execution."""
        env: dict[str, str] = {}

        # inject requested secrets from local secret store
        if self._secret_manager:
            for secret_name in tool.secrets:
                val = self._secret_manager.get_secret(secret_name)
                if val:
                    env[secret_name.upper()] = val

        return env

    async def create_tool(self, tool: CustomTool) -> str:
        """Create a new custom tool in the local store."""
        rkey = f"{TOOL_RKEY_PREFIX}{tool.name}"
        content = json.dumps(tool.to_dict())
        assert self._store.documents is not None

        try:
            await self._store.documents.create(rkey, content)
        except Exception as e:
            msg = str(e).lower()
            if "already exists" in msg or "conflict" in msg:
                await self._store.documents.update(rkey, content)
            else:
                raise

        self._tools[tool.name] = tool
        return f"Custom tool '{tool.name}' created."

    async def update_tool(self, name: str, **fields: Any) -> str:
        """Update an existing custom tool. Resets approval if code or parameters change."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Custom tool '{name}' not found."

        resets_approval = False
        for key, value in fields.items():
            if value is None:
                continue
            if key in ("code", "parameters") and getattr(tool, key) != value:
                resets_approval = True
            setattr(tool, key, value)

        if resets_approval:
            tool.approved = False

        rkey = f"{TOOL_RKEY_PREFIX}{name}"
        content = json.dumps(tool.to_dict())
        assert self._store.documents is not None
        await self._store.documents.update(rkey, content)
        return f"Custom tool '{name}' updated." + (
            " Approval reset." if resets_approval else ""
        )

    async def delete_tool(self, name: str) -> str:
        """Delete a custom tool from the local store."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Custom tool '{name}' not found."

        rkey = f"{TOOL_RKEY_PREFIX}{name}"
        assert self._store.documents is not None
        try:
            await self._store.documents.delete(rkey)
        except Exception:
            logger.exception("Failed to delete custom tool doc %s", rkey)

        self._tools.pop(name, None)
        return f"Custom tool '{name}' deleted."

    async def approve_tool(self, name: str) -> str:
        """Approve a custom tool for use by the LLM."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Custom tool '{name}' not found."

        tool.approved = True
        rkey = f"{TOOL_RKEY_PREFIX}{name}"
        content = json.dumps(tool.to_dict())
        assert self._store.documents is not None
        await self._store.documents.update(rkey, content)
        return f"Custom tool '{name}' approved."

    async def revoke_tool(self, name: str) -> str:
        """Revoke approval for a custom tool."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Custom tool '{name}' not found."

        tool.approved = False
        rkey = f"{TOOL_RKEY_PREFIX}{name}"
        content = json.dumps(tool.to_dict())
        assert self._store.documents is not None
        await self._store.documents.update(rkey, content)
        return f"Custom tool '{name}' approval revoked."

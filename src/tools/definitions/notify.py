import logging
from typing import Any

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter

logger = logging.getLogger(__name__)

DISCORD_WEBHOOK_SECRET = "DISCORD_WEBHOOK_URL"
# Discord's hard limit on webhook `content`. Embeds can hold more in `description`,
# but truncating content keeps the tool's contract simple.
MAX_CONTENT = 2000


@TOOL_REGISTRY.tool(
    name="notify.discord",
    description="""Send a message to Discord via webhook. Useful for alerting the operator about background events — scheduled task results, findings that need attention, errors, etc.

Requires a secret named `DISCORD_WEBHOOK_URL` (a webhook URL from a Discord channel's integration settings). The operator configures it via the Secrets screen in the TUI.

If `title` is provided the message renders as an embed with the title as a heading; otherwise as a plain message. Discord limits content to 2000 characters and this tool truncates beyond that.""",
    parameters=[
        ToolParameter(
            name="content",
            type="string",
            description="Message body. Truncated to 2000 chars.",
        ),
        ToolParameter(
            name="title",
            type="string",
            description="Optional title; if provided, renders as an embed.",
            required=False,
        ),
    ],
)
async def discord(
    ctx: ToolContext, content: str, title: str | None = None
) -> dict[str, Any]:
    if not content:
        return {"error": "content is required"}

    sm = ctx.secret_manager
    if sm is None:
        return {"error": "Secret manager not available"}

    webhook_url = sm.get_secret(DISCORD_WEBHOOK_SECRET)
    if not webhook_url:
        return {
            "error": (
                f"Discord webhook not configured. Set the `{DISCORD_WEBHOOK_SECRET}` "
                "secret via the TUI (press 's' from the session list)."
            )
        }

    content = content[:MAX_CONTENT]

    if title:
        payload: dict[str, Any] = {
            "embeds": [{"title": title[:256], "description": content}]
        }
    else:
        payload = {"content": content}

    try:
        resp = await ctx.http_client.post(webhook_url, json=payload)
    except Exception as e:
        logger.exception("Discord webhook post failed")
        return {"error": str(e)}

    if resp.status_code >= 400:
        return {
            "error": f"Discord webhook returned HTTP {resp.status_code}",
            "body": resp.text[:500],
        }
    return {"success": True, "status_code": resp.status_code}

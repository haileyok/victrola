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

Requires a secret named `DISCORD_WEBHOOK_URL` (a webhook URL from a Discord channel's integration settings). The operator configures it via the Secrets page in the web interface.

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
                "secret via the web interface (Secrets page)."
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


@TOOL_REGISTRY.tool(
    name="notify.signal",
    description="""Send a message to the operator via Signal.

Requires signal-cli-rest-api running and configured in .env (SIGNAL_SERVICE, SIGNAL_BOT_PHONE, SIGNAL_OPERATOR_PHONE). Messages are chunked at ~1900 chars to stay within Signal's limits. Works independently of the SignalBot chat loop — use this for notifications even when the bot isn't running.""",
    parameters=[
        ToolParameter(
            name="content",
            type="string",
            description="Message body.",
        ),
        ToolParameter(
            name="title",
            type="string",
            description="Optional title (prepended to the message).",
            required=False,
        ),
    ],
)
async def signal(
    ctx: ToolContext, content: str, title: str | None = None
) -> dict[str, Any]:
    from src.config import CONFIG
    from src.utils.text import _chunk

    if not content:
        return {"error": "content is required"}

    if (
        not CONFIG.signal_service
        or not CONFIG.signal_bot_phone
        or not CONFIG.signal_operator_phone
    ):
        return {
            "error": (
                "Signal not configured. Set SIGNAL_SERVICE, SIGNAL_BOT_PHONE, "
                "and SIGNAL_OPERATOR_PHONE in .env."
            )
        }

    message = f"{title}\n\n{content}" if title else content

    sent = 0
    for chunk in _chunk(message):
        try:
            resp = await ctx.http_client.post(
                f"http://{CONFIG.signal_service}/v2/send/{CONFIG.signal_bot_phone}",
                json={
                    "message": chunk,
                    "recipients": [CONFIG.signal_operator_phone],
                },
            )
        except Exception as e:
            logger.exception("Signal send failed")
            return {"error": str(e)}

        if resp.status_code >= 400:
            return {
                "error": f"Signal API returned HTTP {resp.status_code}",
                "body": resp.text[:500],
            }
        sent += 1

    return {"success": True, "chunks_sent": sent}


@TOOL_REGISTRY.tool(
    name="notify.send",
    description="""Send a notification to the operator via the default channel.
Routes to Signal if configured, otherwise Discord. Use this for general notifications when you don't care which channel.""",
    parameters=[
        ToolParameter(
            name="content",
            type="string",
            description="Message body.",
        ),
        ToolParameter(
            name="title",
            type="string",
            description="Optional title.",
            required=False,
        ),
    ],
)
async def send(
    ctx: ToolContext, content: str, title: str | None = None
) -> dict[str, Any]:
    from src.config import CONFIG

    if (
        CONFIG.signal_service
        and CONFIG.signal_bot_phone
        and CONFIG.signal_operator_phone
    ):
        return await signal(ctx, content, title)
    return await discord(ctx, content, title)

import base64
import logging
from typing import Any

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB


@TOOL_REGISTRY.tool(
    name="image.view_image",
    description="View an image from a URL. The image will be shown to you directly so you can see its contents.",
    parameters=[
        ToolParameter(
            name="url",
            type="string",
            description="The image URL (e.g., from post embeds)",
        ),
    ],
)
async def view_image(ctx: ToolContext, url: str) -> dict[str, Any]:
    if not url:
        return {"error": "url is required"}

    logger.info("view_image fetching url=%s", url)

    resp = await ctx.http_client.get(url)
    if resp.status_code != 200:
        return {"error": f"failed to fetch image: HTTP {resp.status_code}"}

    content_type = resp.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        return {
            "error": f"URL does not point to an image (content-type: {content_type})"
        }

    image_data = resp.content
    if len(image_data) > MAX_IMAGE_SIZE:
        return {
            "error": f"image too large ({len(image_data)} bytes, max {MAX_IMAGE_SIZE})"
        }

    b64 = base64.b64encode(image_data).decode("ascii")
    # strip parameters from content-type (e.g. "image/jpeg; charset=utf-8" -> "image/jpeg")
    media_type = content_type.split(";")[0].strip()

    logger.info(
        "view_image returning bytes=%d base64_len=%d", len(image_data), len(b64)
    )

    return {
        "type": "image_result",
        "text": "Here is the image:",
        "image": {
            "type": "base64",
            "media_type": media_type,
            "data": b64,
        },
    }

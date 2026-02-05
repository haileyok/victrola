import asyncio
import logging
from typing import Any

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter

logger = logging.getLogger(__name__)

MAX_FETCH_CHARS = 50000
DEFAULT_FETCH_CHARS = 10000
DEFAULT_NUM_RESULTS = 5
MAX_NUM_RESULTS = 10


def _format_search_results(results: list[Any]) -> list[dict[str, Any]]:
    """Format exa search results into a list of dicts."""
    formatted: list[dict[str, Any]] = []
    for r in results:
        entry: dict[str, Any] = {
            "title": getattr(r, "title", None) or "",
            "url": getattr(r, "url", None) or "",
        }
        if getattr(r, "author", None):
            entry["author"] = r.author
        if getattr(r, "published_date", None):
            entry["published_date"] = r.published_date
        if getattr(r, "score", None) is not None:
            entry["score"] = round(r.score, 3)

        text = getattr(r, "text", None) or ""
        if text:
            # Truncate individual result text for search results
            if len(text) > 2000:
                text = text[:2000] + "..."
            entry["text"] = text

        if getattr(r, "highlights", None):
            entry["highlights"] = r.highlights

        formatted.append(entry)
    return formatted


@TOOL_REGISTRY.tool(
    name="web.web_search",
    description="""Search the web for current information via Exa AI. Returns results with titles, URLs, and text snippets.

If summary_focus is provided, the results will include a note about what to focus on when analyzing them.""",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="The search query - be specific and descriptive for best results",
        ),
        ToolParameter(
            name="summary_focus",
            type="string",
            description="If provided, indicates what aspect of the results to focus on when analyzing",
            required=False,
        ),
        ToolParameter(
            name="num_results",
            type="number",
            description="Number of results to return (1-10, default 5)",
            required=False,
            default=5,
        ),
    ],
)
async def web_search(
    ctx: ToolContext,
    query: str,
    summary_focus: str | None = None,
    num_results: int = DEFAULT_NUM_RESULTS,
) -> dict[str, Any]:
    if not query:
        return {"error": "query is required"}

    num_results = max(1, min(num_results, MAX_NUM_RESULTS))
    exa = ctx.exa_client

    logger.info("web_search query=%s num_results=%d", query, num_results)

    # exa_py.Exa.search is synchronous, run in thread pool
    resp = await asyncio.to_thread(
        exa.search,
        query,
        num_results=num_results,
    )

    results = _format_search_results(resp.results)

    result: dict[str, Any] = {
        "query": query,
        "count": len(results),
        "results": results,
    }

    if summary_focus:
        result["summary_focus"] = summary_focus

    return result


@TOOL_REGISTRY.tool(
    name="web.http_get",
    description="""Plain HTTP GET of a URL. Returns the raw response body (truncated). Unlike `fetch_page`, this does NOT go through Exa — use it for JSON APIs, URLs behind auth, or when you just want the raw response.

Declare any required auth tokens as secrets (via `create_custom_tool`) if you need Authorization headers; this tool itself doesn't inject them. For simple GETs to public URLs this is the right tool.""",
    parameters=[
        ToolParameter(
            name="url",
            type="string",
            description="The URL to fetch",
        ),
        ToolParameter(
            name="max_chars",
            type="number",
            description="Maximum characters of body to return (default 10000, max 50000)",
            required=False,
            default=10000,
        ),
    ],
)
async def http_get(
    ctx: ToolContext, url: str, max_chars: int = DEFAULT_FETCH_CHARS
) -> dict[str, Any]:
    if not url:
        return {"error": "url is required"}
    max_chars = max(1, min(max_chars, MAX_FETCH_CHARS))

    try:
        resp = await ctx.http_client.get(url, follow_redirects=True)
    except Exception as e:
        return {"error": f"request failed: {e}"}

    body = resp.text
    truncated = False
    if len(body) > max_chars:
        body = body[:max_chars]
        truncated = True

    return {
        "url": str(resp.url),
        "status_code": resp.status_code,
        "content_type": resp.headers.get("content-type", ""),
        "body": body,
        "truncated": truncated,
    }


@TOOL_REGISTRY.tool(
    name="web.fetch_page",
    description="""Fetch and read the contents of a webpage. Use this to read articles, documentation, blog posts, or any URL you want to examine in detail.""",
    parameters=[
        ToolParameter(
            name="url",
            type="string",
            description="The URL of the webpage to fetch",
        ),
        ToolParameter(
            name="max_chars",
            type="number",
            description="Maximum characters to return (default 10000, max 50000)",
            required=False,
            default=10000,
        ),
    ],
)
async def fetch_page(
    ctx: ToolContext, url: str, max_chars: int = DEFAULT_FETCH_CHARS
) -> dict[str, Any]:
    if not url:
        return {"error": "url is required"}

    max_chars = max(1, min(max_chars, MAX_FETCH_CHARS))
    exa = ctx.exa_client

    logger.info("fetch_page url=%s max_chars=%d", url, max_chars)

    # exa_py.Exa.get_contents is synchronous, run in thread pool
    resp = await asyncio.to_thread(
        exa.get_contents,
        url,
        text={"max_characters": max_chars},
    )

    if not resp.results:
        return {"error": f"No content returned for {url}"}

    page = resp.results[0]
    text = getattr(page, "text", None) or ""

    result: dict[str, Any] = {
        "title": getattr(page, "title", None) or "",
        "url": getattr(page, "url", None) or url,
        "text": text,
    }

    if getattr(page, "author", None):
        result["author"] = page.author
    if getattr(page, "published_date", None):
        result["published_date"] = page.published_date

    return result

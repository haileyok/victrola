import logging
from typing import Any

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter

logger = logging.getLogger(__name__)


async def _get_note(ctx: ToolContext, rkey: str) -> dict[str, Any] | None:
    """Fetch a note by rkey, returning None if not found."""
    assert ctx.store.documents is not None
    try:
        return await ctx.store.documents.get(rkey)
    except Exception as e:
        if "not found" in str(e).lower():
            return None
        raise


@TOOL_REGISTRY.tool(
    name="notes.note_upsert",
    description="""Create or update a note by rkey (note name).

Use this to persist information across sessions. If the note exists, it will be updated (overwritten); otherwise created. To append rather than overwrite, read the existing note first and write the combined content.

Conventional rkeys:
- `self` — your identity, personality, and behavior instructions (loaded into your system prompt at boot)
- `operator` — what you know about the human operator (preferences, context, ongoing projects)
- `skill:<name>` — a reusable procedure you can load and execute
- `task:<name>` — progress / state for a long-running task
- any other rkey — free-form facts, reminders, scratch space

Rkey rules: 1-512 chars, alphanumeric with `-_.~:`""",
    parameters=[
        ToolParameter(
            name="rkey",
            type="string",
            description="The note name/identifier",
        ),
        ToolParameter(
            name="content",
            type="string",
            description="The full note content (overwrites any existing content)",
        ),
    ],
)
async def note_upsert(ctx: ToolContext, rkey: str, content: str) -> str:
    if not rkey:
        return "Error: rkey is required"
    if not content:
        return "Error: content is required"

    docs = ctx.store.documents
    assert docs is not None

    # try update first, fall back to create if not found
    try:
        await docs.update(rkey, content)
        return f"Note '{rkey}' updated."
    except Exception as e:
        if "not found" in str(e).lower():
            try:
                await docs.create(rkey, content)
                return f"Note '{rkey}' created."
            except Exception as create_err:
                return f"Error creating note: {create_err}"
        return f"Error updating note: {e}"


@TOOL_REGISTRY.tool(
    name="notes.note_get",
    description="""Retrieve the full content of one or more notes by rkey. Works for any rkey — `self`, `operator`, `skill:*`, `task:*`, or any free-form note name.

Call this to load full content before relying on a note. The system prompt only shows skill names with short previews — you must `note_get` to see a skill's actual content before executing it. The `operator` note is not preloaded at all — call `note_get(['operator'])` when you need to recall context about the operator.

Pass an array of rkeys to batch multiple reads into one call.""",
    parameters=[
        ToolParameter(
            name="rkeys",
            type="array",
            description="Array of note names to retrieve (must be an array even for a single note)",
        ),
    ],
)
async def note_get(ctx: ToolContext, rkeys: list[str]) -> str:
    if not rkeys:
        return "Error: rkeys array is required and must not be empty"

    parts: list[str] = []
    for rkey in rkeys:
        doc = await _get_note(ctx, rkey)
        if doc is not None:
            parts.append(f"Note '{doc.get('rkey', rkey)}':\n\n{doc.get('content', '')}")
        else:
            parts.append(f"Note '{rkey}': Error - not found")

    return "\n\n---\n\n".join(parts)


@TOOL_REGISTRY.tool(
    name="notes.note_list",
    description="""List all stored notes with short content previews.

Use this to discover what notes exist when you're not sure, or to audit your memory. The system prompt shows `skill:*` notes by name but not your other notes — call `note_list` to see everything you've saved, then `note_get` for full content of anything interesting.""",
    parameters=[
        ToolParameter(
            name="limit",
            type="number",
            description="Max notes to return (default 20, max 100)",
            required=False,
            default=20,
        ),
        ToolParameter(
            name="cursor",
            type="string",
            description="Pagination cursor (only needed if a previous call returned one)",
            required=False,
        ),
    ],
)
async def note_list(
    ctx: ToolContext, limit: int = 20, cursor: str | None = None
) -> str:
    if limit > 100:
        limit = 100

    assert ctx.store.documents is not None
    resp = await ctx.store.documents.list(limit=limit, cursor=cursor)

    documents = resp.get("documents", [])
    if not documents:
        return "No notes found."

    lines: list[str] = [f"Found {len(documents)} note(s):\n"]

    for doc in documents:
        rkey = doc.get("rkey", "")
        content = doc.get("content", "")
        preview = content[:100] + "..." if len(content) > 100 else content
        lines.append(f"- {rkey}: {preview}")

    next_cursor = resp.get("cursor")
    if next_cursor:
        lines.append(f"\nMore available. Use cursor: {next_cursor}")

    return "\n".join(lines)

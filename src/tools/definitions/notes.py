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
    description="""Create or replace a note by rkey. **The `content` you pass REPLACES the entire note — there is no merging.**

### Behavior
- Note does not exist (or is empty) → created/written normally.
- Note already has content → **rejected** unless you pass `overwrite: true`. This is a safety net so you don't clobber context you forgot about.
- `overwrite: true` does NOT merge. Whatever you pass as `content` is what the note becomes.

### How to ADD to a note without losing what's there
There is no `append` tool — you do it yourself. For `self`/`operator` the current content is already in your system prompt, so just build the new content as:

```
<existing content from system prompt>

<your new addition>
```

…and call `note_upsert(..., overwrite=true)`. For other rkeys whose content isn't preloaded, `note_get` them first, then concat, then upsert.

### Conventional rkeys
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
            description="The full note content. REPLACES the existing note entirely — include any prior content you want to preserve.",
        ),
        ToolParameter(
            name="overwrite",
            type="boolean",
            description="Must be true to replace an existing non-empty note. Default false — the call is rejected otherwise, to prevent accidental clobbering. Setting true does NOT merge; `content` still replaces the whole note.",
            required=False,
            default=False,
        ),
    ],
)
async def note_upsert(
    ctx: ToolContext, rkey: str, content: str, overwrite: bool = False
) -> str:
    if not rkey:
        return "Error: rkey is required"
    if not content:
        return "Error: content is required"

    docs = ctx.store.documents
    assert docs is not None

    existed = False
    existing_content = ""
    try:
        doc = await docs.get(rkey)
        existed = True
        existing_content = doc.get("content", "") or ""
    except Exception as e:
        if "not found" not in str(e).lower():
            return f"Error reading note before upsert: {e}"

    if existing_content and not overwrite:
        preview = existing_content[:200].replace("\n", " ")
        if len(existing_content) > 200:
            preview += "…"
        return (
            f"Rejected: note '{rkey}' already has {len(existing_content)} chars of content. "
            f"Preview: {preview}\n\n"
            f"Call note_upsert again with `overwrite: true` to proceed. Remember: `overwrite: true` REPLACES the whole note — it does not merge. "
            f"If you want to keep the existing content, prepend it to your new content before calling again "
            f"(for `self`/`operator` the full current content is in your system prompt; for other notes, `note_get` first)."
        )

    if existed:
        try:
            await docs.update(rkey, content)
            return f"Note '{rkey}' updated."
        except Exception as e:
            return f"Error updating note: {e}"

    try:
        await docs.create(rkey, content)
        return f"Note '{rkey}' created."
    except Exception as e:
        return f"Error creating note: {e}"


@TOOL_REGISTRY.tool(
    name="notes.note_get",
    description="""Retrieve the full content of one or more notes by rkey. Works for any rkey — `self`, `operator`, `skill:*`, `task:*`, or any free-form note name.

The `self` and `operator` notes are preloaded into your system prompt, so you already have their content and don't need to fetch them. For `skill:*` notes, only the name + short preview is in the system prompt, so you must `note_get` before executing a skill. For any other note whose content isn't in your prompt, `note_get` before reasoning about it — and always `note_get` before calling `note_upsert` with `overwrite: true` if you want to preserve the prior content.

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

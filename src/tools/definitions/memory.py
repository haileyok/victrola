"""Memory tools — entry-based memory with hybrid search.

Replaces the old `notes.*` tools. Entries are discrete, independently
editable rows in `memory_entries`, typed as self/operator/skill/episodic/factual.
"""

import logging
from typing import Any

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter

logger = logging.getLogger(__name__)

_VALID_TYPES = {"self", "operator", "skill", "episodic", "factual"}
_SCOPE_PATTERN = "skill:"


@TOOL_REGISTRY.tool(
    name="memory.add",
    description="""Create a new memory entry.

Memory types:
- `self` — your identity, personality, and behavior instructions (single entry; always in system prompt)
- `operator` — facts about the human operator (multi-entry; all in system prompt)
- `skill` — a reusable procedure document (single entry per skill; name + preview in prompt, loaded on demand)
- `episodic` — individual memories of events/experiences (RAG-retrieved per turn)
- `factual` — individual facts and knowledge (RAG-retrieved per turn)

Use `memory.add` to create discrete entries — you don't need to read-then-rewrite
like the old `note_upsert`. For `self`, use `memory.update` if an entry already exists.

Scope is the grouping key:
- `self` type → scope must be `self`
- `skill` type → scope must be `skill:<name>`
- `operator` type → scope is `operator`
- `episodic` / `factual` → scope can be a session ID, topic, or free-form string

Tags are optional strings stored in metadata for filtering.""",
    parameters=[
        ToolParameter(
            name="type",
            type="string",
            description="Memory type: self, operator, skill, episodic, or factual",
        ),
        ToolParameter(
            name="scope",
            type="string",
            description="Grouping key (e.g. 'self', 'operator', 'skill:deploy', session ID, topic)",
        ),
        ToolParameter(
            name="content",
            type="string",
            description="The text content of the memory entry",
        ),
        ToolParameter(
            name="tags",
            type="array",
            description="Optional array of tag strings for filtering",
            required=False,
        ),
    ],
)
async def memory_add(
    ctx: ToolContext,
    type: str,
    scope: str,
    content: str,
    tags: list[str] | None = None,
) -> str:
    if type not in _VALID_TYPES:
        return f"Error: type must be one of {sorted(_VALID_TYPES)}, got '{type}'"
    if not scope:
        return "Error: scope is required"
    if not content:
        return "Error: content is required"

    # Validate scope conventions
    if type == "self" and scope != "self":
        return "Error: for type 'self', scope must be 'self'"
    if type == "skill" and not scope.startswith(_SCOPE_PATTERN):
        return f"Error: for type 'skill', scope must start with 'skill:' (e.g. 'skill:deploy')"

    store = ctx.store
    if store.memory is None:
        raise RuntimeError("MemoryStore is not initialized")

    # Guard: only one 'self' entry allowed
    if type == "self":
        if await store.memory.has_self_entry():
            return (
                "Error: a 'self' entry already exists. Use memory.update(id, content=...) "
                "to modify it instead of creating a new one."
            )

    metadata: dict[str, Any] = {}
    if tags:
        metadata["tags"] = tags

    entry = await store.memory.add_entry(
        type=type,
        scope=scope,
        content=content,
        metadata=metadata,
    )

    # Invalidate search cache
    if ctx._search_engine is not None:
        ctx.search_engine.invalidate_cache()

    return f"Created memory entry (id={entry['id']}, type={type}, scope={scope})."


@TOOL_REGISTRY.tool(
    name="memory.update",
    description="""Update a specific memory entry by ID.

Only the fields you provide are changed — there is no whole-replacement semantics.
If `content` is changed, the embedding is regenerated automatically.

Use `memory.search` or `memory.get` first to find the entry ID you want to update.""",
    parameters=[
        ToolParameter(
            name="id",
            type="number",
            description="The entry ID to update",
        ),
        ToolParameter(
            name="content",
            type="string",
            description="New content (only provided fields are changed)",
            required=False,
        ),
        ToolParameter(
            name="tags",
            type="array",
            description="New tags array (replaces existing tags if provided)",
            required=False,
        ),
    ],
)
async def memory_update(
    ctx: ToolContext,
    id: int,
    content: str | None = None,
    tags: list[str] | None = None,
) -> str:
    if content is None and tags is None:
        return "Error: at least one of 'content' or 'tags' must be provided"

    store = ctx.store
    if store.memory is None:
        raise RuntimeError("MemoryStore is not initialized")

    # Fetch current entry to merge metadata
    current = await store.memory.get_entry(int(id))
    if current is None:
        return f"Error: entry {id} not found"

    metadata = current.get("metadata", {})
    if tags is not None:
        metadata["tags"] = tags

    updated = await store.memory.update_entry(
        id=int(id),
        content=content,
        metadata=metadata,
    )

    if updated is None:
        return f"Error: entry {id} not found"

    # Invalidate search cache
    if ctx._search_engine is not None:
        ctx.search_engine.invalidate_cache()

    return f"Updated memory entry (id={id})."


@TOOL_REGISTRY.tool(
    name="memory.delete",
    description="""Delete a single memory entry by ID. Other entries in the same scope are unaffected.""",
    parameters=[
        ToolParameter(
            name="id",
            type="number",
            description="The entry ID to delete",
        ),
    ],
)
async def memory_delete(ctx: ToolContext, id: int) -> str:
    store = ctx.store
    if store.memory is None:
        raise RuntimeError("MemoryStore is not initialized")

    deleted = await store.memory.delete_entry(int(id))
    if not deleted:
        return f"Error: entry {id} not found"

    # Invalidate search cache
    if ctx._search_engine is not None:
        ctx.search_engine.invalidate_cache()

    return f"Deleted memory entry (id={id})."


@TOOL_REGISTRY.tool(
    name="memory.search",
    description="""Search memory entries using hybrid keyword + semantic search.

Combines FTS5 keyword matching with vector cosine similarity over embeddings.
Results include a `score` (0-1, higher is better) and `matched_by` indicator
(semantic, keyword, or both).

All filter parameters are optional. Use `type` to restrict to a single type,
or `types` for multiple types.""",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Natural-language search query",
        ),
        ToolParameter(
            name="type",
            type="string",
            description="Filter to a single type (self, operator, skill, episodic, factual)",
            required=False,
        ),
        ToolParameter(
            name="types",
            type="array",
            description="Filter to multiple types (e.g. ['episodic', 'factual'])",
            required=False,
        ),
        ToolParameter(
            name="scope",
            type="string",
            description="Filter to a specific scope",
            required=False,
        ),
        ToolParameter(
            name="tags",
            type="array",
            description="Filter by tags (entries matching any tag)",
            required=False,
        ),
        ToolParameter(
            name="limit",
            type="number",
            description="Max results (default 10, max 100)",
            required=False,
            default=10,
        ),
    ],
)
async def memory_search(
    ctx: ToolContext,
    query: str,
    type: str | None = None,
    types: list[str] | None = None,
    scope: str | None = None,
    tags: list[str] | None = None,
    limit: int = 10,
) -> str:
    if not query:
        return "Error: query is required"

    if ctx._search_engine is None:
        return "Error: search engine not configured"

    results = await ctx.search_engine.search(
        query=query,
        type=type,
        types=types,
        scope=scope,
        tags=tags,
        limit=limit,
    )

    if not results:
        return "No matching memory entries found."

    lines = [f"Found {len(results)} matching entr{'y' if len(results) == 1 else 'ies'}:\n"]
    for r in results:
        preview = r["content"][:150].replace("\n", " ")
        if len(r["content"]) > 150:
            preview += "…"
        lines.append(
            f"- [id={r['id']}] [{r['type']}/{r['scope']}] score={r['score']:.2f} "
            f"({r['matched_by']}): {preview}"
        )

    return "\n".join(lines)


@TOOL_REGISTRY.tool(
    name="memory.get",
    description="""Retrieve all entries for a given scope.

Use this to review everything you know about a topic or person. For example,
scope='operator' returns all operator facts. Returns full content of each entry.""",
    parameters=[
        ToolParameter(
            name="scope",
            type="string",
            description="The scope to retrieve entries for (e.g. 'operator', 'skill:deploy')",
        ),
    ],
)
async def memory_get(ctx: ToolContext, scope: str) -> str:
    if not scope:
        return "Error: scope is required"

    store = ctx.store
    if store.memory is None:
        raise RuntimeError("MemoryStore is not initialized")

    # Try to infer the type from scope
    if scope == "self":
        entries = await store.memory.get_by_scope("self", "self")
    elif scope == "operator":
        entries = await store.memory.get_by_scope("operator", "operator")
    elif scope.startswith("skill:"):
        entries = await store.memory.get_by_scope("skill", scope)
    else:
        entries = await store.memory.get_by_scope_any_type(scope)

    if not entries:
        return f"No entries found for scope '{scope}'."

    lines = [f"Found {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} for scope '{scope}':\n"]
    for e in entries:
        lines.append(f"[id={e['id']}] [{e['type']}]\n{e['content']}\n")

    return "\n".join(lines)


@TOOL_REGISTRY.tool(
    name="memory.list_skills",
    description="""List all saved skills with name and 80-char preview.""",
    parameters=[],
)
async def memory_list_skills(ctx: ToolContext) -> str:
    store = ctx.store
    if store.memory is None:
        raise RuntimeError("MemoryStore is not initialized")

    skills = await store.memory.list_skills()
    if not skills:
        return "No skills found."

    lines = [f"Found {len(skills)} skill(s):\n"]
    for s in skills:
        lines.append(f"- **{s['name']}** (id={s['id']}): {s['preview']}")

    return "\n".join(lines)


@TOOL_REGISTRY.tool(
    name="memory.get_skill",
    description="""Load the full content of a skill by name.""",
    parameters=[
        ToolParameter(
            name="name",
            type="string",
            description="Skill name (without the 'skill:' prefix)",
        ),
    ],
)
async def memory_get_skill(ctx: ToolContext, name: str) -> str:
    if not name:
        return "Error: name is required"

    store = ctx.store
    if store.memory is None:
        raise RuntimeError("MemoryStore is not initialized")

    skill = await store.memory.get_skill(name)
    if skill is None:
        return f"Skill '{name}' not found."

    return f"Skill: {name} (id={skill['id']})\n\n{skill['content']}"

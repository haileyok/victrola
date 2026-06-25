"""Memory CRUD + search endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from src.tools.executor import ToolExecutor
from src.web.dependencies import get_executor
from src.web.schemas import (
    CreateMemoryEntryRequest,
    MemoryEntryResponse,
    MemoryListResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MemorySearchResult,
    UpdateMemoryEntryRequest,
)

router = APIRouter()

_VALID_TYPES = {"self", "operator", "skill", "episodic", "factual"}
_SCOPE_PATTERN = "skill:"


def _get_memory_store(executor: ToolExecutor):
    store = executor.store
    if store.memory is None:
        raise HTTPException(500, "Memory store not initialized")
    return store.memory


def _get_search_engine(executor: ToolExecutor):
    se = executor.ctx._search_engine
    if se is None:
        raise HTTPException(503, "Search engine not configured (embeddings/numpy unavailable)")
    return se


def _invalidate_search(executor: ToolExecutor):
    se = executor.ctx._search_engine
    if se is not None:
        se.invalidate_cache()


def _validate_entry(type: str, scope: str):
    """Validate type and scope conventions (mirrors memory.add tool logic)."""
    if type not in _VALID_TYPES:
        raise HTTPException(422, f"Type must be one of {sorted(_VALID_TYPES)}, got '{type}'")
    if not scope:
        raise HTTPException(422, "Scope is required")
    if type == "self" and scope != "self":
        raise HTTPException(422, "For type 'self', scope must be 'self'")
    if type == "skill" and not scope.startswith(_SCOPE_PATTERN):
        raise HTTPException(422, "For type 'skill', scope must start with 'skill:'")


@router.get("/memory", response_model=MemoryListResponse)
async def list_memory(
    type: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    cursor: int | None = Query(default=None),
    executor: ToolExecutor = Depends(get_executor),
) -> MemoryListResponse:
    ms = _get_memory_store(executor)
    if type is not None and type not in _VALID_TYPES:
        raise HTTPException(422, f"Invalid type '{type}'")
    result = await ms.list_entries(type=type, limit=limit, cursor=cursor)
    return MemoryListResponse(**result)


@router.get("/memory/{entry_id}", response_model=MemoryEntryResponse)
async def get_memory(
    entry_id: int,
    executor: ToolExecutor = Depends(get_executor),
) -> MemoryEntryResponse:
    ms = _get_memory_store(executor)
    entry = await ms.get_entry(entry_id)
    if entry is None:
        raise HTTPException(404, f"Memory entry {entry_id} not found")
    return MemoryEntryResponse(**entry)


@router.post("/memory", response_model=MemoryEntryResponse, status_code=201)
async def create_memory(
    body: CreateMemoryEntryRequest,
    executor: ToolExecutor = Depends(get_executor),
) -> MemoryEntryResponse:
    _validate_entry(body.type, body.scope)
    ms = _get_memory_store(executor)

    # Guard: only one 'self' entry allowed
    if body.type == "self":
        if await ms.has_self_entry():
            raise HTTPException(409, "A 'self' entry already exists. Use update instead.")

    metadata: dict = {}
    if body.tags:
        metadata["tags"] = body.tags

    entry = await ms.add_entry(
        type=body.type,
        scope=body.scope,
        content=body.content,
        metadata=metadata,
    )
    _invalidate_search(executor)
    return MemoryEntryResponse(**entry)


@router.put("/memory/{entry_id}", response_model=MemoryEntryResponse)
async def update_memory(
    entry_id: int,
    body: UpdateMemoryEntryRequest,
    executor: ToolExecutor = Depends(get_executor),
) -> MemoryEntryResponse:
    if body.content is None and body.tags is None:
        raise HTTPException(422, "At least one of 'content' or 'tags' must be provided")
    ms = _get_memory_store(executor)

    # Fetch current entry to merge metadata (mirrors memory.update tool)
    current = await ms.get_entry(entry_id)
    if current is None:
        raise HTTPException(404, f"Memory entry {entry_id} not found")

    metadata = current.get("metadata", {})
    if body.tags is not None:
        metadata["tags"] = body.tags

    updated = await ms.update_entry(
        id=entry_id,
        content=body.content,
        metadata=metadata,
    )
    if updated is None:
        raise HTTPException(404, f"Memory entry {entry_id} not found")
    _invalidate_search(executor)
    return MemoryEntryResponse(**updated)


@router.delete("/memory/{entry_id}", status_code=204)
async def delete_memory(
    entry_id: int,
    executor: ToolExecutor = Depends(get_executor),
) -> None:
    ms = _get_memory_store(executor)
    deleted = await ms.delete_entry(entry_id)
    if not deleted:
        raise HTTPException(404, f"Memory entry {entry_id} not found")
    _invalidate_search(executor)


@router.post("/memory/search", response_model=MemorySearchResponse)
async def search_memory(
    body: MemorySearchRequest,
    executor: ToolExecutor = Depends(get_executor),
) -> MemorySearchResponse:
    if not body.query:
        raise HTTPException(422, "Query is required")
    se = _get_search_engine(executor)
    results = await se.search(
        query=body.query,
        type=body.type,
        types=body.types,
        scope=body.scope,
        tags=body.tags,
        limit=body.limit,
    )
    return MemorySearchResponse(
        results=[MemorySearchResult(**r) for r in results]
    )

"""Session CRUD endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.store.store import StoreNotFound
from src.tools.executor import ToolExecutor
from src.web.dependencies import get_executor
from src.web.schemas import (
    CreateSessionRequest,
    MessageListResponse,
    MessageResponse,
    SessionListResponse,
    SessionResponse,
)

router = APIRouter()


def _get_chat_store(executor: ToolExecutor) -> Any:
    store = executor.store
    if store is None:
        raise HTTPException(500, "Store not initialized")
    chat = store.chat
    if chat is None:
        raise HTTPException(500, "ChatStore not initialized")
    return chat


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    limit: int = Query(50, ge=1, le=500),
    cursor: str | None = Query(None),
    executor: ToolExecutor = Depends(get_executor),
) -> SessionListResponse:
    chat = _get_chat_store(executor)
    resp = await chat.list_sessions(limit=limit, cursor=cursor)
    sessions = [SessionResponse(**s) for s in resp.get("sessions", [])]
    return SessionListResponse(
        sessions=sessions,
        cursor=resp.get("cursor"),
    )


@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(
    body: CreateSessionRequest,
    executor: ToolExecutor = Depends(get_executor),
) -> SessionResponse:
    chat = _get_chat_store(executor)
    resp = await chat.create_session(title=body.title)
    return SessionResponse(**resp)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    executor: ToolExecutor = Depends(get_executor),
) -> SessionResponse:
    chat = _get_chat_store(executor)
    session = await chat.get_session(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found")
    return SessionResponse(**session)


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    executor: ToolExecutor = Depends(get_executor),
) -> None:
    chat = _get_chat_store(executor)
    try:
        await chat.delete_session(rkey=session_id)
    except StoreNotFound:
        raise HTTPException(404, f"Session '{session_id}' not found")


@router.get("/sessions/{session_id}/messages", response_model=MessageListResponse)
async def list_messages(
    session_id: str,
    limit: int = Query(100, ge=1, le=500),
    cursor: str | None = Query(None),
    executor: ToolExecutor = Depends(get_executor),
) -> MessageListResponse:
    chat = _get_chat_store(executor)
    # verify session exists so a missing session returns 404, not an empty list
    session = await chat.get_session(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found")
    resp = await chat.list_messages(
        session_id=session_id, limit=limit, cursor=cursor
    )
    messages = [MessageResponse(**m) for m in resp.get("messages", [])]
    return MessageListResponse(
        messages=messages,
        cursor=resp.get("cursor"),
    )

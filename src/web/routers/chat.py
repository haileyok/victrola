"""SSE chat endpoint — streams agent events via Server-Sent Events."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.agent.agent import Agent
from src.agent.conversation import ConversationManager, maybe_generate_session_title
from src.tools.executor import ToolExecutor
from src.web.dependencies import get_agent, get_conversation_manager, get_executor
from src.web.schemas import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter()

# Per-session locks to serialize concurrent chat requests on the same session.
# Without this, two simultaneous requests could interleave save/load/agent calls
# and corrupt the conversation history.
_session_locks: dict[str, asyncio.Lock] = {}
# Guard dict mutation itself so check+create is atomic across concurrent requests.
_locks_guard = asyncio.Lock()


async def _get_session_lock(session_id: str) -> asyncio.Lock:
    async with _locks_guard:
        if session_id not in _session_locks:
            _session_locks[session_id] = asyncio.Lock()
        return _session_locks[session_id]


async def _release_session_lock(session_id: str, lock: asyncio.Lock) -> None:
    """Release the lock and prune it from the dict if no one is waiting."""
    async with _locks_guard:
        lock.release()
        # Only remove if no other task is waiting on it
        if not lock.locked():
            _session_locks.pop(session_id, None)


def _sse(event: str, data: dict[str, Any] | None = None) -> str:
    """Format a Server-Sent Event wire string."""
    payload = json.dumps(data or {})
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/sessions/{session_id}/chat")
async def chat(
    session_id: str,
    body: ChatRequest,
    agent: Agent = Depends(get_agent),
    executor: ToolExecutor = Depends(get_executor),
    conv_manager: ConversationManager = Depends(get_conversation_manager),
) -> StreamingResponse:
    # verify session exists
    store = executor.store
    if store is None or store.chat is None:
        raise HTTPException(500, "Store not initialized")
    session = await store.chat.get_session(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found")

    user_text = body.message.strip()
    if not user_text:
        raise HTTPException(400, "Message cannot be empty")

    # Atomically acquire the per-session lock BEFORE returning the response.
    # This ensures the 409 check and lock acquisition are not racy.
    async with _locks_guard:
        if session_id not in _session_locks:
            _session_locks[session_id] = asyncio.Lock()
        lock = _session_locks[session_id]
        if lock.locked():
            raise HTTPException(409, "A chat is already in progress for this session")
        await lock.acquire()

    async def event_stream():
        try:
            # 1. load full conversation history first — agent.chat() will
            # append the new user turn to this list in place
            try:
                conversation = await conv_manager.load_session(session_id)
            except Exception:
                logger.exception("Failed to load session history")
                conversation = []

            # 2. save the user message to the store
            try:
                await conv_manager.save_message(
                    session_id, {"role": "user", "content": user_text}
                )
            except Exception:
                logger.exception("Failed to persist user message")

            # 3. queue for relaying agent events
            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

            async def on_event(event):
                await queue.put({"kind": event.kind, "data": event.data})

            # 4. run agent.chat as a background task — agent.chat() appends
            # the user_message to the conversation list internally
            chat_task = asyncio.create_task(
                agent.chat(
                    user_text,
                    conversation=conversation,
                    on_event=on_event,
                    images=body.images,
                )
            )

            try:
                # 6. relay events as they arrive
                while True:
                    # check if chat_task finished while queue is empty
                    if chat_task.done():
                        # drain any remaining queued events
                        while not queue.empty():
                            item = queue.get_nowait()
                            if item is not None:
                                yield _sse(item["kind"], item["data"])
                        break

                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue

                    if item is None:
                        break
                    yield _sse(item["kind"], item["data"])

                # 7. get the result (response text or exception)
                try:
                    response = await chat_task
                except Exception as e:
                    logger.exception("Agent chat failed")
                    yield _sse("error", {"message": str(e)})
                    yield _sse("done", {})
                    return

                # 8. synthesize response event
                yield _sse("response", {"text": response})

                # 9. save assistant response
                if response:
                    try:
                        await conv_manager.save_message(
                            session_id,
                            {"role": "assistant", "content": response},
                        )
                    except Exception:
                        logger.exception("Failed to persist assistant message")

                    # 10. trigger title generation
                    try:
                        await maybe_generate_session_title(
                            store, session_id, executor.llm_client
                        )
                    except Exception:
                        logger.exception("Failed to auto-generate session title")

                # 11. done
                yield _sse("done", {})
            finally:
                if not chat_task.done():
                    chat_task.cancel()
                    try:
                        await chat_task
                    except (asyncio.CancelledError, Exception):
                        pass
        finally:
            await _release_session_lock(session_id, lock)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

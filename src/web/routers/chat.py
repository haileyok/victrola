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

# Per-session in-flight tracking to serialize concurrent chat requests on the
# same session. Without this, two simultaneous requests could interleave
# save/load/agent calls and corrupt the conversation history.
#
# The session_id is added to _in_flight only when the event_stream generator
# actually begins iterating — never before the StreamingResponse is returned.
# This ensures a client that disconnects before the body is consumed cannot
# leak the in-flight marker (which would permanently 409 the session).
_in_flight: set[str] = set()
# Guard set mutation itself so check+add is atomic across concurrent requests.
_locks_guard = asyncio.Lock()


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

    # Early peek: if a chat is already in-flight for this session, reject
    # immediately. The authoritative check happens inside the generator to
    # close the tiny race this peek can miss.
    async with _locks_guard:
        if session_id in _in_flight:
            raise HTTPException(409, "A chat is already in progress for this session")

    async def event_stream():
        # Acquire the in-flight marker inside the generator so a client that
        # disconnects before the body is consumed never leaks it.
        async with _locks_guard:
            if session_id in _in_flight:
                yield _sse("error", {"message": "A chat is already in progress for this session"})
                return
            _in_flight.add(session_id)
        try:
            # 1. load full conversation history first — agent.chat() will
            # append the new user turn to this list in place
            try:
                conversation, msg_ids = await conv_manager.load_session_with_ids(
                    session_id, drop_current_user_tail=False
                )
            except Exception:
                logger.exception("Failed to load session history")
                conversation = []
                msg_ids = []

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

            # Persist compaction checkpoints so the summary is reused on
            # reload instead of re-summarizing from scratch each turn.
            async def on_compact(summary: str, split_idx: int) -> None:
                if 0 < split_idx <= len(msg_ids):
                    last_id = msg_ids[split_idx - 1]
                    if last_id >= 0:
                        try:
                            await store.chat.set_compaction_checkpoint(
                                session_id, last_id, summary
                            )
                        except Exception:
                            logger.exception("Failed to persist compaction checkpoint")

            # Persist each structured message (assistant turns and tool-result
            # turns) as the agent produces it, so the agent's tool history
            # survives across turns instead of only the final text reply.
            async def on_message(message: dict[str, Any]) -> None:
                try:
                    await conv_manager.save_message(session_id, message)
                except Exception:
                    logger.exception("Failed to persist conversation message")

            # 4. run agent.chat as a background task — agent.chat() appends
            # the user_message to the conversation list internally
            chat_task = asyncio.create_task(
                agent.chat(
                    user_text,
                    conversation=conversation,
                    on_event=on_event,
                    on_compact=on_compact,
                    on_message=on_message,
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

                # 9. The structured turn (assistant + tool-result messages) was
                # persisted incrementally via on_message during agent.chat();
                # here we only auto-generate a session title.
                if response:
                    try:
                        await maybe_generate_session_title(
                            store, session_id, executor.llm_client
                        )
                    except Exception:
                        logger.exception("Failed to auto-generate session title")

                # 10. done
                yield _sse("done", {})
            finally:
                if not chat_task.done():
                    chat_task.cancel()
                    try:
                        await chat_task
                    except (asyncio.CancelledError, Exception):
                        pass
        finally:
            async with _locks_guard:
                _in_flight.discard(session_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

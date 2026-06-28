import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.store.store import Store
    from src.tools.registry import ToolContext

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation persistence and compaction via the local store."""

    def __init__(
        self,
        ctx: "ToolContext",
        llm_client: Any | None = None,
    ) -> None:
        self._ctx = ctx
        self._llm = llm_client

    @property
    def _store(self) -> "Store":
        return self._ctx.store

    async def save_message(self, session_id: str, message: dict[str, Any]) -> None:
        """Save a message to the session as a chat message record.

        Structured content (lists of tool_use / tool_result / text blocks) is
        embedded directly in the payload so a single ``json.loads`` on load
        restores it as a list, keeping multi-block turns intact across reloads.
        Pre-serializing the list here would double-encode it, and the agent
        would later see a JSON string instead of structured blocks.
        """
        role = message.get("role", "user")
        content = message.get("content", "")

        # Coerce only exotic content to a string; strings and block lists are
        # embedded as-is so the round-trip preserves structure.
        if not isinstance(content, (str, list)):
            content = str(content)

        payload = json.dumps({"role": role, "content": content}, default=str)

        if self._store.chat is None:
            raise RuntimeError("ChatStore is not initialized")
        await self._store.chat.create_message(
            session_id=session_id,
            sender=role,
            content=payload,
        )

    async def load_session(self, session_id: str) -> list[dict[str, Any]]:
        """Load all messages for a session, paginating through all records."""
        messages, _ = await self.load_session_with_ids(session_id)
        return messages

    async def load_session_with_ids(
        self, session_id: str, drop_current_user_tail: bool = True
    ) -> tuple[list[dict[str, Any]], list[int]]:
        """Load messages for a session with compaction checkpoint awareness.

        Returns ``(messages, msg_ids)`` where ``msg_ids[i]`` is the store row
        ID for ``messages[i]``, or ``-1`` for the synthetic summary message
        prepended when a compaction checkpoint exists.

        When a checkpoint exists, only the latest summary + messages after the
        checkpoint are loaded — raw messages covered by the checkpoint are
        skipped (they remain in the database for auditability).

        ``drop_current_user_tail``: when True (default, for save-before-load
        surfaces like Discord/Signal), drops the trailing user message since
        the caller just saved it and ``agent.chat()`` will re-append it.
        Set to False for load-before-save surfaces (web router) where the
        tail may be a legitimately unanswered user message from a prior turn.
        """
        if self._store.chat is None:
            raise RuntimeError("ChatStore is not initialized")

        checkpoint = await self._store.chat.get_compaction_checkpoint(session_id)

        messages: list[dict[str, Any]] = []
        msg_ids: list[int] = []

        if checkpoint:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"[Conversation summary of earlier messages]\n\n"
                        f"{checkpoint['summary']}"
                    ),
                }
            )
            msg_ids.append(-1)
            after_id = checkpoint["compacted_up_to_msg_id"]
        else:
            after_id = 0

        cursor: str | None = None
        while True:
            data = await self._store.chat.list_messages(
                session_id=session_id,
                limit=100,
                cursor=cursor,
                after_id=after_id,
            )
            records = data.get("messages", [])
            if not records:
                break
            for record in records:
                content_str = record.get("content", "{}")
                try:
                    msg = json.loads(content_str)
                    messages.append(msg)
                except (json.JSONDecodeError, TypeError):
                    messages.append({"role": "user", "content": content_str})
                msg_ids.append(record["id"])
            cursor = data.get("cursor")
            if not cursor:
                break

        # Drop the most recent user message — agent.chat() will re-append it.
        # Only applies to save-before-load surfaces (Discord, Signal) where
        # the just-saved user message is at the tail. For load-before-save
        # surfaces (web router), the tail may be an unanswered user message
        # from a prior turn that should stay in context.
        if drop_current_user_tail and messages and messages[-1].get("role") == "user":
            messages = messages[:-1]
            msg_ids = msg_ids[:-1]

        return messages, msg_ids


# Min user-message count in a session before we auto-title it.
TITLE_GEN_MIN_USER_MESSAGES = 2
# After a title exists, don't overwrite it.
TITLE_MAX_LEN = 80


async def maybe_generate_session_title(
    store: "Store",
    session_id: str,
    llm_client: Any | None,
) -> str | None:
    """If the session has no title yet and enough messages to summarize, ask
    the sub-agent LLM for a concise title and persist it. Returns the title
    if one was generated, else None.

    No-ops when: the session doesn't exist, already has a title, lacks enough
    user messages, or no sub-agent LLM is configured.
    """
    if llm_client is None:
        return None
    if store.chat is None:
        raise RuntimeError("ChatStore is not initialized")

    session = await store.chat.get_session(session_id)
    if session is None or session.get("title"):
        return None

    # count user messages and sample the first couple of turns
    msgs = await store.chat.list_messages(session_id=session_id, limit=20)
    records = msgs.get("messages", [])
    user_count = sum(1 for r in records if r.get("sender") == "user")
    if user_count < TITLE_GEN_MIN_USER_MESSAGES:
        return None

    transcript_parts: list[str] = []
    for r in records[:10]:
        sender = r.get("sender", "?")
        content = r.get("content", "")
        try:
            parsed = json.loads(content)
            text = parsed.get("content", "")
            if isinstance(text, list):
                text = " ".join(
                    b.get("text", "") for b in text
                    if isinstance(b, dict) and b.get("type") == "text"
                )
        except (json.JSONDecodeError, TypeError):
            text = str(content)
        text = str(text)[:500]
        transcript_parts.append(f"{sender}: {text}")
    transcript = "\n".join(transcript_parts)

    try:
        title = await llm_client.complete(
            (
                "Summarize the topic of this short conversation as a concise title "
                "(5-8 words, no quotes, no trailing punctuation, plain text only). "
                "Respond with only the title.\n\n" + transcript
            ),
            system="You write short, descriptive chat titles.",
            max_tokens=40,
        )
    except Exception:
        logger.exception("title generation failed for session %s", session_id)
        return None

    title = (title or "").strip().strip('"').strip("'")
    # take first line only in case the model goes long
    title = title.split("\n", 1)[0].strip()
    if not title:
        return None
    if len(title) > TITLE_MAX_LEN:
        title = title[:TITLE_MAX_LEN].rstrip() + "..."

    try:
        if store.chat is None:
            raise RuntimeError("ChatStore is not initialized")
        await store.chat.set_title(session_id, title)
    except Exception:
        logger.exception("failed to persist title for session %s", session_id)
        return None

    logger.info("Generated title for session %s: %s", session_id, title)
    return title

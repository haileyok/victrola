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
        """Save a message to the session as a chat message record."""
        role = message.get("role", "user")
        content = message.get("content", "")

        # serialize structured content (tool results, multi-block assistant messages)
        if isinstance(content, list):
            content = json.dumps(content, default=str)
        elif not isinstance(content, str):
            content = str(content)

        payload = json.dumps({"role": role, "content": content}, default=str)

        assert self._store.chat is not None
        await self._store.chat.create_message(
            session_id=session_id,
            sender=role,
            content=payload,
        )

    async def load_session(self, session_id: str) -> list[dict[str, Any]]:
        """Load all messages for a session, paginating through all records."""
        messages: list[dict[str, Any]] = []
        cursor: str | None = None

        assert self._store.chat is not None
        while True:
            data = await self._store.chat.list_messages(
                session_id=session_id,
                limit=100,
                cursor=cursor,
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

            cursor = data.get("cursor")
            if not cursor:
                break

        return messages


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
    assert store.chat is not None

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
        assert store.chat is not None
        await store.chat.set_title(session_id, title)
    except Exception:
        logger.exception("failed to persist title for session %s", session_id)
        return None

    logger.info("Generated title for session %s: %s", session_id, title)
    return title

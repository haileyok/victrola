"""RAG recall service — retrieves relevant memories per turn.

Before each LLM call, the user's message is embedded and relevant
`episodic` + `factual` entries are retrieved via hybrid search and
injected into the system prompt.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Maximum characters per recalled memory entry
_MAX_ENTRY_CHARS = 500


class RecallService:
    """Wraps a SearchEngine to provide RAG recall for system prompt injection."""

    def __init__(self, search_engine: Any) -> None:
        self._search = search_engine

    async def recall(self, user_message: str, limit: int = 6) -> str:
        """Retrieve relevant episodic + factual memories for the user message.

        Returns a bullet block (no header) suitable for the system prompt.
        The header is added by the caller via RELEVANT_MEMORIES_TEMPLATE.
        Returns empty string if no memories match.
        """
        if not user_message or not user_message.strip():
            return ""

        try:
            results = await self._search.search(
                query=user_message,
                types=["episodic", "factual"],
                limit=limit,
            )
        except Exception:
            logger.warning("Memory recall search failed; continuing without RAG", exc_info=True)
            return ""

        if not results:
            return ""

        lines: list[str] = []
        for r in results:
            content = r.get("content", "")
            # Truncate long entries
            if len(content) > _MAX_ENTRY_CHARS:
                content = content[:_MAX_ENTRY_CHARS].rsplit(" ", 1)[0] + "…"
            entry_type = r.get("type", "unknown")
            lines.append(f"- [{entry_type}] {content}")

        return "\n".join(lines)

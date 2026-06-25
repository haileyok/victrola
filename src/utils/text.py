"""Shared text utilities used across bot surfaces."""

DEFAULT_CHUNK_LIMIT = 1900


def _chunk(text: str, limit: int = DEFAULT_CHUNK_LIMIT) -> list[str]:
    """Split text into <=limit-char chunks, breaking on newlines when possible."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:  # no good break point — hard split
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks

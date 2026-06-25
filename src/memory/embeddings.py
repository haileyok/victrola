"""Ollama embedding client for the memory system.

Uses the local Ollama /api/embed endpoint to generate float32 embedding
vectors via the `nomic-embed-text` model (768 dims by default).
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]
    logger.warning("numpy not available — embedding client will not work")


class EmbeddingClient:
    """HTTP client for Ollama embeddings.

    Returns float32 BLOBs (bytes) directly — the same format stored in
    the `embedding` column of `memory_entries`.
    """

    def __init__(self, endpoint: str, model: str, dimensions: int = 768) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._dimensions = dimensions
        self._http = httpx.AsyncClient(timeout=30.0)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, text: str) -> bytes:
        """Embed a single text. Returns float32 BLOB."""
        if np is None:
            raise RuntimeError("numpy not available — cannot generate embeddings")
        resp = await self._http.post(
            f"{self._endpoint}/api/embed",
            json={"model": self._model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = data["embeddings"][0]
        arr = np.array(embedding, dtype=np.float32)
        return arr.tobytes()

    async def embed_batch(self, texts: list[str]) -> list[bytes]:
        """Embed multiple texts in one API call. Returns list of float32 BLOBs."""
        if np is None:
            raise RuntimeError("numpy not available — cannot generate embeddings")
        if not texts:
            return []
        resp = await self._http.post(
            f"{self._endpoint}/api/embed",
            json={"model": self._model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for embedding in data["embeddings"]:
            arr = np.array(embedding, dtype=np.float32)
            results.append(arr.tobytes())
        return results

    async def close(self) -> None:
        await self._http.aclose()

"""Hybrid search engine for the memory system.

Combines FTS5 keyword search (with LIKE fallback) and NumPy cosine
similarity over float32 embedding BLOBs. Results are merged, scored,
and deduplicated.
"""

import json
import logging
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]
    logger.warning("numpy not available — search degrades to keyword-only")


class SearchEngine:
    """Hybrid search: FTS5 keyword + vector cosine, merged and ranked."""

    def __init__(
        self,
        store: Any,
        embedding_client: Any | None = None,
    ) -> None:
        self._store = store
        self._embedding_client = embedding_client
        self._vector_cache: Any | None = None  # (N, dims) np.ndarray
        self._vector_ids: list[int] | None = None  # row IDs aligned with cache
        self._vector_types: list[str] | None = None  # type per row, aligned with cache
        self._db: aiosqlite.Connection = store._db

    def invalidate_cache(self) -> None:
        """Mark the vector cache as stale. Called after writes."""
        self._vector_cache = None
        self._vector_ids = None
        self._vector_types = None

    async def _refresh_cache(self) -> None:
        """Load all embeddings into an in-memory numpy matrix + ID list."""
        if np is None:
            self._vector_cache = None
            self._vector_ids = None
            self._vector_types = None
            return

        pairs = await self._store.get_all_embeddings()
        if not pairs:
            self._vector_cache = None
            self._vector_ids = None
            self._vector_types = None
            return

        ids: list[int] = []
        vectors: list[Any] = []
        types: list[str] = []

        for entry_id, emb_bytes in pairs:
            try:
                vec = np.frombuffer(emb_bytes, dtype=np.float32)
                if len(vec) == 0:
                    continue
                ids.append(entry_id)
                vectors.append(vec)
            except Exception:
                logger.warning("Failed to parse embedding for entry %d", entry_id)
                continue

        if not ids:
            self._vector_cache = None
            self._vector_ids = None
            self._vector_types = None
            return

        # Fetch types for the cached IDs (for type filtering)
        placeholders = ",".join("?" * len(ids))
        cur = await self._db.execute(
            f"SELECT id, type FROM memory_entries WHERE id IN ({placeholders})",
            ids,
        )
        type_map: dict[int, str] = {}
        for row in await cur.fetchall():
            type_map[row[0]] = row[1]
        types = [type_map.get(i, "") for i in ids]

        self._vector_cache = np.vstack(vectors)
        self._vector_ids = ids
        self._vector_types = types

    async def search(
        self,
        query: str,
        type: str | None = None,
        types: list[str] | None = None,
        scope: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Hybrid search: FTS5 keyword + vector cosine, merged and ranked.

        type: filter to a single type (e.g. 'factual').
        types: filter to multiple types (e.g. ['episodic', 'factual']).
               If both are given, types takes precedence.
        """
        limit = max(1, min(limit, 100))

        # Resolve type filter
        type_filter: list[str] | None = None
        if types:
            type_filter = types
        elif type:
            type_filter = [type]

        # --- keyword search ---
        keyword_results = await self._keyword_search(
            query, type_filter, scope, tags, limit
        )

        # --- vector search ---
        vector_results = await self._vector_search(
            query, type_filter, scope, tags, limit
        )

        # --- merge ---
        return await self._merge_results(keyword_results, vector_results, limit)

    async def _keyword_search(
        self,
        query: str,
        type_filter: list[str] | None,
        scope: str | None,
        tags: list[str] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """FTS5 keyword search (or LIKE fallback). Returns list of {id, rank}."""
        results: list[dict[str, Any]] = []

        # Build WHERE clause for type/scope/tag filters
        where_parts: list[str] = []
        params: list[Any] = []

        if type_filter:
            placeholders = ",".join("?" * len(type_filter))
            where_parts.append(f"e.type IN ({placeholders})")
            params.extend(type_filter)

        if scope:
            where_parts.append("e.scope = ?")
            params.append(scope)

        if tags:
            or_clauses = " OR ".join(
                "EXISTS (SELECT 1 FROM json_each(e.metadata, '$.tags') WHERE value = ?)"
                for _ in tags
            )
            where_parts.append(f"({or_clauses})")
            params.extend(tags)

        where_clause = (" AND " + " AND ".join(where_parts)) if where_parts else ""

        if self._store._fts5_available:
            # FTS5 path — join FTS table to memory_entries for filtering
            fts_query = self._sanitize_fts_query(query)
            if not fts_query:
                return []

            sql = (
                "SELECT e.id, f.rank "
                "FROM memory_entries_fts f "
                "JOIN memory_entries e ON e.id = f.rowid "
                f"WHERE f.memory_entries_fts MATCH ?{where_clause} "
                "ORDER BY f.rank LIMIT ?"
            )
            params_full = [fts_query] + params + [limit]
        else:
            # LIKE fallback — escape SQL wildcards in the query
            escaped_query = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            sql = (
                "SELECT e.id, 0 as rank "
                "FROM memory_entries e "
                "WHERE e.content LIKE ? ESCAPE '\\' "
                f"{where_clause} "
                "LIMIT ?"
            )
            params_full = [f"%{escaped_query}%"] + params + [limit]

        try:
            cur = await self._db.execute(sql, params_full)
            rows = await cur.fetchall()
            for r in rows:
                results.append({"id": r[0], "rank": r[1]})
        except Exception:
            logger.warning("Keyword search failed", exc_info=True)

        return results

    async def _vector_search(
        self,
        query: str,
        type_filter: list[str] | None,
        scope: str | None,
        tags: list[str] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Vector cosine similarity search. Returns list of {id, score}."""
        if np is None or self._embedding_client is None:
            return []

        # Embed the query
        try:
            query_bytes = await self._embedding_client.embed(query)
            query_vec = np.frombuffer(query_bytes, dtype=np.float32)
        except Exception:
            logger.warning("Failed to embed query for vector search", exc_info=True)
            return []

        # Refresh cache if needed
        if self._vector_cache is None:
            await self._refresh_cache()
        if self._vector_cache is None or self._vector_ids is None:
            return []

        # Validate dimension match between query embedding and cached vectors
        if len(query_vec) != self._vector_cache.shape[1]:
            logger.warning(
                "Query embedding dimensions (%d) != cached vector dimensions (%d) — "
                "skipping vector search",
                len(query_vec),
                self._vector_cache.shape[1],
            )
            return []

        # Filter cache by type if needed
        cache_mask = np.ones(len(self._vector_ids), dtype=bool)
        if type_filter:
            type_set = set(type_filter)
            for i, t in enumerate(self._vector_types or []):
                if t not in type_set:
                    cache_mask[i] = False

        # Filter cache by scope/tags — need to query DB for these
        if scope or tags:
            filtered_ids = [
                self._vector_ids[i]
                for i in range(len(self._vector_ids))
                if cache_mask[i]
            ]
            if not filtered_ids:
                return []

            # Query DB for scope/tag match within the filtered IDs
            scope_ids = await self._filter_ids_by_scope_tags(
                filtered_ids, scope, tags
            )
            if not scope_ids:
                return []

            scope_id_set = set(scope_ids)
            new_mask = np.zeros(len(self._vector_ids), dtype=bool)
            for i in range(len(self._vector_ids)):
                if self._vector_ids[i] in scope_id_set:
                    new_mask[i] = True
            cache_mask = new_mask

        filtered_indices = np.where(cache_mask)[0]
        if len(filtered_indices) == 0:
            return []

        filtered_matrix = self._vector_cache[filtered_indices]
        filtered_ids = [self._vector_ids[i] for i in filtered_indices]

        # Compute cosine similarity
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        matrix_norms = np.linalg.norm(filtered_matrix, axis=1)
        # Avoid division by zero
        valid = matrix_norms > 0
        if not valid.any():
            return []

        scores = np.zeros(len(filtered_indices), dtype=np.float32)
        dot_products = filtered_matrix @ query_vec
        scores[valid] = dot_products[valid] / (matrix_norms[valid] * query_norm)

        # Top-k
        k = min(limit, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        # Sort by score descending
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            results.append({
                "id": filtered_ids[idx],
                "score": float(scores[idx]),
            })

        return results

    async def _filter_ids_by_scope_tags(
        self,
        ids: list[int],
        scope: str | None,
        tags: list[str] | None,
    ) -> list[int]:
        """Filter a list of entry IDs by scope and tags."""
        where_parts: list[str] = []
        params: list[Any] = list(ids)

        placeholders = ",".join("?" * len(ids))
        where_parts.append(f"id IN ({placeholders})")

        if scope:
            where_parts.append("scope = ?")
            params.append(scope)

        if tags:
            or_clauses = " OR ".join(
                "EXISTS (SELECT 1 FROM json_each(metadata, '$.tags') WHERE value = ?)"
                for _ in tags
            )
            where_parts.append(f"({or_clauses})")
            params.extend(tags)

        sql = f"SELECT id FROM memory_entries WHERE {' AND '.join(where_parts)}"
        cur = await self._db.execute(sql, params)
        rows = await cur.fetchall()
        return [r[0] for r in rows]

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize a query string for FTS5 MATCH.

        Wraps individual tokens in double quotes to prevent FTS5 syntax
        errors from special characters like -, :, etc. Tokens are joined
        with OR so that any matching token returns results (FTS5 treats
        a space-separated list as implicit AND by default).
        """
        tokens = query.strip().split()
        if not tokens:
            return ""
        quoted = ['"' + t.replace('"', '""') + '"' for t in tokens]
        return " OR ".join(quoted)

    @staticmethod
    def _normalize_scores(
        results: list[dict[str, Any]], score_key: str
    ) -> dict[int, float]:
        """Normalize scores to [0, 1] using min-max scaling.

        For FTS5 ranks (negative BM25): more negative = better.
        We invert so higher = better after normalization.
        """
        if not results:
            return {}

        scores = [r[score_key] for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if min_score == max_score:
            # Single result or all same score → map to 1.0
            return {r["id"]: 1.0 for r in results}

        normalized: dict[int, float] = {}
        for r in results:
            # For ranks (negative BM25): (max - rank) / (max - min)
            # maps most negative (best) → 1.0, least negative (worst) → 0.0
            normalized[r["id"]] = (max_score - r[score_key]) / (max_score - min_score)

        return normalized

    @staticmethod
    def _normalize_vector_scores(
        results: list[dict[str, Any]]
    ) -> dict[int, float]:
        """Normalize vector cosine scores to [0, 1]."""
        if not results:
            return {}

        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if min_score == max_score:
            return {r["id"]: 1.0 for r in results}

        normalized: dict[int, float] = {}
        for r in results:
            normalized[r["id"]] = (r["score"] - min_score) / (max_score - min_score)

        return normalized

    async def _merge_results(
        self,
        keyword_results: list[dict[str, Any]],
        vector_results: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Merge keyword and vector results with combined scores."""
        keyword_scores = self._normalize_scores(keyword_results, "rank")
        vector_scores = self._normalize_vector_scores(vector_results)

        all_ids: set[int] = set()
        all_ids.update(keyword_scores.keys())
        all_ids.update(vector_scores.keys())

        if not all_ids:
            return []

        # Fetch entry data for all result IDs
        id_list = list(all_ids)
        placeholders = ",".join("?" * len(id_list))
        cur = await self._db.execute(
            "SELECT id, type, scope, content FROM memory_entries "
            f"WHERE id IN ({placeholders})",
            id_list,
        )
        rows = await cur.fetchall()

        entry_map: dict[int, dict[str, Any]] = {}
        for r in rows:
            entry_map[r[0]] = {
                "id": r[0],
                "type": r[1],
                "scope": r[2],
                "content": r[3],
            }

        merged: list[dict[str, Any]] = []
        for entry_id in all_ids:
            entry = entry_map.get(entry_id)
            if entry is None:
                continue

            kw_score = keyword_scores.get(entry_id, 0.0)
            vec_score = vector_scores.get(entry_id, 0.0)
            combined = 0.5 * kw_score + 0.5 * vec_score

            # Determine matched_by
            in_keyword = entry_id in keyword_scores
            in_vector = entry_id in vector_scores
            if in_keyword and in_vector:
                matched_by = "both"
            elif in_keyword:
                matched_by = "keyword"
            else:
                matched_by = "semantic"

            merged.append({
                **entry,
                "score": combined,
                "matched_by": matched_by,
            })

        # Sort by combined score descending
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:limit]

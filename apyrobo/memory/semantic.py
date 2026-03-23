"""
Semantic Memory — key-value store with optional vector similarity search.

Stores facts about the environment: object locations, map information,
known entities, and any other structured knowledge the robot accumulates
during operation.

Two recall modes:
1. **Keyword recall** — substring/token match on keys (no extra deps).
2. **Vector recall** — cosine similarity on numpy embeddings when provided.

Usage::

    from apyrobo.memory.semantic import SemanticStore

    store = SemanticStore()
    store.remember("object:red_box", {"location": (2, 3), "color": "red"})
    store.remember("map:waypoint_a", {"x": 1.0, "y": 4.0})

    result = store.recall("red_box")            # keyword match
    store.forget("object:red_box")

    # With embeddings (numpy required):
    import numpy as np
    embedding = np.array([0.1, 0.9, 0.3])
    store.remember("object:blue_cube", {"location": (5, 6)}, embedding=embedding)
    results = store.recall("cube", query_embedding=np.array([0.1, 0.8, 0.4]))
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# numpy is optional — only needed for vector similarity recall
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False


def _cosine_similarity(a: Any, b: Any) -> float:
    """Cosine similarity between two numpy vectors."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# SemanticStore
# ---------------------------------------------------------------------------

class SemanticStore:
    """
    In-memory key-value fact store with optional TTL and vector similarity.

    Keys are arbitrary strings (convention: ``"category:name"``).
    Values are any JSON-serialisable object.
    Embeddings are optional numpy arrays stored alongside values.

    Args:
        default_ttl_s: Default time-to-live for entries (seconds).
            ``None`` means entries never expire by default.
    """

    def __init__(self, default_ttl_s: float | None = None) -> None:
        # key -> {"value": Any, "embedding": np.ndarray|None, "timestamp": float, "ttl_s": float|None}
        self._store: dict[str, dict[str, Any]] = {}
        self._default_ttl_s = default_ttl_s

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def remember(
        self,
        key: str,
        value: Any,
        embedding: Any | None = None,
        ttl_s: float | None = None,
    ) -> None:
        """
        Store a fact.

        Args:
            key: Unique identifier for this fact (e.g. ``"object:red_box"``).
            value: The fact value — any serialisable data.
            embedding: Optional numpy array for vector similarity recall.
            ttl_s: Time-to-live in seconds; overrides the store default.
                   Pass ``0`` to use the store default; use ``None`` in
                   the default constructor to mean "never expire".
        """
        effective_ttl = ttl_s if ttl_s is not None else self._default_ttl_s
        self._store[key] = {
            "value": value,
            "embedding": embedding,
            "timestamp": time.time(),
            "ttl_s": effective_ttl,
        }
        logger.debug("SemanticStore: remembered %r", key)

    def forget(self, key: str) -> bool:
        """
        Remove a fact.

        Returns True if the key existed and was removed, False otherwise.
        """
        existed = key in self._store
        self._store.pop(key, None)
        if existed:
            logger.debug("SemanticStore: forgot %r", key)
        return existed

    def update(self, key: str, value: Any) -> bool:
        """
        Update the value of an existing fact without resetting its TTL.

        Returns True if the key existed, False otherwise.
        """
        entry = self._store.get(key)
        if entry is None or self._is_expired(entry):
            return False
        entry["value"] = value
        return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Return the value for *key*, or None if missing or expired."""
        entry = self._store.get(key)
        if entry is None or self._is_expired(entry):
            self._store.pop(key, None)
            return None
        return entry["value"]

    def has(self, key: str) -> bool:
        """Return True if *key* exists and is not expired."""
        return self.get(key) is not None

    def keys(self) -> list[str]:
        """Return all non-expired keys."""
        self._expire_stale()
        return sorted(self._store.keys())

    def items(self) -> list[tuple[str, Any]]:
        """Return all non-expired (key, value) pairs."""
        self._expire_stale()
        return [(k, v["value"]) for k, v in self._store.items()]

    # ------------------------------------------------------------------
    # Recall (search)
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        top_k: int = 10,
        query_embedding: Any | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for facts matching *query*.

        If *query_embedding* is provided **and** numpy is available, ranks
        results by cosine similarity against stored embeddings (only entries
        that have embeddings stored participate in vector ranking; keywordonly
        entries are included last if they also match the keyword query).

        Without embeddings, falls back to keyword/token matching on keys.

        Args:
            query: Search string (used for keyword matching).
            top_k: Maximum number of results to return.
            query_embedding: Optional numpy array for vector similarity.

        Returns:
            List of dicts with keys: ``key``, ``value``, ``score``,
            ``match_type`` (``"vector"`` | ``"keyword"``).
        """
        self._expire_stale()
        results: list[dict[str, Any]] = []
        query_lower = query.lower()
        tokens = set(query_lower.split())

        for key, entry in self._store.items():
            if self._is_expired(entry):
                continue

            score = 0.0
            match_type = "keyword"
            stored_embedding = entry.get("embedding")

            # Vector similarity (if both embeddings available)
            if (
                query_embedding is not None
                and stored_embedding is not None
                and _NUMPY_AVAILABLE
            ):
                try:
                    q_arr = np.asarray(query_embedding, dtype=float)
                    s_arr = np.asarray(stored_embedding, dtype=float)
                    score = _cosine_similarity(q_arr, s_arr)
                    match_type = "vector"
                    results.append({
                        "key": key,
                        "value": entry["value"],
                        "score": score,
                        "match_type": match_type,
                    })
                    continue
                except Exception:
                    pass  # fall through to keyword matching

            # Keyword / token matching
            key_lower = key.lower()
            if query_lower in key_lower:
                score = 1.0
            elif tokens and tokens.intersection(set(key_lower.split(":"))):
                score = 0.5
            elif tokens and any(tok in key_lower for tok in tokens):
                score = 0.3

            if score > 0:
                results.append({
                    "key": key,
                    "value": entry["value"],
                    "score": score,
                    "match_type": "keyword",
                })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def _is_expired(self, entry: dict[str, Any]) -> bool:
        ttl = entry.get("ttl_s")
        if ttl is None:
            return False
        return (time.time() - entry["timestamp"]) > ttl

    def _expire_stale(self) -> int:
        """Remove expired entries and return count removed."""
        stale = [k for k, v in self._store.items() if self._is_expired(v)]
        for k in stale:
            del self._store[k]
        return len(stale)

    def expire_stale(self) -> int:
        """Public alias for _expire_stale."""
        return self._expire_stale()

    def clear(self) -> int:
        """Remove all entries. Returns the count removed."""
        count = len(self._store)
        self._store.clear()
        return count

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of non-expired entries."""
        self._expire_stale()
        return len(self._store)

    def __len__(self) -> int:
        return self.count

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __repr__(self) -> str:
        return f"<SemanticStore entries={self.count} ttl={self._default_ttl_s}>"

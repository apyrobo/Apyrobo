"""
Plan Cache — cache and reuse skill graphs for repeated tasks.

Caches successful skill execution plans by a hash of the task description.
On new task requests, the cache is checked first before re-planning with
an LLM — reducing latency and API costs for repeated tasks.

Features:
- SHA-256 hash key from task description (normalised)
- TTL-based entry expiry
- Hit/miss/eviction metrics
- Optional SQLite persistence for durability across restarts
- Thread-safe via lock-free design (dict operations are GIL-protected)

Usage::

    cache = PlanCache(ttl_s=3600)              # in-memory, 1h TTL
    cache = PlanCache(db_path="plans.db")      # SQLite-persisted

    # On successful planning:
    cache.store(task="patrol sector 3", plan=[...])

    # Before re-planning:
    cached = cache.lookup(task="patrol sector 3")
    if cached:
        plan = cached["plan"]
    else:
        plan = agent.plan(task)
        cache.store(task=task, plan=plan)

    # Metrics:
    print(cache.stats())
    # {'hits': 5, 'misses': 3, 'hit_rate': 0.625, 'size': 12}
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TTL = 3600.0  # 1 hour

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS plan_cache (
    cache_key   TEXT PRIMARY KEY,
    task        TEXT NOT NULL,
    plan        TEXT NOT NULL,
    created_at  REAL NOT NULL,
    ttl_s       REAL NOT NULL,
    hit_count   INTEGER NOT NULL DEFAULT 0
);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_hash(task: str) -> str:
    """Normalise and SHA-256 hash a task description."""
    normalised = " ".join(task.strip().lower().split())
    return hashlib.sha256(normalised.encode()).hexdigest()


# ---------------------------------------------------------------------------
# PlanCache
# ---------------------------------------------------------------------------

class PlanCache:
    """
    LRU-like cache for task → skill-plan mappings with TTL-based expiry.

    Args:
        ttl_s: Time-to-live for cached plans (seconds).
        db_path: Optional path for SQLite persistence.  ``None`` → in-memory.
        max_size: Maximum number of entries to retain.  Oldest entries are
            evicted when the limit is reached.
    """

    def __init__(
        self,
        ttl_s: float = _DEFAULT_TTL,
        db_path: str | Path | None = None,
        max_size: int = 1000,
    ) -> None:
        self._ttl_s = ttl_s
        self._max_size = max_size

        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Storage
        if db_path is not None:
            self._mode = "sqlite"
            db_path = Path(db_path).expanduser()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn: sqlite3.Connection | None = sqlite3.connect(
                str(db_path),
                check_same_thread=False,
                isolation_level=None,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_CREATE_TABLE)
        else:
            self._mode = "memory"
            self._conn = None
            # In-memory: key -> {"task": str, "plan": Any, "created_at": float, "ttl_s": float, "hit_count": int}
            self._cache: dict[str, dict[str, Any]] = {}

        logger.info(
            "PlanCache initialised (mode=%s, ttl=%.0fs, max_size=%d)",
            self._mode, self._ttl_s, self._max_size,
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store(
        self,
        task: str,
        plan: Any,
        ttl_s: float | None = None,
    ) -> str:
        """
        Cache a plan for *task*.

        Args:
            task: Task description string (used as cache key after hashing).
            plan: The plan to cache (any JSON-serialisable value).
            ttl_s: TTL override; uses the cache default if not specified.

        Returns the cache key (SHA-256 hex string).
        """
        key = _task_hash(task)
        effective_ttl = ttl_s if ttl_s is not None else self._ttl_s
        now = time.time()

        if self._mode == "sqlite" and self._conn is not None:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO plan_cache
                    (cache_key, task, plan, created_at, ttl_s, hit_count)
                VALUES (?, ?, ?, ?, ?, 0)
                """,
                (key, task, json.dumps(plan), now, effective_ttl),
            )
            self._maybe_evict_sqlite()
        else:
            self._cache[key] = {
                "task": task,
                "plan": plan,
                "created_at": now,
                "ttl_s": effective_ttl,
                "hit_count": 0,
            }
            self._maybe_evict_memory()

        logger.debug("PlanCache: stored plan for %r (key=%s)", task[:50], key[:8])
        return key

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def lookup(self, task: str) -> dict[str, Any] | None:
        """
        Look up a cached plan for *task*.

        Returns a dict with keys ``"plan"``, ``"task"``, ``"created_at"``,
        ``"hit_count"`` on hit, or ``None`` on miss/expiry.
        """
        key = _task_hash(task)

        if self._mode == "sqlite" and self._conn is not None:
            row = self._conn.execute(
                "SELECT * FROM plan_cache WHERE cache_key = ?", (key,)
            ).fetchone()

            if row is None:
                self._misses += 1
                return None

            age = time.time() - row["created_at"]
            if age > row["ttl_s"]:
                self._conn.execute(
                    "DELETE FROM plan_cache WHERE cache_key = ?", (key,)
                )
                self._misses += 1
                return None

            # Update hit count
            self._conn.execute(
                "UPDATE plan_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                (key,),
            )
            self._hits += 1
            return {
                "task": row["task"],
                "plan": json.loads(row["plan"]),
                "created_at": row["created_at"],
                "hit_count": row["hit_count"] + 1,
                "cache_key": key,
            }
        else:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            age = time.time() - entry["created_at"]
            if age > entry["ttl_s"]:
                del self._cache[key]
                self._misses += 1
                return None

            entry["hit_count"] += 1
            self._hits += 1
            return {
                "task": entry["task"],
                "plan": entry["plan"],
                "created_at": entry["created_at"],
                "hit_count": entry["hit_count"],
                "cache_key": key,
            }

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate(self, task: str) -> bool:
        """
        Remove the cached plan for *task*.

        Returns True if an entry was found and removed, False otherwise.
        """
        key = _task_hash(task)
        if self._mode == "sqlite" and self._conn is not None:
            cur = self._conn.execute(
                "DELETE FROM plan_cache WHERE cache_key = ?", (key,)
            )
            removed = cur.rowcount > 0
        else:
            removed = key in self._cache
            self._cache.pop(key, None)

        if removed:
            logger.debug("PlanCache: invalidated %r", task[:50])
        return removed

    def invalidate_all(self) -> int:
        """
        Remove all cached plans.

        Returns the number of entries removed.
        """
        if self._mode == "sqlite" and self._conn is not None:
            cur = self._conn.execute("DELETE FROM plan_cache")
            count = cur.rowcount
        else:
            count = len(self._cache)
            self._cache.clear()
        logger.info("PlanCache: cleared %d entries", count)
        return count

    def expire_stale(self) -> int:
        """
        Remove all expired entries.

        Returns the count removed.
        """
        now = time.time()
        if self._mode == "sqlite" and self._conn is not None:
            cur = self._conn.execute(
                "DELETE FROM plan_cache WHERE (? - created_at) > ttl_s", (now,)
            )
            return cur.rowcount
        else:
            stale = [
                k for k, v in self._cache.items()
                if (now - v["created_at"]) > v["ttl_s"]
            ]
            for k in stale:
                del self._cache[k]
            return len(stale)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def hits(self) -> int:
        """Total cache hits since creation."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total cache misses since creation."""
        return self._misses

    @property
    def evictions(self) -> int:
        """Total entries evicted due to max_size limit."""
        return self._evictions

    @property
    def hit_rate(self) -> float:
        """Hit rate as a fraction in [0, 1]."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def size(self) -> int:
        """Current number of entries (including not-yet-expired)."""
        if self._mode == "sqlite" and self._conn is not None:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM plan_cache"
            ).fetchone()
            return row[0] if row else 0
        return len(self._cache)

    def stats(self) -> dict[str, Any]:
        """Return a summary of cache metrics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": round(self.hit_rate, 4),
            "size": self.size(),
            "mode": self._mode,
            "ttl_s": self._ttl_s,
        }

    # ------------------------------------------------------------------
    # Eviction helpers
    # ------------------------------------------------------------------

    def _maybe_evict_memory(self) -> None:
        """Evict oldest entries when memory cache exceeds max_size."""
        if len(self._cache) <= self._max_size:
            return
        # First, remove expired entries
        self.expire_stale()
        if len(self._cache) <= self._max_size:
            return
        # Evict oldest by created_at
        excess = len(self._cache) - self._max_size
        oldest_keys = sorted(
            self._cache, key=lambda k: self._cache[k]["created_at"]
        )[:excess]
        for k in oldest_keys:
            del self._cache[k]
            self._evictions += 1

    def _maybe_evict_sqlite(self) -> None:
        """Evict oldest SQLite entries when count exceeds max_size."""
        if self._conn is None:
            return
        row = self._conn.execute("SELECT COUNT(*) FROM plan_cache").fetchone()
        current = row[0] if row else 0
        if current <= self._max_size:
            return
        # Evict oldest entries
        excess = current - self._max_size
        self._conn.execute(
            """
            DELETE FROM plan_cache
            WHERE cache_key IN (
                SELECT cache_key FROM plan_cache
                ORDER BY created_at ASC
                LIMIT ?
            )
            """,
            (excess,),
        )
        self._evictions += excess

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection (SQLite mode only)."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __repr__(self) -> str:
        return (
            f"<PlanCache mode={self._mode!r} size={self.size()} "
            f"hit_rate={self.hit_rate:.2%} ttl={self._ttl_s}s>"
        )

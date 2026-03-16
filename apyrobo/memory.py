"""
Agent Memory — episodic log + semantic fact store for long-term autonomy.

VC-02: Provides AgentMemory with two subsystems:
    - EpisodicMemory: records (task, plan, result, timestamp) as episodes
    - SemanticMemory: key-value facts with TTL-based expiry

Memory enables agents to:
    - Remember previous tasks and their outcomes
    - Build a map of object locations and world knowledge
    - Improve planning over time with historical context
    - Persist knowledge across sessions via persist()/load()
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Episodic Memory
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    Records task episodes as structured entries.

    Each episode captures: task description, plan steps, result, and timing.
    Maintains a bounded deque of the N most recent episodes.
    """

    def __init__(self, max_episodes: int = 100) -> None:
        self._episodes: deque[dict[str, Any]] = deque(maxlen=max_episodes)

    def record(self, task: str, plan: list[dict[str, Any]] | None = None,
               result: dict[str, Any] | None = None) -> dict[str, Any]:
        """Record a new episode and return it."""
        episode = {
            "task": task,
            "plan": plan or [],
            "result": result or {},
            "timestamp": time.time(),
        }
        self._episodes.append(episode)
        logger.debug("Recorded episode: %r", task)
        return episode

    def recall_recent(self, n: int = 5) -> list[dict[str, Any]]:
        """Return the N most recent episodes (newest first)."""
        episodes = list(self._episodes)
        return list(reversed(episodes[-n:]))

    def recall_all(self) -> list[dict[str, Any]]:
        """Return all episodes (oldest first)."""
        return list(self._episodes)

    def search(self, query: str) -> list[dict[str, Any]]:
        """Simple keyword search over episode tasks."""
        query_lower = query.lower()
        tokens = set(query_lower.split())
        results = []
        for ep in self._episodes:
            task_lower = ep["task"].lower()
            if query_lower in task_lower or tokens.intersection(set(task_lower.split())):
                results.append(ep)
        return results

    @property
    def count(self) -> int:
        return len(self._episodes)

    def clear(self) -> None:
        self._episodes.clear()

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize to a plain list for persistence."""
        return list(self._episodes)

    def load_from_list(self, data: list[dict[str, Any]]) -> None:
        """Restore from a serialized list."""
        self._episodes.clear()
        for ep in data:
            self._episodes.append(ep)


# ---------------------------------------------------------------------------
# Semantic Memory
# ---------------------------------------------------------------------------

class SemanticMemory:
    """
    Key-value fact store with optional TTL-based expiry.

    Facts are structured as: key -> {"value": Any, "timestamp": float, "ttl_s": float | None}

    Examples:
        set_fact("object:red_box", {"last_seen": (2, 3)}, ttl_s=3600)
        get_fact("object:red_box")  -> {"last_seen": (2, 3)}
    """

    def __init__(self, default_ttl_s: float | None = None) -> None:
        self._facts: dict[str, dict[str, Any]] = {}
        self._default_ttl_s = default_ttl_s

    def set_fact(self, key: str, value: Any, ttl_s: float | None = None) -> None:
        """Store a fact with optional TTL (seconds)."""
        self._facts[key] = {
            "value": value,
            "timestamp": time.time(),
            "ttl_s": ttl_s if ttl_s is not None else self._default_ttl_s,
        }
        logger.debug("Set fact: %s = %r", key, value)

    def get_fact(self, key: str) -> Any:
        """Retrieve a fact's value, returning None if expired or missing."""
        entry = self._facts.get(key)
        if entry is None:
            return None
        if self._is_expired(entry):
            del self._facts[key]
            return None
        return entry["value"]

    def has_fact(self, key: str) -> bool:
        """Check if a fact exists and is not expired."""
        return self.get_fact(key) is not None

    def remove_fact(self, key: str) -> bool:
        """Remove a fact. Returns True if it existed."""
        return self._facts.pop(key, None) is not None

    def expire_stale(self) -> int:
        """Remove all expired facts. Returns count removed."""
        now = time.time()
        stale = [k for k, v in self._facts.items() if self._is_expired(v, now)]
        for k in stale:
            del self._facts[k]
        return len(stale)

    def search(self, prefix: str) -> dict[str, Any]:
        """Return all non-expired facts whose keys start with prefix."""
        self.expire_stale()
        return {
            k: v["value"] for k, v in self._facts.items()
            if k.startswith(prefix) and not self._is_expired(v)
        }

    @property
    def count(self) -> int:
        """Count of non-expired facts."""
        self.expire_stale()
        return len(self._facts)

    def clear(self) -> None:
        self._facts.clear()

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Serialize for persistence."""
        return dict(self._facts)

    def load_from_dict(self, data: dict[str, dict[str, Any]]) -> None:
        """Restore from serialized dict."""
        self._facts = dict(data)

    @staticmethod
    def _is_expired(entry: dict[str, Any], now: float | None = None) -> bool:
        ttl = entry.get("ttl_s")
        if ttl is None:
            return False
        now = now or time.time()
        return (now - entry["timestamp"]) > ttl


# ---------------------------------------------------------------------------
# AgentMemory — unified interface
# ---------------------------------------------------------------------------

class AgentMemory:
    """
    Unified memory system combining episodic and semantic memory.

    Provides:
        - Episode recording and retrieval
        - Semantic fact storage with TTL
        - Context string generation for LLM prompt injection
        - Persistence via persist()/load()
        - Semantic search via recall(query)
    """

    def __init__(self, max_episodes: int = 100,
                 fact_ttl_s: float | None = 3600) -> None:
        self.episodes = EpisodicMemory(max_episodes=max_episodes)
        self.facts = SemanticMemory(default_ttl_s=fact_ttl_s)

    # -- Episode shortcuts --

    def record_episode(self, task: str, plan: list[dict[str, Any]] | None = None,
                       result: dict[str, Any] | None = None) -> dict[str, Any]:
        """Record a task episode."""
        return self.episodes.record(task, plan, result)

    def recall_recent(self, n: int = 5) -> list[dict[str, Any]]:
        """Return N most recent episodes."""
        return self.episodes.recall_recent(n)

    # -- Fact shortcuts --

    def set_fact(self, key: str, value: Any, ttl_s: float | None = None) -> None:
        """Store a semantic fact."""
        self.facts.set_fact(key, value, ttl_s)

    def get_fact(self, key: str) -> Any:
        """Retrieve a semantic fact."""
        return self.facts.get_fact(key)

    # -- Context for LLM --

    def to_context_string(self, n_episodes: int = 5, include_facts: bool = True) -> str:
        """
        Format memory as a string suitable for LLM system prompt injection.

        Returns a human-readable summary of recent episodes and relevant facts.
        """
        parts: list[str] = []

        # Recent episodes
        recent = self.episodes.recall_recent(n_episodes)
        if recent:
            parts.append("## Recent Task History")
            for i, ep in enumerate(recent, 1):
                status = ep.get("result", {}).get("status", "unknown")
                parts.append(f"{i}. Task: {ep['task']} → {status}")
                plan = ep.get("plan", [])
                if plan:
                    steps_str = ", ".join(
                        s.get("skill_id", "?") for s in plan[:5]
                    )
                    parts.append(f"   Plan: {steps_str}")

        # Semantic facts
        if include_facts:
            self.facts.expire_stale()
            facts_dict = {
                k: v["value"] for k, v in self.facts._facts.items()
                if not self.facts._is_expired(v)
            }
            if facts_dict:
                parts.append("")
                parts.append("## Known Facts")
                for key, value in facts_dict.items():
                    parts.append(f"- {key}: {value}")

        return "\n".join(parts) if parts else ""

    # -- Semantic search --

    def recall(self, query: str) -> list[dict[str, Any]]:
        """
        Semantic search over past knowledge.

        Searches both episodes (by task text) and facts (by key prefix).
        Returns a list of matching entries.
        """
        results: list[dict[str, Any]] = []

        # Search episodes
        for ep in self.episodes.search(query):
            results.append({"type": "episode", **ep})

        # Search facts
        matching_facts = self.facts.search(query)
        for key, value in matching_facts.items():
            results.append({"type": "fact", "key": key, "value": value})

        return results

    # -- Persistence --

    def persist(self, path: str | Path) -> None:
        """Save memory state to a JSON file."""
        path = Path(path)
        data = {
            "episodes": self.episodes.to_list(),
            "facts": self.facts.to_dict(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Memory persisted to %s", path)

    def load(self, path: str | Path) -> None:
        """Load memory state from a JSON file."""
        path = Path(path)
        if not path.exists():
            logger.warning("Memory file not found: %s", path)
            return
        with open(path) as f:
            data = json.load(f)
        self.episodes.load_from_list(data.get("episodes", []))
        self.facts.load_from_dict(data.get("facts", {}))
        logger.info("Memory loaded from %s (%d episodes, %d facts)",
                     path, self.episodes.count, self.facts.count)

    def clear(self) -> None:
        """Clear all memory."""
        self.episodes.clear()
        self.facts.clear()

    def __repr__(self) -> str:
        return (
            f"<AgentMemory episodes={self.episodes.count} "
            f"facts={self.facts.count}>"
        )

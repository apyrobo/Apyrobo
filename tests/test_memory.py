"""
Tests for VC-02: Long-term agent memory — episodic + semantic store.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from apyrobo.memory import AgentMemory, EpisodicMemory, SemanticMemory


# ---------------------------------------------------------------------------
# EpisodicMemory tests
# ---------------------------------------------------------------------------

class TestEpisodicMemory:
    def test_record_and_recall(self):
        mem = EpisodicMemory(max_episodes=10)
        mem.record("pick up red box", plan=[{"skill_id": "pick_object"}], result={"status": "completed"})
        mem.record("navigate to room 3", plan=[{"skill_id": "navigate_to"}], result={"status": "completed"})

        recent = mem.recall_recent(5)
        assert len(recent) == 2
        # Newest first
        assert recent[0]["task"] == "navigate to room 3"
        assert recent[1]["task"] == "pick up red box"

    def test_record_includes_timestamp(self):
        mem = EpisodicMemory()
        ep = mem.record("test task")
        assert "timestamp" in ep
        assert isinstance(ep["timestamp"], float)

    def test_max_episodes_bounded(self):
        mem = EpisodicMemory(max_episodes=3)
        for i in range(5):
            mem.record(f"task {i}")
        assert mem.count == 3
        recent = mem.recall_recent(10)
        assert recent[0]["task"] == "task 4"
        assert recent[-1]["task"] == "task 2"

    def test_recall_recent_n(self):
        mem = EpisodicMemory()
        for i in range(10):
            mem.record(f"task {i}")
        recent = mem.recall_recent(3)
        assert len(recent) == 3
        assert recent[0]["task"] == "task 9"

    def test_search(self):
        mem = EpisodicMemory()
        mem.record("pick up red box")
        mem.record("navigate to room 3")
        mem.record("pick up blue box")

        results = mem.search("pick")
        assert len(results) == 2
        assert all("pick" in r["task"] for r in results)

    def test_clear(self):
        mem = EpisodicMemory()
        mem.record("task")
        mem.clear()
        assert mem.count == 0

    def test_serialization_roundtrip(self):
        mem = EpisodicMemory()
        mem.record("task 1", result={"status": "completed"})
        mem.record("task 2", result={"status": "failed"})

        data = mem.to_list()
        mem2 = EpisodicMemory()
        mem2.load_from_list(data)

        assert mem2.count == 2
        assert mem2.recall_recent(1)[0]["task"] == "task 2"


# ---------------------------------------------------------------------------
# SemanticMemory tests
# ---------------------------------------------------------------------------

class TestSemanticMemory:
    def test_set_and_get_fact(self):
        mem = SemanticMemory()
        mem.set_fact("object:red_box", {"last_seen": (2, 3)})
        assert mem.get_fact("object:red_box") == {"last_seen": (2, 3)}

    def test_get_missing_fact(self):
        mem = SemanticMemory()
        assert mem.get_fact("nonexistent") is None

    def test_ttl_expiry_removes_stale_facts(self):
        """TTL expiry removes stale facts."""
        mem = SemanticMemory()
        mem.set_fact("stale", "old_value", ttl_s=0.1)
        assert mem.get_fact("stale") == "old_value"

        time.sleep(0.15)
        assert mem.get_fact("stale") is None

    def test_ttl_not_expired(self):
        mem = SemanticMemory()
        mem.set_fact("fresh", "value", ttl_s=3600)
        assert mem.get_fact("fresh") == "value"

    def test_no_ttl_never_expires(self):
        mem = SemanticMemory()
        mem.set_fact("permanent", "value")
        assert mem.get_fact("permanent") == "value"

    def test_default_ttl(self):
        """Default TTL applies when no per-fact TTL is given."""
        mem = SemanticMemory(default_ttl_s=0.1)
        mem.set_fact("fact", "value")
        assert mem.get_fact("fact") == "value"
        time.sleep(0.15)
        assert mem.get_fact("fact") is None

    def test_has_fact(self):
        mem = SemanticMemory()
        mem.set_fact("key", "value")
        assert mem.has_fact("key") is True
        assert mem.has_fact("other") is False

    def test_remove_fact(self):
        mem = SemanticMemory()
        mem.set_fact("key", "value")
        assert mem.remove_fact("key") is True
        assert mem.remove_fact("key") is False

    def test_expire_stale(self):
        mem = SemanticMemory()
        mem.set_fact("a", 1, ttl_s=0.1)
        mem.set_fact("b", 2, ttl_s=3600)
        time.sleep(0.15)
        removed = mem.expire_stale()
        assert removed == 1
        assert mem.get_fact("b") == 2

    def test_search_by_prefix(self):
        mem = SemanticMemory()
        mem.set_fact("object:red_box", (2, 3))
        mem.set_fact("object:blue_box", (1, 1))
        mem.set_fact("robot:position", (0, 0))

        results = mem.search("object:")
        assert len(results) == 2
        assert "object:red_box" in results
        assert "object:blue_box" in results

    def test_count(self):
        mem = SemanticMemory()
        mem.set_fact("a", 1)
        mem.set_fact("b", 2)
        assert mem.count == 2

    def test_clear(self):
        mem = SemanticMemory()
        mem.set_fact("a", 1)
        mem.clear()
        assert mem.count == 0

    def test_serialization_roundtrip(self):
        mem = SemanticMemory()
        mem.set_fact("key", "value", ttl_s=3600)
        data = mem.to_dict()

        mem2 = SemanticMemory()
        mem2.load_from_dict(data)
        assert mem2.get_fact("key") == "value"


# ---------------------------------------------------------------------------
# AgentMemory tests
# ---------------------------------------------------------------------------

class TestAgentMemory:
    def test_record_and_recall_episode(self):
        """Record episode after execute() — task, steps, result, timing."""
        mem = AgentMemory()
        ep = mem.record_episode(
            task="deliver package",
            plan=[{"skill_id": "navigate_to"}, {"skill_id": "pick_object"}],
            result={"status": "completed", "steps_completed": 2, "steps_total": 2},
        )
        assert ep["task"] == "deliver package"
        assert ep["timestamp"] > 0
        assert len(ep["plan"]) == 2
        assert ep["result"]["status"] == "completed"

        recent = mem.recall_recent(5)
        assert len(recent) == 1
        assert recent[0]["task"] == "deliver package"

    def test_set_and_get_semantic_fact(self):
        """Set semantic fact 'box:red -> (2,3)' after picking; retrieve on next task."""
        mem = AgentMemory()
        mem.set_fact("box:red", (2, 3))
        assert mem.get_fact("box:red") == (2, 3)

    def test_to_context_string_episodes(self):
        """Inject recent episodes into LLM system prompt on next plan()."""
        mem = AgentMemory()
        mem.record_episode("go to room 1", result={"status": "completed"})
        mem.record_episode("pick up box", result={"status": "failed"})

        ctx = mem.to_context_string()
        assert "Recent Task History" in ctx
        assert "go to room 1" in ctx
        assert "pick up box" in ctx

    def test_to_context_string_facts(self):
        mem = AgentMemory()
        mem.set_fact("object:box", {"pos": (1, 2)})
        ctx = mem.to_context_string()
        assert "Known Facts" in ctx
        assert "object:box" in ctx

    def test_to_context_string_empty(self):
        mem = AgentMemory()
        assert mem.to_context_string() == ""

    def test_recall_search(self):
        """recall(query) for semantic search over past knowledge."""
        mem = AgentMemory()
        mem.record_episode("pick up red box")
        mem.set_fact("object:red_box", (2, 3))

        results = mem.recall("red")
        assert any(r["type"] == "episode" for r in results)

    def test_persist_and_load(self):
        """persist() + load() round-trips memory across sessions."""
        mem = AgentMemory()
        mem.record_episode("task 1", result={"status": "completed"})
        mem.set_fact("key", "value", ttl_s=3600)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        mem.persist(tmp_path)

        # Load into fresh memory
        mem2 = AgentMemory()
        mem2.load(tmp_path)

        assert mem2.episodes.count == 1
        assert mem2.recall_recent(1)[0]["task"] == "task 1"
        assert mem2.get_fact("key") == "value"

        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)

    def test_load_missing_file(self):
        """Load from nonexistent path doesn't crash."""
        mem = AgentMemory()
        mem.load("/tmp/nonexistent_apyrobo_memory.json")
        assert mem.episodes.count == 0

    def test_clear(self):
        mem = AgentMemory()
        mem.record_episode("task")
        mem.set_fact("key", "value")
        mem.clear()
        assert mem.episodes.count == 0
        assert mem.facts.count == 0

    def test_repr(self):
        mem = AgentMemory()
        assert "AgentMemory" in repr(mem)


# ---------------------------------------------------------------------------
# Agent integration with memory
# ---------------------------------------------------------------------------

class TestAgentWithMemory:
    def test_agent_with_no_memory_stateless(self):
        """Agent with no memory arg behaves identically to current stateless behavior."""
        from apyrobo.core.robot import Robot
        from apyrobo.skills.agent import Agent

        robot = Robot.discover("mock://turtlebot4")
        agent = Agent(provider="rule")
        assert agent.memory is None

        result = agent.execute("go to (1, 2)", robot)
        assert result.status.value == "completed"
        # No memory recorded
        assert agent.memory is None

    def test_agent_with_memory_records_episode(self):
        """Agent with memory records episode after execute()."""
        from apyrobo.core.robot import Robot
        from apyrobo.skills.agent import Agent

        robot = Robot.discover("mock://turtlebot4")
        memory = AgentMemory()
        agent = Agent(provider="rule", memory=memory)

        result = agent.execute("go to (3, 4)", robot)
        assert result.status.value == "completed"

        # Episode should be recorded
        episodes = memory.recall_recent(5)
        assert len(episodes) == 1
        assert "go to (3, 4)" in episodes[0]["task"]
        assert episodes[0]["result"]["status"] == "completed"

    def test_agent_memory_accumulates(self):
        """Multiple executions accumulate episodes in memory."""
        from apyrobo.core.robot import Robot
        from apyrobo.skills.agent import Agent

        robot = Robot.discover("mock://turtlebot4")
        memory = AgentMemory()
        agent = Agent(provider="rule", memory=memory)

        agent.execute("go to (1, 1)", robot)
        agent.execute("go to (2, 2)", robot)
        agent.execute("report status", robot)

        episodes = memory.recall_recent(10)
        assert len(episodes) == 3

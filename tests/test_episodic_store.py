"""
Tests for apyrobo/memory/episodic.py — EpisodicStore (SQLite-backed).
"""

from __future__ import annotations

import time

import pytest

from apyrobo.memory.episodic import Episode, EpisodicStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    """Fresh in-memory EpisodicStore for each test."""
    return EpisodicStore()  # :memory:


@pytest.fixture
def sample_episode():
    return Episode(
        task="deliver package to room 3",
        robot_id="turtlebot4",
        skills_run=["navigate_to", "pick_object", "place_object"],
        robot_state={"x": 1.0, "y": 2.0, "battery": 80},
        outcome="success",
        task_type="delivery",
        duration_s=12.5,
        metadata={"steps_completed": 3},
    )


# ---------------------------------------------------------------------------
# Episode dataclass
# ---------------------------------------------------------------------------

class TestEpisodeDataclass:
    def test_defaults(self):
        ep = Episode()
        assert ep.task == ""
        assert ep.robot_id == ""
        assert ep.outcome == "unknown"
        assert ep.skills_run == []
        assert ep.robot_state == {}
        assert ep.id is None

    def test_timestamp_auto_set(self):
        before = time.time()
        ep = Episode()
        after = time.time()
        assert before <= ep.timestamp <= after

    def test_to_dict(self, sample_episode):
        d = sample_episode.to_dict()
        assert d["task"] == "deliver package to room 3"
        assert d["robot_id"] == "turtlebot4"
        assert d["outcome"] == "success"
        assert "id" not in d

    def test_to_dict_skills_run(self, sample_episode):
        d = sample_episode.to_dict()
        assert d["skills_run"] == ["navigate_to", "pick_object", "place_object"]


# ---------------------------------------------------------------------------
# EpisodicStore — init
# ---------------------------------------------------------------------------

class TestEpisodicStoreInit:
    def test_memory_store(self):
        s = EpisodicStore()
        assert s.count() == 0

    def test_explicit_memory(self):
        s = EpisodicStore(":memory:")
        assert s.count() == 0

    def test_file_store(self, tmp_path):
        db = tmp_path / "test.db"
        s = EpisodicStore(db)
        assert s.count() == 0
        s.close()
        assert db.exists()

    def test_repr(self, store):
        r = repr(store)
        assert "EpisodicStore" in r


# ---------------------------------------------------------------------------
# EpisodicStore — record
# ---------------------------------------------------------------------------

class TestRecord:
    def test_record_returns_episode_with_id(self, store, sample_episode):
        recorded = store.record(sample_episode)
        assert recorded.id is not None
        assert recorded.id > 0

    def test_count_increases(self, store, sample_episode):
        store.record(sample_episode)
        store.record(Episode(task="other task"))
        assert store.count() == 2

    def test_record_sets_timestamp_when_zero(self, store):
        ep = Episode(task="task", timestamp=0)
        before = time.time()
        store.record(ep)
        assert ep.timestamp >= before

    def test_record_preserves_existing_timestamp(self, store):
        ts = 1000.0
        ep = Episode(task="task", timestamp=ts)
        store.record(ep)
        assert ep.timestamp == ts

    def test_record_preserves_all_fields(self, store, sample_episode):
        store.record(sample_episode)
        retrieved = store.get(sample_episode.id)
        assert retrieved.task == sample_episode.task
        assert retrieved.robot_id == sample_episode.robot_id
        assert retrieved.skills_run == sample_episode.skills_run
        assert retrieved.robot_state == sample_episode.robot_state
        assert retrieved.outcome == sample_episode.outcome
        assert retrieved.task_type == sample_episode.task_type
        assert retrieved.duration_s == sample_episode.duration_s
        assert retrieved.metadata == sample_episode.metadata


# ---------------------------------------------------------------------------
# EpisodicStore — get
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_existing(self, store, sample_episode):
        store.record(sample_episode)
        retrieved = store.get(sample_episode.id)
        assert retrieved is not None
        assert retrieved.task == sample_episode.task

    def test_get_nonexistent_returns_none(self, store):
        assert store.get(99999) is None


# ---------------------------------------------------------------------------
# EpisodicStore — query
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_all(self, store):
        for i in range(5):
            store.record(Episode(task=f"task_{i}", robot_id="bot"))
        results = store.query()
        assert len(results) == 5

    def test_query_newest_first_by_default(self, store):
        store.record(Episode(task="first", timestamp=1000.0))
        store.record(Episode(task="second", timestamp=2000.0))
        results = store.query()
        assert results[0].task == "second"

    def test_query_oldest_first(self, store):
        store.record(Episode(task="first", timestamp=1000.0))
        store.record(Episode(task="second", timestamp=2000.0))
        results = store.query(order="ASC")
        assert results[0].task == "first"

    def test_query_by_robot_id(self, store):
        store.record(Episode(task="a", robot_id="bot1"))
        store.record(Episode(task="b", robot_id="bot2"))
        store.record(Episode(task="c", robot_id="bot1"))
        results = store.query(robot_id="bot1")
        assert len(results) == 2
        assert all(r.robot_id == "bot1" for r in results)

    def test_query_by_outcome(self, store):
        store.record(Episode(task="a", outcome="success"))
        store.record(Episode(task="b", outcome="failure"))
        store.record(Episode(task="c", outcome="success"))
        results = store.query(outcome="success")
        assert len(results) == 2

    def test_query_by_task_type(self, store):
        store.record(Episode(task="a", task_type="delivery"))
        store.record(Episode(task="b", task_type="patrol"))
        results = store.query(task_type="delivery")
        assert len(results) == 1

    def test_query_by_time_from(self, store):
        store.record(Episode(task="old", timestamp=1000.0))
        store.record(Episode(task="new", timestamp=3000.0))
        results = store.query(time_from=2000.0)
        assert len(results) == 1
        assert results[0].task == "new"

    def test_query_by_time_to(self, store):
        store.record(Episode(task="old", timestamp=1000.0))
        store.record(Episode(task="new", timestamp=3000.0))
        results = store.query(time_to=2000.0)
        assert len(results) == 1
        assert results[0].task == "old"

    def test_query_time_range(self, store):
        for ts in [1000.0, 2000.0, 3000.0, 4000.0]:
            store.record(Episode(task=f"t{ts}", timestamp=ts))
        results = store.query(time_from=1500.0, time_to=3500.0)
        assert len(results) == 2

    def test_query_limit(self, store):
        for i in range(20):
            store.record(Episode(task=f"task_{i}"))
        results = store.query(limit=5)
        assert len(results) == 5

    def test_query_offset(self, store):
        for i in range(10):
            store.record(Episode(task=f"task_{i}", timestamp=float(i)))
        results_first = store.query(limit=3, order="ASC")
        results_offset = store.query(limit=3, offset=3, order="ASC")
        assert results_first[0].task != results_offset[0].task

    def test_query_combined_filters(self, store):
        store.record(Episode(task="a", robot_id="bot1", outcome="success", timestamp=1000.0))
        store.record(Episode(task="b", robot_id="bot2", outcome="success", timestamp=2000.0))
        store.record(Episode(task="c", robot_id="bot1", outcome="failure", timestamp=3000.0))
        results = store.query(robot_id="bot1", outcome="success")
        assert len(results) == 1
        assert results[0].task == "a"


# ---------------------------------------------------------------------------
# EpisodicStore — count
# ---------------------------------------------------------------------------

class TestCount:
    def test_count_zero_initially(self, store):
        assert store.count() == 0

    def test_count_after_records(self, store):
        for i in range(7):
            store.record(Episode(task=f"t{i}"))
        assert store.count() == 7

    def test_count_by_robot_id(self, store):
        store.record(Episode(task="a", robot_id="bot1"))
        store.record(Episode(task="b", robot_id="bot2"))
        assert store.count(robot_id="bot1") == 1

    def test_count_by_outcome(self, store):
        store.record(Episode(task="a", outcome="success"))
        store.record(Episode(task="b", outcome="failure"))
        assert store.count(outcome="success") == 1


# ---------------------------------------------------------------------------
# EpisodicStore — maintenance
# ---------------------------------------------------------------------------

class TestMaintenance:
    def test_delete_older_than(self, store):
        store.record(Episode(task="old", timestamp=1000.0))
        store.record(Episode(task="new", timestamp=time.time()))
        deleted = store.delete_older_than(2000.0)
        assert deleted == 1
        assert store.count() == 1

    def test_clear(self, store):
        for i in range(5):
            store.record(Episode(task=f"task_{i}"))
        count = store.clear()
        assert count == 5
        assert store.count() == 0


# ---------------------------------------------------------------------------
# EpisodicStore — SQLite file persistence
# ---------------------------------------------------------------------------

class TestSQLitePersistence:
    def test_data_persists_across_connections(self, tmp_path):
        db = tmp_path / "persist.db"

        s1 = EpisodicStore(db)
        s1.record(Episode(task="persisted task", robot_id="bot"))
        s1.close()

        s2 = EpisodicStore(db)
        assert s2.count() == 1
        results = s2.query()
        assert results[0].task == "persisted task"
        s2.close()

    def test_tilde_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        s = EpisodicStore("~/test_episodes.db")
        s.record(Episode(task="x"))
        assert s.count() == 1
        s.close()

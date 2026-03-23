"""
Tests for apyrobo/memory/plan_cache.py — PlanCache.
"""

from __future__ import annotations

import time

import pytest

from apyrobo.memory.plan_cache import PlanCache, _task_hash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache():
    return PlanCache(ttl_s=60.0)


@pytest.fixture
def short_ttl_cache():
    return PlanCache(ttl_s=0.05)  # 50 ms — expires fast


SAMPLE_PLAN = [
    {"skill_id": "navigate_to_0", "params": {"x": 1.0, "y": 2.0}},
    {"skill_id": "pick_object_0", "params": {}},
]


# ---------------------------------------------------------------------------
# _task_hash helper
# ---------------------------------------------------------------------------

class TestTaskHash:
    def test_same_task_same_hash(self):
        assert _task_hash("patrol sector 3") == _task_hash("patrol sector 3")

    def test_case_insensitive(self):
        assert _task_hash("Patrol Sector 3") == _task_hash("patrol sector 3")

    def test_extra_whitespace_normalised(self):
        assert _task_hash("  patrol  sector  3  ") == _task_hash("patrol sector 3")

    def test_different_tasks_different_hashes(self):
        assert _task_hash("task a") != _task_hash("task b")

    def test_hash_is_hex_string(self):
        h = _task_hash("any task")
        assert isinstance(h, str)
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# store and lookup — in-memory
# ---------------------------------------------------------------------------

class TestStoreLookupMemory:
    def test_store_returns_key(self, cache):
        key = cache.store("patrol sector 3", SAMPLE_PLAN)
        assert isinstance(key, str)
        assert len(key) > 0

    def test_lookup_hit(self, cache):
        cache.store("patrol sector 3", SAMPLE_PLAN)
        result = cache.lookup("patrol sector 3")
        assert result is not None
        assert result["plan"] == SAMPLE_PLAN

    def test_lookup_miss(self, cache):
        result = cache.lookup("never stored task")
        assert result is None

    def test_lookup_result_has_expected_keys(self, cache):
        cache.store("task", SAMPLE_PLAN)
        result = cache.lookup("task")
        assert "plan" in result
        assert "task" in result
        assert "created_at" in result
        assert "hit_count" in result
        assert "cache_key" in result

    def test_lookup_increments_hit_count(self, cache):
        cache.store("task", SAMPLE_PLAN)
        cache.lookup("task")
        cache.lookup("task")
        result = cache.lookup("task")
        assert result["hit_count"] == 3

    def test_store_then_lookup_complex_plan(self, cache):
        plan = {"nested": {"data": [1, 2, 3]}, "skills": SAMPLE_PLAN}
        cache.store("complex task", plan)
        result = cache.lookup("complex task")
        assert result["plan"] == plan

    def test_case_insensitive_lookup(self, cache):
        cache.store("Patrol Sector 3", SAMPLE_PLAN)
        result = cache.lookup("patrol sector 3")
        assert result is not None


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------

class TestTTL:
    def test_entry_expires_after_ttl(self, short_ttl_cache):
        short_ttl_cache.store("task", SAMPLE_PLAN)
        assert short_ttl_cache.lookup("task") is not None
        time.sleep(0.1)
        assert short_ttl_cache.lookup("task") is None

    def test_per_store_ttl_override(self, cache):
        cache.store("task", SAMPLE_PLAN, ttl_s=0.05)
        time.sleep(0.1)
        assert cache.lookup("task") is None

    def test_expire_stale_removes_expired(self, short_ttl_cache):
        short_ttl_cache.store("a", [])
        short_ttl_cache.store("b", [])
        time.sleep(0.1)
        removed = short_ttl_cache.expire_stale()
        assert removed == 2

    def test_expire_stale_keeps_valid(self, cache):
        cache.store("keep", [])
        cache.store("expire", [], ttl_s=0.01)
        time.sleep(0.05)
        cache.expire_stale()
        assert cache.lookup("keep") is not None


# ---------------------------------------------------------------------------
# invalidate
# ---------------------------------------------------------------------------

class TestInvalidate:
    def test_invalidate_existing(self, cache):
        cache.store("task", SAMPLE_PLAN)
        result = cache.invalidate("task")
        assert result is True
        assert cache.lookup("task") is None

    def test_invalidate_nonexistent(self, cache):
        result = cache.invalidate("no such task")
        assert result is False

    def test_invalidate_all(self, cache):
        for i in range(5):
            cache.store(f"task_{i}", [])
        count = cache.invalidate_all()
        assert count == 5
        assert cache.size() == 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_initial_state(self, cache):
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.hit_rate == 0.0

    def test_hit_counts(self, cache):
        cache.store("task", SAMPLE_PLAN)
        cache.lookup("task")
        cache.lookup("task")
        assert cache.hits == 2

    def test_miss_counts(self, cache):
        cache.lookup("nonexistent")
        cache.lookup("nonexistent")
        assert cache.misses == 2

    def test_hit_rate_calculation(self, cache):
        cache.store("task", SAMPLE_PLAN)
        cache.lookup("task")    # hit
        cache.lookup("no")      # miss
        assert cache.hit_rate == pytest.approx(0.5)

    def test_size(self, cache):
        assert cache.size() == 0
        cache.store("a", [])
        cache.store("b", [])
        assert cache.size() == 2

    def test_stats_dict(self, cache):
        stats = cache.stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "size" in stats
        assert "mode" in stats
        assert "ttl_s" in stats
        assert stats["mode"] == "memory"

    def test_repr(self, cache):
        assert "PlanCache" in repr(cache)


# ---------------------------------------------------------------------------
# max_size eviction
# ---------------------------------------------------------------------------

class TestEviction:
    def test_evicts_when_over_max_size(self):
        cache = PlanCache(ttl_s=3600, max_size=3)
        cache.store("task_1", [])
        cache.store("task_2", [])
        cache.store("task_3", [])
        cache.store("task_4", [])  # triggers eviction
        assert cache.size() <= 3

    def test_eviction_counter_increments(self):
        cache = PlanCache(ttl_s=3600, max_size=2)
        cache.store("a", [])
        cache.store("b", [])
        cache.store("c", [])
        assert cache.evictions > 0


# ---------------------------------------------------------------------------
# SQLite mode
# ---------------------------------------------------------------------------

class TestSQLiteMode:
    def test_sqlite_store_and_lookup(self, tmp_path):
        db = tmp_path / "cache.db"
        cache = PlanCache(ttl_s=60, db_path=db)
        cache.store("task", SAMPLE_PLAN)
        result = cache.lookup("task")
        assert result is not None
        assert result["plan"] == SAMPLE_PLAN
        cache.close()

    def test_sqlite_persists_across_connections(self, tmp_path):
        db = tmp_path / "cache.db"
        c1 = PlanCache(ttl_s=60, db_path=db)
        c1.store("persisted task", SAMPLE_PLAN)
        c1.close()

        c2 = PlanCache(ttl_s=60, db_path=db)
        result = c2.lookup("persisted task")
        assert result is not None
        assert result["plan"] == SAMPLE_PLAN
        c2.close()

    def test_sqlite_expiry(self, tmp_path):
        db = tmp_path / "cache.db"
        cache = PlanCache(ttl_s=0.05, db_path=db)
        cache.store("task", SAMPLE_PLAN)
        time.sleep(0.1)
        result = cache.lookup("task")
        assert result is None
        cache.close()

    def test_sqlite_invalidate_all(self, tmp_path):
        db = tmp_path / "cache.db"
        cache = PlanCache(ttl_s=60, db_path=db)
        for i in range(5):
            cache.store(f"task_{i}", [])
        count = cache.invalidate_all()
        assert count == 5
        assert cache.size() == 0
        cache.close()

    def test_sqlite_stats_mode(self, tmp_path):
        db = tmp_path / "cache.db"
        cache = PlanCache(db_path=db)
        assert cache.stats()["mode"] == "sqlite"
        cache.close()

    def test_sqlite_expire_stale(self, tmp_path):
        db = tmp_path / "cache.db"
        cache = PlanCache(ttl_s=0.05, db_path=db)
        cache.store("task", SAMPLE_PLAN)
        time.sleep(0.1)
        removed = cache.expire_stale()
        assert removed >= 1
        cache.close()

    def test_sqlite_hit_count_increments(self, tmp_path):
        db = tmp_path / "cache.db"
        cache = PlanCache(ttl_s=60, db_path=db)
        cache.store("task", SAMPLE_PLAN)
        r1 = cache.lookup("task")
        r2 = cache.lookup("task")
        assert r2["hit_count"] > r1["hit_count"]
        cache.close()


# ---------------------------------------------------------------------------
# Package-level import
# ---------------------------------------------------------------------------

class TestPackageImport:
    def test_import_from_memory_package(self):
        from apyrobo.memory import PlanCache as PC
        assert PC is PlanCache

    def test_import_episode_store_from_package(self):
        from apyrobo.memory import EpisodicStore, Episode
        assert EpisodicStore is not None
        assert Episode is not None

    def test_import_semantic_store_from_package(self):
        from apyrobo.memory import SemanticStore
        assert SemanticStore is not None

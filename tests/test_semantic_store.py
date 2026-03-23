"""
Tests for apyrobo/memory/semantic.py — SemanticStore with vector similarity.
"""

from __future__ import annotations

import time

import pytest

from apyrobo.memory.semantic import SemanticStore, _cosine_similarity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    return SemanticStore()


@pytest.fixture
def ttl_store():
    return SemanticStore(default_ttl_s=0.1)  # 100 ms TTL


# ---------------------------------------------------------------------------
# Basic remember / get / forget
# ---------------------------------------------------------------------------

class TestRememberAndGet:
    def test_remember_and_get(self, store):
        store.remember("object:red_box", {"location": (2, 3)})
        val = store.get("object:red_box")
        assert val == {"location": (2, 3)}

    def test_get_missing_returns_none(self, store):
        assert store.get("no:such:key") is None

    def test_has_existing(self, store):
        store.remember("key", "value")
        assert store.has("key")

    def test_has_missing(self, store):
        assert not store.has("missing")

    def test_contains_operator(self, store):
        store.remember("x", 1)
        assert "x" in store
        assert "y" not in store

    def test_forget_existing(self, store):
        store.remember("key", "val")
        result = store.forget("key")
        assert result is True
        assert store.get("key") is None

    def test_forget_nonexistent(self, store):
        result = store.forget("nope")
        assert result is False

    def test_overwrite_value(self, store):
        store.remember("key", "old")
        store.remember("key", "new")
        assert store.get("key") == "new"

    def test_remember_various_types(self, store):
        store.remember("int_key", 42)
        store.remember("list_key", [1, 2, 3])
        store.remember("dict_key", {"nested": True})
        assert store.get("int_key") == 42
        assert store.get("list_key") == [1, 2, 3]

    def test_update_existing(self, store):
        store.remember("key", "old_value")
        result = store.update("key", "new_value")
        assert result is True
        assert store.get("key") == "new_value"

    def test_update_missing_returns_false(self, store):
        result = store.update("missing", "value")
        assert result is False


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------

class TestTTL:
    def test_entry_expires_after_ttl(self, ttl_store):
        ttl_store.remember("temp", "value")
        assert ttl_store.has("temp")
        time.sleep(0.15)
        assert not ttl_store.has("temp")
        assert ttl_store.get("temp") is None

    def test_per_entry_ttl_override(self, store):
        store.remember("fast", "value", ttl_s=0.05)
        time.sleep(0.1)
        assert store.get("fast") is None

    def test_no_ttl_never_expires(self, store):
        store.remember("persistent", "value")
        assert store.get("persistent") == "value"

    def test_expire_stale_removes_expired(self, ttl_store):
        ttl_store.remember("a", 1)
        ttl_store.remember("b", 2)
        time.sleep(0.15)
        removed = ttl_store.expire_stale()
        assert removed == 2
        assert ttl_store.count == 0

    def test_expire_stale_keeps_valid(self, store):
        store.remember("keep", "value")
        store.remember("expire", "value", ttl_s=0.01)
        time.sleep(0.05)
        store.expire_stale()
        assert store.has("keep")
        assert not store.has("expire")


# ---------------------------------------------------------------------------
# count / len / keys / items
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_count_empty(self, store):
        assert store.count == 0
        assert len(store) == 0

    def test_count_after_adds(self, store):
        store.remember("a", 1)
        store.remember("b", 2)
        assert store.count == 2
        assert len(store) == 2

    def test_keys_sorted(self, store):
        store.remember("c", 1)
        store.remember("a", 2)
        store.remember("b", 3)
        assert store.keys() == ["a", "b", "c"]

    def test_items(self, store):
        store.remember("x", 10)
        store.remember("y", 20)
        items = dict(store.items())
        assert items["x"] == 10
        assert items["y"] == 20

    def test_clear(self, store):
        store.remember("a", 1)
        store.remember("b", 2)
        count = store.clear()
        assert count == 2
        assert len(store) == 0

    def test_repr(self, store):
        assert "SemanticStore" in repr(store)


# ---------------------------------------------------------------------------
# recall — keyword matching
# ---------------------------------------------------------------------------

class TestRecallKeyword:
    def test_recall_exact_key(self, store):
        store.remember("object:red_box", {"color": "red"})
        results = store.recall("red_box")
        assert len(results) >= 1
        assert results[0]["key"] == "object:red_box"

    def test_recall_substring_match(self, store):
        store.remember("waypoint:alpha", {"x": 1})
        store.remember("waypoint:beta", {"x": 2})
        results = store.recall("waypoint")
        assert len(results) == 2

    def test_recall_no_match_returns_empty(self, store):
        store.remember("object:box", {"x": 1})
        results = store.recall("zzzyyyy")
        assert results == []

    def test_recall_respects_top_k(self, store):
        for i in range(20):
            store.remember(f"tag:{i}", i)
        results = store.recall("tag", top_k=5)
        assert len(results) == 5

    def test_recall_sorted_by_score(self, store):
        store.remember("navigate_to_target", "val1")
        store.remember("navigating:something", "val2")
        results = store.recall("navigate_to_target")
        # Exact substring match should score higher
        assert results[0]["key"] == "navigate_to_target"

    def test_recall_returns_value(self, store):
        store.remember("key:thing", {"data": 42})
        results = store.recall("thing")
        assert results[0]["value"] == {"data": 42}

    def test_recall_includes_match_type(self, store):
        store.remember("key:thing", "val")
        results = store.recall("thing")
        assert results[0]["match_type"] == "keyword"


# ---------------------------------------------------------------------------
# recall — vector similarity (numpy required)
# ---------------------------------------------------------------------------

class TestRecallVector:
    @pytest.fixture(autouse=True)
    def skip_without_numpy(self):
        try:
            import numpy
        except ImportError:
            pytest.skip("numpy not available")

    def test_vector_recall_finds_similar(self, store):
        import numpy as np
        store.remember("near", "close match", embedding=np.array([1.0, 0.0, 0.0]))
        store.remember("far", "distant", embedding=np.array([0.0, 1.0, 0.0]))

        results = store.recall(
            "query", query_embedding=np.array([0.9, 0.1, 0.0])
        )
        assert results[0]["key"] == "near"
        assert results[0]["match_type"] == "vector"

    def test_vector_recall_score_range(self, store):
        import numpy as np
        store.remember("x", "val", embedding=np.array([1.0, 0.0]))
        results = store.recall("query", query_embedding=np.array([1.0, 0.0]))
        assert 0.0 <= results[0]["score"] <= 1.0

    def test_vector_identical_score_is_one(self, store):
        import numpy as np
        v = np.array([1.0, 2.0, 3.0])
        store.remember("same", "value", embedding=v)
        results = store.recall("same", query_embedding=v)
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-6)

    def test_keyword_fallback_for_entries_without_embeddings(self, store):
        import numpy as np
        store.remember("object:box", {"color": "red"})  # no embedding
        results = store.recall(
            "box", query_embedding=np.array([1.0, 0.0])
        )
        # Should still find keyword match
        assert any(r["key"] == "object:box" for r in results)


# ---------------------------------------------------------------------------
# _cosine_similarity helper
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    @pytest.fixture(autouse=True)
    def skip_without_numpy(self):
        try:
            import numpy
        except ImportError:
            pytest.skip("numpy not available")

    def test_identical_vectors(self):
        import numpy as np
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        import numpy as np
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        import numpy as np
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        import numpy as np
        z = np.array([0.0, 0.0])
        v = np.array([1.0, 0.0])
        assert _cosine_similarity(z, v) == 0.0

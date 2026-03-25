"""Tests for edge inference adapter, mock adapter, and router."""

from __future__ import annotations

import pytest

from apyrobo.inference.edge import (
    EdgeInferenceAdapter,
    EdgeInferenceResult,
    EdgeInferenceRouter,
    EdgeModelConfig,
    MockEdgeInferenceAdapter,
)


def make_config(model_id: str = "tiny-llm") -> EdgeModelConfig:
    return EdgeModelConfig(
        model_id=model_id,
        model_path="/tmp/fake-model.gguf",
        max_tokens=64,
        warmup_on_load=False,
    )


class TestMockEdgeInferenceAdapter:
    def test_load_and_is_loaded(self):
        adapter = MockEdgeInferenceAdapter()
        assert not adapter.is_loaded()
        adapter.load()
        assert adapter.is_loaded()

    def test_unload(self):
        adapter = MockEdgeInferenceAdapter()
        adapter.load()
        adapter.unload()
        assert not adapter.is_loaded()

    def test_infer_returns_result(self):
        adapter = MockEdgeInferenceAdapter("test-model")
        adapter.load()
        result = adapter.infer("hello world")
        assert isinstance(result, EdgeInferenceResult)
        assert "hello world" in result.text
        assert result.model_id == "test-model"
        assert result.tokens_used > 0
        assert result.latency_ms >= 0

    def test_infer_not_loaded_raises(self):
        adapter = MockEdgeInferenceAdapter()
        with pytest.raises(RuntimeError, match="not loaded"):
            adapter.infer("test")

    def test_infer_batch(self):
        adapter = MockEdgeInferenceAdapter()
        adapter.load()
        results = adapter.infer_batch(["prompt a", "prompt b", "prompt c"])
        assert len(results) == 3
        assert all(isinstance(r, EdgeInferenceResult) for r in results)

    def test_infer_batch_empty(self):
        adapter = MockEdgeInferenceAdapter()
        adapter.load()
        assert adapter.infer_batch([]) == []

    def test_get_stats_initial(self):
        adapter = MockEdgeInferenceAdapter("m1")
        stats = adapter.get_stats()
        assert stats["model_id"] == "m1"
        assert stats["total_inferences"] == 0
        assert stats["avg_latency_ms"] == 0.0

    def test_get_stats_after_inferences(self):
        adapter = MockEdgeInferenceAdapter()
        adapter.load()
        adapter.infer("a")
        adapter.infer("b")
        stats = adapter.get_stats()
        assert stats["total_inferences"] == 2
        assert stats["avg_latency_ms"] >= 0

    def test_deterministic_model_id_in_result(self):
        adapter = MockEdgeInferenceAdapter("edge-v1")
        adapter.load()
        result = adapter.infer("test prompt")
        assert result.model_id == "edge-v1"


class TestEdgeInferenceAdapterStubMode:
    """EdgeInferenceAdapter should enter stub mode when llama_cpp is absent."""

    def test_load_enters_stub_mode_when_no_llama_cpp(self):
        config = make_config()
        adapter = EdgeInferenceAdapter(config)
        adapter.load()
        assert adapter.is_loaded()
        assert adapter._stub_mode

    def test_infer_stub_mode_returns_result(self):
        config = make_config("stub-model")
        adapter = EdgeInferenceAdapter(config)
        adapter.load()
        result = adapter.infer("navigate to dock")
        assert isinstance(result, EdgeInferenceResult)
        assert result.model_id == "stub-model"
        assert "stub-model" in result.text

    def test_infer_before_load_raises(self):
        adapter = EdgeInferenceAdapter(make_config())
        with pytest.raises(RuntimeError, match="not loaded"):
            adapter.infer("test")

    def test_unload_clears_model(self):
        adapter = EdgeInferenceAdapter(make_config())
        adapter.load()
        adapter.unload()
        assert not adapter.is_loaded()


class TestEdgeInferenceRouter:
    def _loaded_adapters(self, n: int) -> list[MockEdgeInferenceAdapter]:
        adapters = [MockEdgeInferenceAdapter(f"model-{i}") for i in range(n)]
        for a in adapters:
            a.load()
        return adapters

    def test_round_robin_cycles_through_adapters(self):
        adapters = self._loaded_adapters(3)
        router = EdgeInferenceRouter(adapters, strategy="round_robin")
        results = [router.infer(f"p{i}") for i in range(6)]
        model_ids = [r.model_id for r in results]
        assert model_ids == ["model-0", "model-1", "model-2", "model-0", "model-1", "model-2"]

    def test_router_infer_batch(self):
        adapters = self._loaded_adapters(2)
        router = EdgeInferenceRouter(adapters)
        results = router.infer_batch(["a", "b", "c", "d"])
        assert len(results) == 4

    def test_router_no_adapters_raises(self):
        router = EdgeInferenceRouter([])
        with pytest.raises(RuntimeError, match="No adapters"):
            router.infer("test")

    def test_router_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            EdgeInferenceRouter([], strategy="magic")

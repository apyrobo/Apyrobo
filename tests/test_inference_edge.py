"""Tests for edge inference adapter."""
import pytest
from apyrobo.inference.edge import (
    MockEdgeInferenceAdapter, EdgeInferenceAdapter,
    EdgeInferenceRouter, EdgeModelConfig, EdgeInferenceResult,
)


@pytest.fixture
def mock_adapter():
    return MockEdgeInferenceAdapter()


@pytest.fixture
def loaded_adapter(mock_adapter):
    mock_adapter.load()
    return mock_adapter


def test_config_defaults():
    cfg = EdgeModelConfig(model_id="test-model")
    assert cfg.max_tokens == 256
    assert cfg.device == "cpu"


def test_mock_load(mock_adapter):
    mock_adapter.load()
    assert mock_adapter.is_loaded() is True


def test_mock_unload(loaded_adapter):
    loaded_adapter.unload()
    assert loaded_adapter.is_loaded() is False


def test_mock_infer(loaded_adapter):
    result = loaded_adapter.infer("What should the robot do next?")
    assert isinstance(result, EdgeInferenceResult)
    assert "mock:" in result.text or result.text


def test_mock_infer_queued_response(loaded_adapter):
    loaded_adapter.queue_response("turn left")
    result = loaded_adapter.infer("navigate")
    assert result.text == "turn left"


def test_mock_infer_batch(loaded_adapter):
    prompts = ["prompt 1", "prompt 2", "prompt 3"]
    results = loaded_adapter.infer_batch(prompts)
    assert len(results) == 3


def test_mock_get_stats(loaded_adapter):
    loaded_adapter.infer("hello")
    stats = loaded_adapter.get_stats()
    assert stats["total_inferences"] == 1
    assert "avg_latency_ms" in stats


def test_edge_adapter_stub_mode():
    cfg = EdgeModelConfig(model_id="no-such-model", model_path="")
    adapter = EdgeInferenceAdapter(cfg)
    adapter.load()
    assert adapter.is_loaded()
    result = adapter.infer("test")
    assert result.model_id == "no-such-model"


def test_router_round_robin():
    adapters = [MockEdgeInferenceAdapter(), MockEdgeInferenceAdapter()]
    for a in adapters:
        a.load()
    router = EdgeInferenceRouter(adapters)
    r1 = router.infer("p1")
    r2 = router.infer("p2")
    assert isinstance(r1, EdgeInferenceResult)
    assert isinstance(r2, EdgeInferenceResult)


def test_router_no_adapters():
    router = EdgeInferenceRouter([])
    with pytest.raises(RuntimeError):
        router.infer("test")


def test_router_load_all():
    adapters = [MockEdgeInferenceAdapter(), MockEdgeInferenceAdapter()]
    router = EdgeInferenceRouter(adapters)
    router.load_all()
    assert all(a.is_loaded() for a in adapters)


def test_result_latency_positive(loaded_adapter):
    result = loaded_adapter.infer("timing test")
    assert result.latency_ms >= 0

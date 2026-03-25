"""Tests for apyrobo.inference.vlm"""
import pytest
from apyrobo.inference.vlm import MockVLMAdapter, VLMRouter


FAKE_IMAGE = b"\xff\xd8\xff" + b"\x00" * 100  # minimal fake JPEG bytes


class TestMockVLMAdapter:
    def test_describe_scene_returns_string(self):
        adapter = MockVLMAdapter()
        result = adapter.describe_scene(FAKE_IMAGE)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_describe_scene_custom_description(self):
        adapter = MockVLMAdapter(scene_description="custom scene")
        assert adapter.describe_scene(FAKE_IMAGE) == "custom scene"

    def test_answer_question_default(self):
        adapter = MockVLMAdapter()
        answer = adapter.answer_question(FAKE_IMAGE, "What is on the table?")
        assert "What is on the table?" in answer

    def test_answer_question_canned(self):
        adapter = MockVLMAdapter(answers={"Is it red?": "Yes, it is red."})
        assert adapter.answer_question(FAKE_IMAGE, "Is it red?") == "Yes, it is red."

    def test_describe_calls_recorded(self):
        adapter = MockVLMAdapter()
        adapter.describe_scene(FAKE_IMAGE)
        adapter.describe_scene(FAKE_IMAGE)
        assert len(adapter.describe_calls) == 2

    def test_question_calls_recorded(self):
        adapter = MockVLMAdapter()
        adapter.answer_question(FAKE_IMAGE, "Q1")
        adapter.answer_question(FAKE_IMAGE, "Q2")
        assert len(adapter.question_calls) == 2
        assert adapter.question_calls[0][1] == "Q1"


class TestVLMRouter:
    def test_route_vision_returns_string(self):
        router = VLMRouter()
        result = router.route_vision(FAKE_IMAGE, "Describe the scene")
        assert isinstance(result, str)

    def test_describe_delegates_to_adapter(self):
        adapter = MockVLMAdapter(scene_description="test scene")
        router = VLMRouter(adapter=adapter)
        assert router.describe(FAKE_IMAGE) == "test scene"

    def test_custom_adapter_injected(self):
        adapter = MockVLMAdapter(answers={"q": "a"})
        router = VLMRouter(adapter=adapter)
        assert router.route_vision(FAKE_IMAGE, "q") == "a"

    def test_set_adapter(self):
        router = VLMRouter()
        new_adapter = MockVLMAdapter(scene_description="new")
        router.set_adapter(new_adapter)
        assert router.describe(FAKE_IMAGE) == "new"


class TestInferenceRouterVision:
    """Integration: InferenceRouter.route_vision() uses mock when no VLM tier."""

    def test_route_vision_fallback_to_mock(self):
        from apyrobo.inference.router import InferenceRouter
        router = InferenceRouter()
        result = router.route_vision(FAKE_IMAGE, "What do you see?")
        assert isinstance(result, str)
        assert len(result) > 0

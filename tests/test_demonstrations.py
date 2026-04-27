"""Tests for apyrobo.skills.demonstrations."""
import json
import os
import pytest
from datetime import datetime, timezone

from apyrobo.skills.demonstrations import (
    Demonstration,
    DemonstrationRecorder,
    DemonstrationReplayer,
    DemonstrationStep,
    DemonstrationStore,
    SkillLearner,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockExecutor:
    def __init__(self):
        self.calls = []

    def execute_skill(self, name, params):
        self.calls.append((name, params))
        return {"status": "ok"}


def _make_step(skill_name="navigate_to", params=None, success=True) -> DemonstrationStep:
    return DemonstrationStep(
        timestamp=1.0,
        skill_name=skill_name,
        params=params or {"x": 1.0, "y": 2.0},
        state_before={"pos": [0, 0]},
        state_after={"pos": [1, 2]},
        success=success,
    )


def _make_demo(robot_id="robot-1", steps=None) -> Demonstration:
    return Demonstration(
        demo_id="demo-abc",
        robot_id=robot_id,
        steps=[_make_step()] if steps is None else steps,
        metadata={},
        recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# DemonstrationStep
# ---------------------------------------------------------------------------

def test_step_fields():
    step = DemonstrationStep(
        timestamp=42.0,
        skill_name="pick_object",
        params={"object_id": "box1"},
        state_before={"gripper": "open"},
        state_after={"gripper": "closed"},
        success=True,
    )
    assert step.timestamp == 42.0
    assert step.skill_name == "pick_object"
    assert step.params == {"object_id": "box1"}
    assert step.success is True


def test_step_default_success():
    step = DemonstrationStep(
        timestamp=0.0,
        skill_name="wait",
        params={},
        state_before={},
        state_after={},
    )
    assert step.success is True


# ---------------------------------------------------------------------------
# DemonstrationRecorder
# ---------------------------------------------------------------------------

def test_recorder_start_returns_uuid():
    rec = DemonstrationRecorder()
    demo_id = rec.start_recording("robot-1")
    assert isinstance(demo_id, str) and len(demo_id) == 36  # uuid4 format


def test_recorder_is_recording_state_machine():
    rec = DemonstrationRecorder()
    assert rec.is_recording() is False
    rec.start_recording("robot-1")
    assert rec.is_recording() is True
    rec.stop_recording()
    assert rec.is_recording() is False


def test_recorder_record_and_stop():
    rec = DemonstrationRecorder()
    rec.start_recording("robot-2")
    rec.record_step(
        "navigate_to",
        {"x": 1, "y": 2},
        {"pos": [0, 0]},
        {"pos": [1, 2]},
    )
    rec.record_step(
        "pick_object",
        {"id": "box"},
        {"gripper": "open"},
        {"gripper": "closed"},
        success=False,
    )
    demo = rec.stop_recording()
    assert demo.robot_id == "robot-2"
    assert len(demo.steps) == 2
    assert demo.steps[0].skill_name == "navigate_to"
    assert demo.steps[1].success is False


def test_recorder_stop_resets_state():
    rec = DemonstrationRecorder()
    rec.start_recording("robot-x")
    rec.stop_recording()
    assert rec.is_recording() is False


def test_recorder_record_without_start_raises():
    rec = DemonstrationRecorder()
    with pytest.raises(RuntimeError):
        rec.record_step("navigate_to", {}, {}, {})


def test_recorder_stop_without_start_raises():
    rec = DemonstrationRecorder()
    with pytest.raises(RuntimeError):
        rec.stop_recording()


def test_recorder_second_recording_fresh_steps():
    rec = DemonstrationRecorder()
    rec.start_recording("robot-1")
    rec.record_step("navigate_to", {}, {}, {})
    rec.stop_recording()

    rec.start_recording("robot-1")
    demo2 = rec.stop_recording()
    assert demo2.steps == []


# ---------------------------------------------------------------------------
# DemonstrationStore
# ---------------------------------------------------------------------------

def test_store_save_load_roundtrip(tmp_path):
    demo = _make_demo()
    store = DemonstrationStore()
    path = str(tmp_path / "demo.json")
    store.save(demo, path)
    loaded = store.load(path)

    assert loaded.demo_id == demo.demo_id
    assert loaded.robot_id == demo.robot_id
    assert len(loaded.steps) == len(demo.steps)
    assert loaded.steps[0].skill_name == demo.steps[0].skill_name
    assert loaded.steps[0].params == demo.steps[0].params
    assert loaded.steps[0].success == demo.steps[0].success


def test_store_saved_file_is_valid_json(tmp_path):
    demo = _make_demo()
    store = DemonstrationStore()
    path = str(tmp_path / "demo.json")
    store.save(demo, path)
    with open(path) as f:
        data = json.load(f)
    assert "demo_id" in data
    assert "steps" in data


def test_store_list_demos(tmp_path):
    store = DemonstrationStore()
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("ignore me")
    demos = store.list_demos(str(tmp_path))
    basenames = [os.path.basename(p) for p in demos]
    assert "a.json" in basenames
    assert "b.json" in basenames
    assert "notes.txt" not in basenames


def test_store_list_demos_missing_directory():
    store = DemonstrationStore()
    result = store.list_demos("/nonexistent/path/xyz")
    assert result == []


# ---------------------------------------------------------------------------
# DemonstrationReplayer
# ---------------------------------------------------------------------------

def test_replayer_replay_calls_executor():
    executor = MockExecutor()
    replayer = DemonstrationReplayer(executor)
    demo = _make_demo(steps=[
        _make_step("navigate_to", {"x": 1}),
        _make_step("pick_object", {"id": "box"}),
    ])
    results = replayer.replay(demo)
    assert len(results) == 2
    assert executor.calls[0] == ("navigate_to", {"x": 1})
    assert executor.calls[1] == ("pick_object", {"id": "box"})


def test_replayer_replay_step_returns_executor_result():
    executor = MockExecutor()
    replayer = DemonstrationReplayer(executor)
    step = _make_step("navigate_to", {"x": 5})
    result = replayer.replay_step(step)
    assert result == {"status": "ok"}


def test_replayer_empty_demo():
    executor = MockExecutor()
    replayer = DemonstrationReplayer(executor)
    demo = _make_demo(steps=[])
    results = replayer.replay(demo)
    assert results == []
    assert executor.calls == []


# ---------------------------------------------------------------------------
# SkillLearner
# ---------------------------------------------------------------------------

def _two_demos():
    steps_a = [
        _make_step("navigate_to", {"x": 1}),
        _make_step("pick_object", {"id": "box1"}),
        _make_step("deliver", {"location": "dock"}),
    ]
    steps_b = [
        _make_step("navigate_to", {"x": 2}),
        _make_step("pick_object", {"id": "box2"}, success=False),
        _make_step("deliver", {"location": "dock"}),
    ]
    return [_make_demo(steps=steps_a), _make_demo(steps=steps_b)]


def test_learner_learn_returns_summary():
    learner = SkillLearner()
    summary = learner.learn_from_demonstrations(_two_demos())
    assert "skill_counts" in summary
    assert "success_rates" in summary
    assert "top_sequences" in summary
    assert summary["demo_count"] == 2


def test_learner_skill_counts():
    learner = SkillLearner()
    summary = learner.learn_from_demonstrations(_two_demos())
    counts = summary["skill_counts"]
    assert counts["navigate_to"] == 2
    assert counts["pick_object"] == 2
    assert counts["deliver"] == 2


def test_learner_success_rates():
    learner = SkillLearner()
    summary = learner.learn_from_demonstrations(_two_demos())
    rates = summary["success_rates"]
    assert rates["navigate_to"] == 1.0
    assert rates["pick_object"] == 0.5
    assert rates["deliver"] == 1.0


def test_learner_extract_skill_template_averages_numeric_params():
    steps = [
        _make_step("navigate_to", {"x": 2.0, "y": 4.0}),
        _make_step("navigate_to", {"x": 4.0, "y": 0.0}),
    ]
    demo = _make_demo(steps=steps)
    learner = SkillLearner()
    template = learner.extract_skill_template("navigate_to", [demo])
    assert template["x"] == pytest.approx(3.0)
    assert template["y"] == pytest.approx(2.0)


def test_learner_extract_skill_template_missing_skill():
    demo = _make_demo(steps=[_make_step("navigate_to")])
    learner = SkillLearner()
    template = learner.extract_skill_template("nonexistent_skill", [demo])
    assert template == {}


def test_learner_suggest_next_step_returns_string_or_none():
    learner = SkillLearner()
    demos = _two_demos()
    learner.learn_from_demonstrations(demos)
    history = [_make_step("navigate_to")]
    result = learner.suggest_next_step({}, history)
    assert result is None or isinstance(result, str)


def test_learner_suggest_next_step_uses_patterns():
    learner = SkillLearner()
    demos = _two_demos()
    learner.learn_from_demonstrations(demos)
    # After navigate_to, both demos proceed to pick_object
    history = [_make_step("navigate_to")]
    suggestion = learner.suggest_next_step({}, history)
    assert suggestion == "pick_object"


def test_learner_suggest_next_step_no_history():
    learner = SkillLearner()
    learner.learn_from_demonstrations(_two_demos())
    result = learner.suggest_next_step({}, [])
    assert result is None


def test_learner_suggest_next_step_no_training():
    learner = SkillLearner()
    history = [_make_step("navigate_to")]
    result = learner.suggest_next_step({}, history)
    assert result is None

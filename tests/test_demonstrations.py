"""Tests for apyrobo.skills.demonstrations."""

from __future__ import annotations

import json
import pytest

from apyrobo.skills.demonstrations import (
    Demonstration,
    DemonstrationRecorder,
    DemonstrationReplayer,
    DemonstrationStep,
    DemonstrationStore,
    LearnedPattern,
    SkillLearner,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockExecutor:
    def __init__(self, fail_on: set[str] | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.fail_on = fail_on or set()

    def dispatch(self, skill_name: str, **params) -> dict:
        self.calls.append((skill_name, params))
        if skill_name in self.fail_on:
            raise ValueError(f"Simulated failure for {skill_name!r}")
        return {"status": "ok", "skill": skill_name}


def _make_step(skill_name: str = "navigate_to", **params) -> DemonstrationStep:
    return DemonstrationStep(
        skill_name=skill_name,
        parameters=params or {"x": 1.0, "y": 2.0},
        duration_s=0.5,
    )


def _make_demo(name: str = "test_demo", *skill_names: str) -> Demonstration:
    if not skill_names:
        skill_names = ("navigate_to",)
    demo = Demonstration(name=name)
    for sk in skill_names:
        demo.steps.append(_make_step(sk))
    return demo


# ---------------------------------------------------------------------------
# DemonstrationStep
# ---------------------------------------------------------------------------

class TestDemonstrationStep:
    def test_minimal_construction(self):
        step = DemonstrationStep(skill_name="pick_object")
        assert step.skill_name == "pick_object"
        assert step.parameters == {}
        assert step.success is True
        assert step.duration_s == 0.0

    def test_full_construction(self):
        step = DemonstrationStep(
            skill_name="navigate_to",
            parameters={"x": 3.0, "y": 4.0},
            duration_s=1.2,
            state_before={"pos": [0, 0]},
            state_after={"pos": [3, 4]},
            success=False,
            notes="hit obstacle",
        )
        assert step.parameters == {"x": 3.0, "y": 4.0}
        assert step.success is False
        assert step.notes == "hit obstacle"

    def test_timestamp_auto_set(self):
        step = DemonstrationStep(skill_name="wait")
        assert step.timestamp > 0


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

class TestDemonstration:
    def test_demo_id_auto_generated(self):
        d = Demonstration(name="a")
        assert len(d.demo_id) == 36  # UUID4

    def test_two_demos_have_different_ids(self):
        a = Demonstration(name="a")
        b = Demonstration(name="b")
        assert a.demo_id != b.demo_id

    def test_step_count(self):
        d = _make_demo("x", "nav", "pick", "drop")
        assert d.step_count == 3

    def test_skill_sequence(self):
        d = _make_demo("x", "nav", "pick", "drop")
        assert d.skill_sequence == ["nav", "pick", "drop"]

    def test_successful_steps_filters(self):
        d = Demonstration(name="x")
        d.steps.append(DemonstrationStep(skill_name="a", success=True))
        d.steps.append(DemonstrationStep(skill_name="b", success=False))
        assert len(d.successful_steps()) == 1

    def test_duration_s_none_end_time(self):
        d = Demonstration(name="x")
        assert d.duration_s == 0.0


# ---------------------------------------------------------------------------
# DemonstrationRecorder
# ---------------------------------------------------------------------------

class TestDemonstrationRecorder:
    def test_start_returns_demonstration(self):
        rec = DemonstrationRecorder()
        demo = rec.start("pick_and_place")
        assert isinstance(demo, Demonstration)
        assert demo.name == "pick_and_place"

    def test_is_recording_state_machine(self):
        rec = DemonstrationRecorder()
        assert rec.is_recording is False
        rec.start("t")
        assert rec.is_recording is True
        rec.stop()
        assert rec.is_recording is False

    def test_record_step_appends(self):
        rec = DemonstrationRecorder()
        rec.start("demo")
        rec.record_step("navigate_to", {"x": 1.0})
        rec.record_step("pick_object", {"id": "box"})
        demo = rec.stop()
        assert len(demo.steps) == 2
        assert demo.steps[0].skill_name == "navigate_to"

    def test_stop_sets_end_time(self):
        rec = DemonstrationRecorder()
        rec.start("d")
        demo = rec.stop()
        assert demo.end_time is not None
        assert demo.end_time >= demo.start_time

    def test_start_twice_raises(self):
        rec = DemonstrationRecorder()
        rec.start("first")
        with pytest.raises(RuntimeError, match="Already recording"):
            rec.start("second")

    def test_record_without_start_raises(self):
        rec = DemonstrationRecorder()
        with pytest.raises(RuntimeError, match="Not recording"):
            rec.record_step("nav", {})

    def test_stop_without_start_raises(self):
        rec = DemonstrationRecorder()
        with pytest.raises(RuntimeError, match="Not recording"):
            rec.stop()

    def test_second_recording_starts_fresh(self):
        rec = DemonstrationRecorder()
        rec.start("first")
        rec.record_step("a")
        rec.stop()
        rec.start("second")
        demo2 = rec.stop()
        assert demo2.name == "second"
        assert demo2.steps == []

    def test_record_step_with_all_fields(self):
        rec = DemonstrationRecorder()
        rec.start("d")
        step = rec.record_step(
            "pick",
            parameters={"id": "box"},
            duration_s=0.8,
            state_before={"gripper": "open"},
            state_after={"gripper": "closed"},
            success=False,
            notes="slipped",
        )
        assert step.success is False
        assert step.notes == "slipped"
        rec.stop()

    def test_current_demo_while_recording(self):
        rec = DemonstrationRecorder()
        assert rec.current_demo() is None
        rec.start("d")
        assert rec.current_demo() is not None
        rec.stop()
        assert rec.current_demo() is None


# ---------------------------------------------------------------------------
# DemonstrationStore
# ---------------------------------------------------------------------------

class TestDemonstrationStore:
    def test_save_creates_file(self, tmp_path):
        store = DemonstrationStore(directory=str(tmp_path))
        demo = _make_demo("nav_demo", "navigate_to")
        path = store.save(demo)
        assert path.exists()

    def test_save_load_roundtrip(self, tmp_path):
        store = DemonstrationStore(directory=str(tmp_path))
        demo = _make_demo("test", "navigate_to", "pick_object")
        store.save(demo)
        loaded = store.load(demo.demo_id)
        assert loaded.name == "test"
        assert len(loaded.steps) == 2
        assert loaded.steps[0].skill_name == "navigate_to"

    def test_load_missing_raises(self, tmp_path):
        store = DemonstrationStore(directory=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent-id")

    def test_list_all_returns_sorted_by_time(self, tmp_path):
        store = DemonstrationStore(directory=str(tmp_path))
        d1 = Demonstration(name="first", start_time=1.0)
        d2 = Demonstration(name="second", start_time=2.0)
        store.save(d2)
        store.save(d1)
        demos = store.list_all()
        assert demos[0].name == "first"
        assert demos[1].name == "second"

    def test_list_ids(self, tmp_path):
        store = DemonstrationStore(directory=str(tmp_path))
        d1 = _make_demo("a")
        d2 = _make_demo("b")
        store.save(d1)
        store.save(d2)
        ids = store.list_ids()
        assert d1.demo_id in ids
        assert d2.demo_id in ids

    def test_count(self, tmp_path):
        store = DemonstrationStore(directory=str(tmp_path))
        store.save(_make_demo("a"))
        store.save(_make_demo("b"))
        assert store.count() == 2

    def test_delete_existing(self, tmp_path):
        store = DemonstrationStore(directory=str(tmp_path))
        demo = _make_demo("d")
        store.save(demo)
        assert store.delete(demo.demo_id) is True
        assert store.count() == 0

    def test_delete_missing_returns_false(self, tmp_path):
        store = DemonstrationStore(directory=str(tmp_path))
        assert store.delete("not-there") is False

    def test_load_by_name(self, tmp_path):
        store = DemonstrationStore(directory=str(tmp_path))
        store.save(Demonstration(name="patrol"))
        store.save(Demonstration(name="patrol"))
        store.save(Demonstration(name="deliver"))
        results = store.load_by_name("patrol")
        assert len(results) == 2

    def test_saved_file_is_valid_json(self, tmp_path):
        store = DemonstrationStore(directory=str(tmp_path))
        demo = _make_demo("test", "navigate_to")
        path = store.save(demo)
        data = json.loads(path.read_text())
        assert "demo_id" in data
        assert "steps" in data
        assert data["name"] == "test"


# ---------------------------------------------------------------------------
# DemonstrationReplayer
# ---------------------------------------------------------------------------

class TestDemonstrationReplayer:
    def test_replay_calls_executor_dispatch(self):
        exc = MockExecutor()
        replayer = DemonstrationReplayer(exc)
        demo = _make_demo("d", "navigate_to", "pick_object")
        records = replayer.replay(demo)
        assert len(records) == 2
        assert exc.calls[0][0] == "navigate_to"
        assert exc.calls[1][0] == "pick_object"

    def test_replay_empty_demo(self):
        exc = MockExecutor()
        replayer = DemonstrationReplayer(exc)
        records = replayer.replay(Demonstration(name="empty"))
        assert records == []

    def test_replay_step_returns_record(self):
        exc = MockExecutor()
        replayer = DemonstrationReplayer(exc)
        step = _make_step("navigate_to", x=1.0)
        record = replayer.replay_step(step)
        assert record["skill"] == "navigate_to"
        assert record["result"] is not None
        assert record["error"] is None

    def test_replay_executor_error_captured(self):
        exc = MockExecutor(fail_on={"broken_skill"})
        replayer = DemonstrationReplayer(exc)
        step = _make_step("broken_skill")
        record = replayer.replay_step(step)
        assert record["error"] is not None
        assert record["result"] is None

    def test_replay_skip_failed_steps(self):
        exc = MockExecutor()
        replayer = DemonstrationReplayer(exc)
        demo = Demonstration(name="d")
        demo.steps.append(DemonstrationStep(skill_name="ok_skill", success=True))
        demo.steps.append(DemonstrationStep(skill_name="bad_step", success=False))
        records = replayer.replay(demo, skip_failed=True)
        assert len(records) == 2
        assert records[1].get("skipped") is True
        assert len(exc.calls) == 1  # only ok_skill dispatched


# ---------------------------------------------------------------------------
# SkillLearner
# ---------------------------------------------------------------------------

class TestSkillLearner:
    def _two_demos(self) -> list[Demonstration]:
        d1 = Demonstration(name="delivery", demo_id="d1")
        d1.steps = [_make_step("navigate_to"), _make_step("pick_object"), _make_step("deliver")]
        d2 = Demonstration(name="delivery", demo_id="d2")
        d2.steps = [_make_step("navigate_to"), _make_step("pick_object"), _make_step("deliver")]
        return [d1, d2]

    def test_learn_empty_returns_empty(self):
        learner = SkillLearner()
        assert learner.learn([]) == []

    def test_learn_finds_repeated_bigrams(self):
        demos = self._two_demos()
        learner = SkillLearner(min_frequency=2)
        patterns = learner.learn(demos)
        seqs = [p.skill_sequence for p in patterns]
        assert ["navigate_to", "pick_object"] in seqs or \
               ["pick_object", "deliver"] in seqs

    def test_learn_frequency_count(self):
        demos = self._two_demos()
        learner = SkillLearner(min_frequency=2)
        patterns = learner.learn(demos)
        for p in patterns:
            assert p.frequency >= 2

    def test_learn_min_frequency_filters(self):
        demos = self._two_demos()
        learner = SkillLearner(min_frequency=3)  # 3 required, only 2 demos
        patterns = learner.learn(demos)
        assert patterns == []

    def test_extract_unique_skills(self):
        demos = self._two_demos()
        learner = SkillLearner()
        skills = learner.extract_unique_skills(demos)
        assert "navigate_to" in skills
        assert "pick_object" in skills
        assert "deliver" in skills
        assert len(skills) == 3

    def test_most_common_sequence_returns_list(self):
        demos = self._two_demos()
        learner = SkillLearner(min_frequency=2)
        seq = learner.most_common_sequence(demos)
        assert isinstance(seq, list)
        assert len(seq) >= 2

    def test_most_common_sequence_empty_demos(self):
        learner = SkillLearner()
        assert learner.most_common_sequence([]) == []

    def test_summarise_empty(self):
        learner = SkillLearner()
        s = learner.summarise([])
        assert s["demos"] == 0

    def test_summarise_counts(self):
        demos = self._two_demos()
        learner = SkillLearner(min_frequency=2)
        s = learner.summarise(demos)
        assert s["demos"] == 2
        assert s["total_steps"] == 6
        assert s["unique_skills"] == 3

    def test_suggest_next_step(self):
        demos = self._two_demos()
        learner = SkillLearner()
        suggestion = learner.suggest_next_step("navigate_to", demos)
        assert suggestion == "pick_object"

    def test_suggest_next_step_unknown_skill(self):
        demos = self._two_demos()
        learner = SkillLearner()
        suggestion = learner.suggest_next_step("nonexistent", demos)
        assert suggestion is None

    def test_learned_pattern_has_source_demo_ids(self):
        demos = self._two_demos()
        learner = SkillLearner(min_frequency=2)
        patterns = learner.learn(demos)
        for p in patterns:
            assert len(p.source_demo_ids) >= 2

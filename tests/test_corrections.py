"""Tests for correction learning (CL-01)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from apyrobo.skills.corrections import (
    Correction,
    CorrectionLearner,
    CorrectionStore,
    _word_overlap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store() -> CorrectionStore:
    return CorrectionStore(db_path=":memory:")


def _learner() -> CorrectionLearner:
    return CorrectionLearner(_store())


def _nav_orig() -> dict:
    return {"skill_id": "navigate_to", "parameters": {"x": 1.0, "y": 0.0}}


def _nav_corr() -> dict:
    return {"skill_id": "navigate_to", "parameters": {"x": 2.0, "y": 0.0}}


# ---------------------------------------------------------------------------
# _word_overlap helper
# ---------------------------------------------------------------------------

class TestWordOverlap:
    def test_identical_strings(self):
        assert _word_overlap("deliver package to lab", "deliver package to lab") == 1.0

    def test_no_overlap(self):
        assert _word_overlap("deliver package", "inspect room") == 0.0

    def test_partial_overlap(self):
        score = _word_overlap("deliver package to lab", "deliver robot to dock")
        assert 0.0 < score < 1.0

    def test_empty_strings(self):
        assert _word_overlap("", "") == 1.0

    def test_one_empty(self):
        assert _word_overlap("hello", "") == 0.0


# ---------------------------------------------------------------------------
# Correction dataclass
# ---------------------------------------------------------------------------

class TestCorrectionDataclass:
    def test_fields_stored(self):
        c = Correction(
            correction_id="c1",
            task_description="navigate to lab",
            original_step={"skill_id": "navigate_to", "parameters": {"x": 0.0}},
            corrected_step={"skill_id": "navigate_to", "parameters": {"x": 5.0}},
            reason="wrong target",
            timestamp="2026-01-01T00:00:00",
        )
        assert c.correction_id == "c1"
        assert c.applied_count == 0
        assert c.reason == "wrong target"

    def test_applied_count_defaults_zero(self):
        c = Correction("id", "task", {}, {}, None, "2026-01-01")
        assert c.applied_count == 0


# ---------------------------------------------------------------------------
# CorrectionStore — record
# ---------------------------------------------------------------------------

class TestCorrectionStoreRecord:
    def test_record_returns_correction(self):
        s = _store()
        c = s.record("navigate to lab", _nav_orig(), _nav_corr(), "wrong target")
        assert isinstance(c, Correction)

    def test_record_generates_unique_ids(self):
        s = _store()
        c1 = s.record("task A", _nav_orig(), _nav_corr())
        c2 = s.record("task B", _nav_orig(), _nav_corr())
        assert c1.correction_id != c2.correction_id

    def test_record_stores_task_description(self):
        s = _store()
        c = s.record("navigate to lab", _nav_orig(), _nav_corr())
        assert c.task_description == "navigate to lab"

    def test_record_stores_steps(self):
        s = _store()
        c = s.record("task", _nav_orig(), _nav_corr())
        assert c.original_step == _nav_orig()
        assert c.corrected_step == _nav_corr()

    def test_record_stores_reason(self):
        s = _store()
        c = s.record("task", _nav_orig(), _nav_corr(), "overshoot")
        assert c.reason == "overshoot"

    def test_record_empty_reason_stored_as_none(self):
        s = _store()
        c = s.record("task", _nav_orig(), _nav_corr(), "")
        assert c.reason is None

    def test_record_timestamp_set(self):
        s = _store()
        c = s.record("task", _nav_orig(), _nav_corr())
        assert "T" in c.timestamp  # ISO 8601


# ---------------------------------------------------------------------------
# CorrectionStore — list_all
# ---------------------------------------------------------------------------

class TestCorrectionStoreListAll:
    def test_empty_store(self):
        s = _store()
        assert s.list_all() == []

    def test_one_correction(self):
        s = _store()
        s.record("task", _nav_orig(), _nav_corr())
        assert len(s.list_all()) == 1

    def test_multiple_corrections(self):
        s = _store()
        s.record("task A", _nav_orig(), _nav_corr())
        s.record("task B", _nav_orig(), _nav_corr())
        assert len(s.list_all()) == 2

    def test_round_trip_original_step(self):
        s = _store()
        s.record("task", _nav_orig(), _nav_corr())
        c = s.list_all()[0]
        assert c.original_step == _nav_orig()

    def test_round_trip_corrected_step(self):
        s = _store()
        s.record("task", _nav_orig(), _nav_corr())
        c = s.list_all()[0]
        assert c.corrected_step == _nav_corr()


# ---------------------------------------------------------------------------
# CorrectionStore — mark_applied
# ---------------------------------------------------------------------------

class TestCorrectionStoreMarkApplied:
    def test_increments_applied_count(self):
        s = _store()
        c = s.record("task", _nav_orig(), _nav_corr())
        assert c.applied_count == 0
        s.mark_applied(c.correction_id)
        updated = s.list_all()[0]
        assert updated.applied_count == 1

    def test_mark_applied_twice(self):
        s = _store()
        c = s.record("task", _nav_orig(), _nav_corr())
        s.mark_applied(c.correction_id)
        s.mark_applied(c.correction_id)
        updated = s.list_all()[0]
        assert updated.applied_count == 2


# ---------------------------------------------------------------------------
# CorrectionStore — find_relevant
# ---------------------------------------------------------------------------

class TestCorrectionStoreFindRelevant:
    def test_returns_matching_skill_id(self):
        s = _store()
        s.record("navigate to lab A", _nav_orig(), _nav_corr())
        results = s.find_relevant("navigate to lab", skill_id="navigate_to")
        assert len(results) == 1

    def test_filters_by_skill_id(self):
        s = _store()
        s.record("navigate to lab", _nav_orig(), _nav_corr())
        stop_orig = {"skill_id": "stop", "parameters": {}}
        stop_corr = {"skill_id": "stop", "parameters": {}}
        s.record("stop the robot", stop_orig, stop_corr)
        results = s.find_relevant("navigate somewhere", skill_id="navigate_to")
        assert all(
            r.original_step.get("skill_id") == "navigate_to"
            or r.corrected_step.get("skill_id") == "navigate_to"
            for r in results
        )

    def test_top_k_limit(self):
        s = _store()
        for i in range(5):
            s.record(f"navigate to room {i}", _nav_orig(), _nav_corr())
        results = s.find_relevant("navigate to room", skill_id="navigate_to", top_k=2)
        assert len(results) <= 2

    def test_no_results_for_empty_store(self):
        s = _store()
        assert s.find_relevant("task", skill_id="navigate_to") == []

    def test_empty_skill_id_returns_all(self):
        s = _store()
        s.record("navigate to lab", _nav_orig(), _nav_corr())
        stop_orig = {"skill_id": "stop", "parameters": {}}
        stop_corr = {"skill_id": "stop", "parameters": {}}
        s.record("stop robot", stop_orig, stop_corr)
        results = s.find_relevant("something", skill_id="")
        assert len(results) == 2

    def test_ranked_by_task_similarity(self):
        s = _store()
        s.record("navigate to lab A", _nav_orig(), _nav_corr())
        s.record("deliver package to dock", _nav_orig(), _nav_corr())
        results = s.find_relevant("navigate to lab B", skill_id="navigate_to", top_k=2)
        assert results[0].task_description == "navigate to lab A"


# ---------------------------------------------------------------------------
# CorrectionLearner
# ---------------------------------------------------------------------------

class TestCorrectionLearner:
    def test_record_correction_stores(self):
        l = _learner()
        l.record_correction("task", _nav_orig(), _nav_corr(), "overshoot")
        assert len(l.store.list_all()) == 1

    def test_record_correction_returns_correction(self):
        l = _learner()
        c = l.record_correction("task", _nav_orig(), _nav_corr())
        assert isinstance(c, Correction)

    def test_augment_prompt_no_corrections(self):
        l = _learner()
        result = l.augment_prompt("navigate to lab", "base prompt")
        assert result == "base prompt"

    def test_augment_prompt_prepends_corrections(self):
        l = _learner()
        l.record_correction("navigate to lab", _nav_orig(), _nav_corr(), "wrong target")
        result = l.augment_prompt("navigate to lab", "base prompt", skill_id="navigate_to")
        assert "Past corrections" in result
        assert "base prompt" in result

    def test_augment_prompt_corrections_before_base(self):
        l = _learner()
        l.record_correction("navigate to lab", _nav_orig(), _nav_corr())
        result = l.augment_prompt("navigate to lab", "base prompt", skill_id="navigate_to")
        assert result.index("Past corrections") < result.index("base prompt")

    def test_augment_prompt_marks_corrections_applied(self):
        l = _learner()
        l.record_correction("navigate to lab", _nav_orig(), _nav_corr())
        l.augment_prompt("navigate to lab", "base prompt", skill_id="navigate_to")
        c = l.store.list_all()[0]
        assert c.applied_count == 1

    def test_augment_prompt_includes_reason(self):
        l = _learner()
        l.record_correction("navigate to lab", _nav_orig(), _nav_corr(), "overshoot")
        result = l.augment_prompt("navigate to lab", "base", skill_id="navigate_to")
        assert "overshoot" in result


# ---------------------------------------------------------------------------
# Agent integration — correction_learner kwarg
# ---------------------------------------------------------------------------

import apyrobo.skills.builtins  # noqa: F401
from apyrobo import Robot
from apyrobo.core.schemas import TaskStatus
from apyrobo.skills.agent import Agent
from apyrobo.skills.executor import SkillExecutor


def _robot():
    return Robot.discover("mock://turtlebot4")


def _success_result():
    from apyrobo.core.schemas import TaskResult
    return TaskResult(task_name="t", status=TaskStatus.COMPLETED, steps_completed=1, steps_total=1)


class TestAgentCorrectionIntegration:
    def test_execute_accepts_correction_learner(self):
        robot = _robot()
        agent = Agent(provider="rule")
        learner = CorrectionLearner(CorrectionStore(":memory:"))
        with patch.object(SkillExecutor, "execute_graph", return_value=_success_result()):
            result = agent.execute("stop the robot", robot, correction_learner=learner)
        assert result.status == TaskStatus.COMPLETED

    def test_correction_learner_augment_called(self):
        robot = _robot()
        agent = Agent(provider="rule")
        learner = MagicMock()
        learner.augment_prompt.return_value = "stop the robot"
        with patch.object(SkillExecutor, "execute_graph", return_value=_success_result()):
            agent.execute("stop the robot", robot, correction_learner=learner)
        learner.augment_prompt.assert_called_once()

    def test_record_correction_convenience_method(self):
        robot = _robot()
        agent = Agent(provider="rule")
        learner = CorrectionLearner(CorrectionStore(":memory:"))
        with patch.object(SkillExecutor, "execute_graph", return_value=_success_result()):
            agent.execute("stop the robot", robot, correction_learner=learner)
        c = agent.record_correction(_nav_orig(), _nav_corr(), "test")
        assert isinstance(c, Correction)
        assert c.task_description == "stop the robot"

    def test_record_correction_no_learner_returns_none(self):
        agent = Agent(provider="rule")
        result = agent.record_correction(_nav_orig(), _nav_corr())
        assert result is None

    def test_correction_learner_stored_on_agent(self):
        robot = _robot()
        agent = Agent(provider="rule")
        learner = CorrectionLearner(CorrectionStore(":memory:"))
        with patch.object(SkillExecutor, "execute_graph", return_value=_success_result()):
            agent.execute("stop the robot", robot, correction_learner=learner)
        assert agent._correction_learner is learner

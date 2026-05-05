"""Tests for long-horizon planning (milestone decomposition and execution)."""
from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

import apyrobo.skills.builtins  # noqa: F401

from apyrobo import Robot
from apyrobo.skills.longterm import (
    LongHorizonPlan,
    LongHorizonPlanner,
    Milestone,
    MockLongHorizonPlanner,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _robot():
    return Robot.discover("mock://turtlebot4")


def _make_plan(milestones: list[Milestone], goal: str = "test goal") -> LongHorizonPlan:
    import uuid
    from datetime import datetime
    return LongHorizonPlan(
        plan_id=str(uuid.uuid4()),
        goal=goal,
        milestones=milestones,
        created_at=datetime(2026, 1, 1),
    )


# ---------------------------------------------------------------------------
# Milestone
# ---------------------------------------------------------------------------

class TestMilestone:
    def test_fields_stored(self):
        m = Milestone(
            milestone_id="m1",
            description="Go to lab A",
            skills=[{"skill_id": "navigate_to", "parameters": {"x": 1.0, "y": 2.0}}],
            depends_on=["m0"],
            assigned_robot="tb4_1",
            status="completed",
        )
        assert m.milestone_id == "m1"
        assert m.description == "Go to lab A"
        assert m.assigned_robot == "tb4_1"
        assert m.status == "completed"

    def test_default_status_is_pending(self):
        m = Milestone(milestone_id="m1", description="x", skills=[])
        assert m.status == "pending"

    def test_depends_on_defaults_empty(self):
        m = Milestone(milestone_id="m1", description="x", skills=[])
        assert m.depends_on == []


# ---------------------------------------------------------------------------
# LongHorizonPlan serialisation
# ---------------------------------------------------------------------------

class TestLongHorizonPlanJSON:
    def _plan(self):
        milestones = [
            Milestone("m1", "step 1", [{"skill_id": "stop", "parameters": {}}]),
            Milestone("m2", "step 2", [], depends_on=["m1"], status="completed"),
        ]
        return _make_plan(milestones, goal="do everything")

    def test_to_json_contains_goal(self):
        data = json.loads(self._plan().to_json())
        assert data["goal"] == "do everything"

    def test_to_json_contains_plan_id(self):
        plan = self._plan()
        data = json.loads(plan.to_json())
        assert data["plan_id"] == plan.plan_id

    def test_to_json_contains_milestones(self):
        data = json.loads(self._plan().to_json())
        assert len(data["milestones"]) == 2

    def test_round_trip_goal(self):
        plan = self._plan()
        assert LongHorizonPlan.from_json(plan.to_json()).goal == plan.goal

    def test_round_trip_plan_id(self):
        plan = self._plan()
        assert LongHorizonPlan.from_json(plan.to_json()).plan_id == plan.plan_id

    def test_round_trip_milestone_ids(self):
        plan = self._plan()
        restored = LongHorizonPlan.from_json(plan.to_json())
        ids = [m.milestone_id for m in restored.milestones]
        assert ids == ["m1", "m2"]

    def test_round_trip_milestone_statuses(self):
        plan = self._plan()
        restored = LongHorizonPlan.from_json(plan.to_json())
        assert restored.milestones[1].status == "completed"

    def test_round_trip_depends_on(self):
        plan = self._plan()
        restored = LongHorizonPlan.from_json(plan.to_json())
        assert restored.milestones[1].depends_on == ["m1"]

    def test_round_trip_skills(self):
        plan = self._plan()
        restored = LongHorizonPlan.from_json(plan.to_json())
        assert restored.milestones[0].skills == [{"skill_id": "stop", "parameters": {}}]


# ---------------------------------------------------------------------------
# MockLongHorizonPlanner — plan()
# ---------------------------------------------------------------------------

class TestMockPlannerPlan:
    def test_returns_three_milestones_by_default(self):
        planner = MockLongHorizonPlanner()
        plan = planner.plan("clean the lab")
        assert len(plan.milestones) == 3

    def test_plan_has_correct_goal(self):
        planner = MockLongHorizonPlanner()
        plan = planner.plan("clean the lab")
        assert plan.goal == "clean the lab"

    def test_plan_records_call(self):
        planner = MockLongHorizonPlanner()
        planner.plan("goal A")
        planner.plan("goal B")
        assert len(planner.plan_calls) == 2
        assert planner.plan_calls[0]["goal"] == "goal A"

    def test_plan_milestones_start_pending(self):
        planner = MockLongHorizonPlanner()
        plan = planner.plan("x")
        assert all(m.status == "pending" for m in plan.milestones)

    def test_set_plan_overrides_default(self):
        planner = MockLongHorizonPlanner()
        planner.set_plan([
            Milestone("a1", "only step", []),
        ])
        plan = planner.plan("custom goal")
        assert len(plan.milestones) == 1
        assert plan.milestones[0].milestone_id == "a1"


# ---------------------------------------------------------------------------
# MockLongHorizonPlanner — execute()
# ---------------------------------------------------------------------------

class TestMockPlannerExecute:
    def test_execute_completes_all_milestones(self):
        planner = MockLongHorizonPlanner()
        plan = planner.plan("x")
        result = planner.execute(plan)
        assert result["status"] == "completed"
        assert result["total"] == 3

    def test_execute_returns_completed_ids(self):
        planner = MockLongHorizonPlanner()
        plan = planner.plan("x")
        result = planner.execute(plan)
        assert set(result["completed"]) == {"m1", "m2", "m3"}

    def test_execute_calls_on_milestone_complete(self):
        planner = MockLongHorizonPlanner()
        plan = planner.plan("x")
        fired = []
        planner.execute(plan, on_milestone_complete=fired.append)
        assert len(fired) == 3

    def test_execute_respects_dependency_order(self):
        planner = MockLongHorizonPlanner()
        plan = planner.plan("x")
        fired_order = []
        planner.execute(plan, on_milestone_complete=lambda m: fired_order.append(m.milestone_id))
        # m1 before m2, m2 before m3 (linear chain)
        assert fired_order.index("m1") < fired_order.index("m2")
        assert fired_order.index("m2") < fired_order.index("m3")

    def test_execute_summary_dict_keys(self):
        planner = MockLongHorizonPlanner()
        plan = planner.plan("x")
        result = planner.execute(plan)
        assert {"goal", "status", "completed", "failed", "total"} <= set(result)


# ---------------------------------------------------------------------------
# MockLongHorizonPlanner — resume()
# ---------------------------------------------------------------------------

class TestMockPlannerResume:
    def test_resume_skips_completed_milestones(self):
        planner = MockLongHorizonPlanner()
        plan = planner.plan("x")
        plan.milestones[0].status = "completed"  # m1 already done
        result = planner.resume(plan)
        # m1 was already done, only m2/m3 should be in completed
        assert "m1" not in result["completed"]
        assert "m2" in result["completed"]
        assert "m3" in result["completed"]

    def test_resume_resets_running_to_pending(self):
        planner = MockLongHorizonPlanner()
        plan = planner.plan("x")
        plan.milestones[1].status = "running"  # crashed mid-flight
        result = planner.resume(plan)
        assert result["status"] == "completed"

    def test_resume_fully_completed_plan_noop(self):
        planner = MockLongHorizonPlanner()
        plan = planner.plan("x")
        for m in plan.milestones:
            m.status = "completed"
        result = planner.resume(plan)
        assert result["completed"] == []  # nothing new to complete
        assert result["total"] == 3


# ---------------------------------------------------------------------------
# LongHorizonPlanner — _parse_milestones
# ---------------------------------------------------------------------------

class TestParseMillestones:
    def _planner(self):
        return LongHorizonPlanner(agent=MagicMock())

    def test_parses_valid_json(self):
        text = json.dumps([
            {"milestone_id": "m1", "description": "Go", "skills": [], "depends_on": []},
        ])
        milestones = self._planner()._parse_milestones(text)
        assert len(milestones) == 1
        assert milestones[0].milestone_id == "m1"

    def test_strips_markdown_fence(self):
        text = "```json\n" + json.dumps([
            {"milestone_id": "m1", "description": "Go", "skills": [], "depends_on": []},
        ]) + "\n```"
        milestones = self._planner()._parse_milestones(text)
        assert len(milestones) == 1

    def test_empty_response_returns_empty(self):
        milestones = self._planner()._parse_milestones("")
        assert milestones == []

    def test_invalid_json_returns_empty(self):
        milestones = self._planner()._parse_milestones("not json at all")
        assert milestones == []

    def test_skills_preserved(self):
        skills = [{"skill_id": "navigate_to", "parameters": {"x": 1.0}}]
        text = json.dumps([
            {"milestone_id": "m1", "description": "x", "skills": skills, "depends_on": []},
        ])
        ms = self._planner()._parse_milestones(text)
        assert ms[0].skills == skills

    def test_depends_on_preserved(self):
        text = json.dumps([
            {"milestone_id": "m2", "description": "x", "skills": [], "depends_on": ["m1"]},
        ])
        ms = self._planner()._parse_milestones(text)
        assert ms[0].depends_on == ["m1"]


# ---------------------------------------------------------------------------
# LongHorizonPlanner — plan() with mocked LLM
# ---------------------------------------------------------------------------

class TestLongHorizonPlannerPlan:
    def test_plan_calls_llm_decompose(self):
        planner = LongHorizonPlanner(agent=MagicMock())
        llm_response = json.dumps([
            {"milestone_id": "m1", "description": "Do it", "skills": [], "depends_on": []},
        ])
        with patch.object(planner, "_llm_decompose", return_value=llm_response):
            plan = planner.plan("achieve world peace")
        assert plan.goal == "achieve world peace"
        assert len(plan.milestones) == 1

    def test_plan_passes_time_budget_to_llm(self):
        planner = LongHorizonPlanner(agent=MagicMock())
        captured = {}

        def _fake(goal, robots, time_budget_s):
            captured["time_budget_s"] = time_budget_s
            return "[]"

        with patch.object(planner, "_llm_decompose", side_effect=_fake):
            planner.plan("goal", time_budget_s=3600.0)

        assert captured["time_budget_s"] == 3600.0

    def test_plan_generates_unique_plan_id(self):
        planner = LongHorizonPlanner(agent=MagicMock())
        with patch.object(planner, "_llm_decompose", return_value="[]"):
            p1 = planner.plan("goal")
            p2 = planner.plan("goal")
        assert p1.plan_id != p2.plan_id


# ---------------------------------------------------------------------------
# LongHorizonPlanner — execute() with mocked _execute_milestone
# ---------------------------------------------------------------------------

class TestLongHorizonPlannerExecute:
    def _planner(self):
        return LongHorizonPlanner(agent=MagicMock())

    def test_executes_all_milestones(self):
        planner = self._planner()
        plan = _make_plan([
            Milestone("m1", "s1", []),
            Milestone("m2", "s2", [], depends_on=["m1"]),
        ])
        with patch.object(planner, "_execute_milestone", return_value=True):
            result = planner.execute(plan)
        assert result["status"] == "completed"
        assert len(result["completed"]) == 2

    def test_stops_on_first_failure(self):
        planner = self._planner()
        plan = _make_plan([
            Milestone("m1", "s1", []),
            Milestone("m2", "s2", [], depends_on=["m1"]),
        ])
        with patch.object(planner, "_execute_milestone", return_value=False):
            result = planner.execute(plan)
        assert result["status"] == "failed"
        assert "m1" in result["failed"]
        assert "m2" not in result["completed"]

    def test_parallel_independent_milestones(self):
        """M2 and M3 both depend only on M1 — they run when M1 is done."""
        planner = self._planner()
        plan = _make_plan([
            Milestone("m1", "s1", []),
            Milestone("m2", "s2", [], depends_on=["m1"]),
            Milestone("m3", "s3", [], depends_on=["m1"]),
        ])
        # Complete m1 first so m2 and m3 become ready simultaneously
        plan.milestones[0].status = "completed"

        with patch.object(planner, "_execute_milestone", return_value=True):
            result = planner.execute(plan)

        assert result["status"] == "completed"
        assert set(result["completed"]) == {"m2", "m3"}

    def test_calls_on_milestone_complete_callback(self):
        planner = self._planner()
        plan = _make_plan([Milestone("m1", "s1", [])])
        fired = []
        with patch.object(planner, "_execute_milestone", return_value=True):
            planner.execute(plan, on_milestone_complete=fired.append)
        assert len(fired) == 1
        assert fired[0].milestone_id == "m1"


# ---------------------------------------------------------------------------
# LongHorizonPlanner — resume()
# ---------------------------------------------------------------------------

class TestLongHorizonPlannerResume:
    def test_resume_resets_running_to_pending(self):
        planner = LongHorizonPlanner(agent=MagicMock())
        plan = _make_plan([
            Milestone("m1", "s1", [], status="completed"),
            Milestone("m2", "s2", [], depends_on=["m1"], status="running"),
        ])
        executed = []
        with patch.object(
            planner, "_execute_milestone",
            side_effect=lambda m, p: executed.append(m.milestone_id) or True,
        ):
            result = planner.resume(plan)
        assert "m2" in executed
        assert result["status"] == "completed"

    def test_resume_skips_completed(self):
        planner = LongHorizonPlanner(agent=MagicMock())
        plan = _make_plan([
            Milestone("m1", "s1", [], status="completed"),
            Milestone("m2", "s2", [], depends_on=["m1"]),
        ])
        executed = []
        with patch.object(
            planner, "_execute_milestone",
            side_effect=lambda m, p: executed.append(m.milestone_id) or True,
        ):
            planner.resume(plan)
        assert "m1" not in executed
        assert "m2" in executed

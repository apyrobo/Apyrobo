"""Smoke tests for the orchestration-flow demo (demos/orchestration_flow/flow.py).

The demo's promise is that every stage panel shows *real* pipeline data, so
the tests assert exactly that: the stages come from the genuine
Agent/SkillGraph/SafetyEnforcer objects, not hard-coded strings.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1] / "demos" / "orchestration_flow" / "flow.py"


@pytest.fixture(scope="module")
def flow():
    spec = importlib.util.spec_from_file_location("orchestration_flow", DEMO)
    module = importlib.util.module_from_spec(spec)
    sys.modules["orchestration_flow"] = module
    spec.loader.exec_module(module)
    return module


def _stage(stages, key):
    return next(s for s in stages if s.key == key)


def _text(stage) -> str:
    return " | ".join(line for line, _ in stage.lines)


class TestRealPipelineData:
    def test_discover_lists_real_capabilities(self, flow):
        stages, _ = flow.run_pipeline()
        text = _text(_stage(stages, "discover"))
        # MockAdapter declares navigate/pick/place with max_speed 1.5.
        assert "navigate" in text and "pick" in text and "place" in text
        assert "1.5" in text

    def test_graph_stage_matches_planned_skills(self, flow):
        stages, plan_names = flow.run_pipeline()
        graph_lines = [line for line, _ in _stage(stages, "graph").lines]
        assert len(graph_lines) == len(plan_names) >= 1
        for name, line in zip(plan_names, graph_lines, strict=True):
            assert name in line
            assert "[" in line and "]" in line  # capability tag

    def test_safety_shows_the_real_clamp(self, flow):
        stages, _ = flow.run_pipeline()
        text = _text(_stage(stages, "safety"))
        # SafetyEnforcer clamps REQUESTED_SPEED (2.5) to the strict max (0.5).
        assert f"{flow.REQUESTED_SPEED}" in text
        assert "0.5" in text
        assert "REJECTED" in text

    def test_plan_is_nonempty(self, flow):
        _, plan_names = flow.run_pipeline()
        assert plan_names  # the rule planner found skills for the delivery task


class TestPathGeometry:
    def test_path_starts_at_start_and_ends_at_dock(self, flow):
        path = flow.build_path()
        assert path[0][1] == flow.START
        assert path[-1][1] == flow.DOCK

    def test_path_avoids_the_nogo_zone(self, flow):
        z = flow.NOGO
        for _, (x, y) in flow.build_path():
            inside = (z["x_min"] < x < z["x_max"]) and (z["y_min"] < y < z["y_max"])
            assert not inside, f"path point ({x:.1f},{y:.1f}) enters the no-go zone"

    def test_every_leg_skill_is_a_planned_skill(self, flow):
        _, plan_names = flow.run_pipeline()
        leg_skills = {leg[0] for leg in flow.LEGS}
        assert leg_skills <= set(plan_names)

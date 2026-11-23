"""benchmarks/benchmark_suite.py times tasks the robot really carries out.

It used to plan "navigate_to", "pick_object" and "navigate_to_A" with the rule
planner, which reads no target in them, drives to a made-up (0, 0) and reports
completed; the table then timed a robot going nowhere and printed
success_rate 1.0 without checking anything.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from apyrobo.mcp_server import build_server

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / "benchmarks" / "benchmark_suite.py"


@pytest.fixture()
def suite(monkeypatch):
    name = "benchmark_suite_under_test"
    spec = importlib.util.spec_from_file_location(name, SUITE)
    module = importlib.util.module_from_spec(spec)
    # dataclasses look the module up while the class body runs.
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _plan(task: str) -> dict:
    server = build_server("mock://bench_check")
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "plan_task", "arguments": {"task": task}},
        }
    )
    return json.loads(response["result"]["content"][0]["text"])


def _tasks(suite) -> list[str]:
    return sorted(
        {suite.NAVIGATE_TASK, suite.PICK_AND_PLACE_TASK, *suite.PATROL_TASKS, *suite.FLEET_TASKS}
    )


def test_every_benchmark_task_is_one_apyrobo_mcp_accepts(suite):
    for task in _tasks(suite):
        plan = _plan(task)
        assert plan["can_proceed"] is True, f"{task!r}: {plan.get('reason')}"


def test_the_old_tasks_were_refused():
    for task in ["navigate_to", "navigate_to_A"]:
        assert _plan(task)["can_proceed"] is False


def test_patrol_visits_five_different_waypoints(suite):
    assert len(set(suite.WAYPOINTS)) == 5


def test_every_run_completes(suite):
    results = suite.run_all(2)
    assert [r.task_name for r in results] == [
        "navigate_to_point", "pick_and_place", "patrol_route",
        "fleet_coordination", "safety_policy",
    ]
    for r in results:
        assert r.runs == 2, r.task_name
        assert r.success_rate == 1.0, r.task_name
        assert r.to_dict()["success_rate"] == 1.0


def test_a_task_with_no_target_counts_as_not_completed(suite, monkeypatch):
    monkeypatch.setattr(suite, "NAVIGATE_TASK", "navigate_to")
    result = suite.bench_navigate(1)
    assert result.success_rate == 0.0


def test_a_missed_violation_counts_as_not_completed(suite, monkeypatch):
    actions = [(action, not violates) for action, violates in suite.SAFETY_ACTIONS]
    monkeypatch.setattr(suite, "SAFETY_ACTIONS", actions)
    assert suite.bench_safety_policy(1).success_rate == 0.0


def test_the_script_exits_1_when_a_run_fails(suite, monkeypatch, capsys):
    monkeypatch.setattr(suite, "NAVIGATE_TASK", "navigate_to")
    monkeypatch.setattr(sys, "argv", ["benchmark_suite.py", "--iterations", "1", "--json"])
    with pytest.raises(SystemExit) as exc:
        suite.main()
    assert exc.value.code == 1
    assert "navigate_to_point" in capsys.readouterr().err


def test_the_readme_points_at_the_workflow_that_exists():
    readme = (ROOT / "benchmarks" / "README.md").read_text(encoding="utf-8")
    paths = re.findall(r"`([^`]*benchmark\.yml)`", readme)
    assert paths
    for path in paths:
        assert (ROOT / path).is_file(), path

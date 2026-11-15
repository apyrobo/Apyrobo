"""
examples/03_fleet.py runs tasks the rule planner reads exactly: every task
passes the checks apyrobo plan / execute / mcp make, and each robot ends at
the place its last task named, not at a guessed (0, 0).
"""
from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

import pytest

from apyrobo import Agent, Robot
from apyrobo.mcp_server import RobotToolbox
from apyrobo.safety.enforcer import SafetyEnforcer

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "03_fleet.py"


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("fleet_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("index", range(5))
def test_every_task_is_planned_without_a_guess(example, index):
    task = example.TASKS[index]
    robot = Robot.discover("mock://fleet-check")
    toolbox = RobotToolbox(robot, SafetyEnforcer(robot), Agent(provider="rule"), "mock://fleet-check")
    _, plan = toolbox._plan(task)
    assert plan.get("reason") is None, f"{task!r}: {plan.get('reason')}"


def test_each_robot_ends_where_its_task_said(example):
    results = example.main()
    assert len(results) == len(example.TASKS)
    for _robot_id, task, status, position in results:
        assert status == "completed"
        x, y = (float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", task.rsplit("(", 1)[1]))
        assert math.dist(position, (x, y)) < 1e-6, f"{task!r} ended at {position}"

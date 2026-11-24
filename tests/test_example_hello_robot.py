"""
examples/01_hello_robot.py and examples/02_custom_skill.py, the first two
examples a new user runs, work as written on mock:// and say what they do.

01 used to print "Task: 'navigate to position 3, 5 and pick up the object'"
while it ran "deliver package from position 1, 2 to position 3, 5", so the
output described a task that never ran.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from apyrobo import Agent, Robot
from apyrobo.mcp_server import RobotToolbox
from apyrobo.safety.enforcer import SafetyEnforcer

ROOT = Path(__file__).resolve().parents[1]
HELLO = ROOT / "examples" / "01_hello_robot.py"
CUSTOM_SKILL = ROOT / "examples" / "02_custom_skill.py"


def _run(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture(scope="module")
def hello():
    spec = importlib.util.spec_from_file_location("hello_robot_example", HELLO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hello_run():
    return _run(HELLO)


def test_hello_robot_prints_the_task_it_runs(hello, hello_run):
    assert hello_run.returncode == 0, hello_run.stderr
    task_lines = [line for line in hello_run.stdout.splitlines() if line.startswith("Task:")]
    assert task_lines == [f"Task: {hello.TASK!r}"]


def test_hello_robot_task_passes_the_mcp_checks(hello):
    robot = Robot.discover("mock://hello-check")
    toolbox = RobotToolbox(robot, SafetyEnforcer(robot), Agent(provider="rule"), "mock://hello-check")
    _, plan = toolbox._plan(hello.TASK)
    assert plan.get("reason") is None, plan.get("reason")
    targets = [
        (step["parameters"]["x"], step["parameters"]["y"])
        for step in plan["steps"]
        if step["skill_id"].startswith("navigate_to")
    ]
    assert targets == [(1.0, 2.0), (3.0, 5.0)]


def test_hello_robot_completes_and_ends_at_the_drop_off(hello_run):
    assert hello_run.returncode == 0, hello_run.stderr
    out = hello_run.stdout
    assert "Status : completed" in out
    assert "Steps  : 4/4" in out
    assert "Position after task: (3.0, 5.0)" in out
    assert "Error" not in out


def test_custom_skill_example_runs_its_skill():
    result = _run(CUSTOM_SKILL)
    assert result.returncode == 0, result.stderr
    assert "scanning... found 7 items" in result.stdout
    assert "Status: completed  (1/1 steps)" in result.stdout

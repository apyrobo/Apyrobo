"""
examples/demo_full.py runs its agent through the safety policy it builds.

It used to build a SafetyEnforcer and then pass the raw robot to
agent.execute, so no move went through the policy and its closing
"Safety: No violations" held for any task. Now the agent drives
safe_robot, and a second task into the restricted area is refused before
the robot moves.
"""
from __future__ import annotations

import contextlib
import io
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo_full.py"


@pytest.fixture(scope="module")
def demo_run():
    return subprocess.run(
        [sys.executable, str(DEMO)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture(scope="module")
def demo_globals():
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(str(DEMO), run_name="demo_full_check")


def test_demo_runs_and_delivers(demo_run):
    assert demo_run.returncode == 0, demo_run.stderr
    out = demo_run.stdout
    assert "Task: 'deliver package from (1, 2) to (5, 5)'" in out
    assert "Result: completed\n  Steps: 4/4" in out


def test_demo_policy_refuses_the_restricted_area(demo_run):
    assert demo_run.returncode == 0, demo_run.stderr
    out = demo_run.stdout
    assert "Task: 'go to (9, 9)'" in out
    assert "Result: failed\n  Steps: 0/1" in out
    assert "Reason: Skill 'navigate_to_0' failed: Position (9, 9) is inside collision zone" in out
    assert "Safety violations: 1" in out
    assert "No violations" not in out


def test_demo_agent_drives_through_the_enforcer(demo_globals):
    safe_robot = demo_globals["safe_robot"]
    robot = demo_globals["robot"]
    assert len(safe_robot.violations) == 1
    assert safe_robot.violations[0]["type"] == "collision_zone"
    # The refused task left the robot where the delivery ended, outside the zone.
    x, y = robot.get_position()
    assert (x, y) == pytest.approx((5.0, 5.0))

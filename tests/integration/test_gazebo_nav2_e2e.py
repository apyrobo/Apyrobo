"""
End-to-end integration test: natural-language task → rule-based plan →
Nav2 NavigateToPose → a physics-simulated robot navigates in Gazebo.

This is the flagship-stack proof. Where test_gazebo_turtlebot.py shows the
adapter's cmd_vel fallback moves a physics robot, this test runs the whole
pipeline a user touches:

    Agent(provider="rule").execute("navigate to (-1.2, -0.5)", robot)
      → plan: navigate_to skill with extracted coordinates
      → SkillExecutor → builtins navigate_to → robot.move()
      → ROS2Adapter prefers the live Nav2 NavigateToPose action
      → slam_toolbox + Nav2 drive the TurtleBot3 burger through
        turtlebot3_world, with physics in the loop.

Run inside the Docker 'gazebo-nav' profile:

    docker compose -f docker/docker-compose.yml --profile gazebo-nav up \
        --abort-on-container-exit --exit-code-from gazebo-nav-test

Assertions are progress-shaped, not exact-pose-shaped: SLAM and physics are
in the loop, so the point is that the NL task measurably gets the robot to
the goal region via Nav2 — not centimeter repeatability.
"""
from __future__ import annotations

import contextlib
import math
import os

import pytest

# Gated behind APYROBO_GAZEBO_NAV, set only by the compose 'gazebo-nav-test'
# service — mirrors the APYROBO_GAZEBO gate in test_gazebo_turtlebot.py and
# keeps this file out of the plain integration job and the cmd_vel-only
# gazebo job (where no Nav2 server exists).
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("APYROBO_GAZEBO_NAV"),
        reason="Nav2-in-Gazebo sim not present — set APYROBO_GAZEBO_NAV=1 "
        "(docker compose 'gazebo-nav' profile) to run",
    ),
]

# Spawn pose in gazebo_nav_boot.sh and the goal the NL task names: a short
# hop through free corridor, well inside the default 60 s skill timeout.
START = (-2.0, -0.5)
GOAL = (-1.2, -0.5)
TASK = f"navigate to ({GOAL[0]}, {GOAL[1]})"


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@pytest.fixture(scope="module")
def robot():
    from apyrobo.core.robot import Robot
    from apyrobo.core.ros2_bridge import _ROS2NodeManager

    # Nav2 activation after SLAM bringup can be slow on CI — wait generously
    # for the action server; the compose healthcheck has already seen it.
    bot = Robot.discover(
        "ros2://burger",
        nav2_server_wait_sec=90.0,
        odom_wait_sec=60.0,
    )
    yield bot
    with contextlib.suppress(Exception):
        _ROS2NodeManager.shutdown()


@pytest.mark.integration
def test_nav2_action_server_discovered(robot):
    """The adapter found the live NavigateToPose server (not the fallback)."""
    adapter = robot._adapter
    assert adapter._has_nav2, (
        "NavigateToPose action server not found — Nav2 bringup failed or "
        "the adapter fell back to cmd_vel. Check gazebo-nav-sim logs."
    )
    assert adapter._has_odom, "No /odom from Gazebo — sim not up"


@pytest.mark.integration
def test_nl_task_plans_navigate_skill(robot):
    """The rule agent turns the NL task into navigate_to with the coords."""
    from apyrobo.skills.agent import Agent

    graph = Agent(provider="rule").plan(TASK, robot)
    order = graph.get_execution_order()
    nav_steps = [s for s in order if "navigate" in s.name.lower() or "navigate" in s.skill_id.lower()]
    assert nav_steps, f"no navigate skill planned for {TASK!r}: {[s.name for s in order]}"
    params = nav_steps[-1].parameters
    assert params.get("x") == GOAL[0] and params.get("y") == GOAL[1], (
        f"coordinates not extracted into the plan: {params}"
    )


@pytest.mark.integration
def test_nl_task_navigates_the_robot_via_nav2(robot):
    """The whole pipeline: NL task in, robot at the goal region out."""
    from apyrobo.core.schemas import TaskStatus
    from apyrobo.skills.agent import Agent

    adapter = robot._adapter
    start = adapter.position
    d_before = _dist(start, GOAL)

    agent = Agent(provider="rule")
    result = agent.execute(task=TASK, robot=robot)

    assert result.status == TaskStatus.COMPLETED, (
        f"NL task did not complete: {result.status} error={result.error!r}"
    )

    end = adapter.position
    d_after = _dist(end, GOAL)
    travelled = _dist(start, end)

    assert travelled > 0.3, (
        f"robot barely moved (Δ={travelled:.2f} m) — Nav2 accepted the goal "
        f"but nothing drove the physics robot"
    )
    assert d_after < d_before - 0.3, (
        f"no progress toward goal: {d_before:.2f} m → {d_after:.2f} m "
        f"(start={start}, end={end}, goal={GOAL})"
    )
    assert d_after < 0.5, (
        f"did not reach the goal region: {d_after:.2f} m away "
        f"(end={end}, goal={GOAL})"
    )


@pytest.mark.integration
def test_stop_after_nav_is_safe(robot):
    """stop() stays safe and the adapter queryable after Nav2 activity."""
    robot.stop()
    robot.stop()  # idempotent
    pos = robot.get_position()
    assert isinstance(pos, tuple) and len(pos) == 2
    assert isinstance(robot._adapter.get_health(), dict)

"""
Integration test: the real ros2:// adapter drives a physics-simulated
TurtleBot3 in headless Gazebo.

Unlike test_ros2_adapter.py (which talks to the kinematic fake_turtlebot4
node), this exercises the adapter against Gazebo Classic physics: the burger
model's gazebo_ros diff-drive plugin subscribes /cmd_vel and publishes /odom.
There is no Nav2 here, so move() uses the adapter's cmd_vel proportional
controller — the robot is driven by real velocity commands and its motion is
read back from real odometry.

Run inside the Docker 'gazebo' profile:

    docker compose -f docker/docker-compose.yml --profile gazebo up \
        --abort-on-container-exit --exit-code-from gazebo-test

Assertions are intentionally loose (progress, not exact poses): the point is
that APYROBO commands make a physics robot actually move, with friction and
inertia in the loop.
"""
from __future__ import annotations

import contextlib
import math
import time

import pytest


def _rclpy_available() -> bool:
    try:
        import rclpy  # noqa: F401

        return True
    except Exception:
        return False


# Every test here is opt-in (integration) AND requires a live ROS 2 stack;
# outside the Docker 'gazebo' profile it skips cleanly rather than erroring.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _rclpy_available(),
        reason="rclpy not importable — run inside the Docker 'gazebo' profile",
    ),
]


@pytest.fixture(scope="module")
def robot():
    from apyrobo.core.robot import Robot
    from apyrobo.core.ros2_bridge import _ROS2NodeManager

    # No Nav2 in this minimal sim, so keep the nav2 wait short; give odom a
    # generous window since Gazebo + spawn can be slow to start on CI.
    bot = Robot.discover(
        "ros2://burger",
        nav2_server_wait_sec=3.0,
        odom_wait_sec=30.0,
    )
    yield bot
    with contextlib.suppress(Exception):
        _ROS2NodeManager.shutdown()


@pytest.mark.integration
def test_odometry_online_from_gazebo(robot):
    """The adapter receives /odom from the Gazebo diff-drive plugin."""
    adapter = robot._adapter
    assert adapter._has_odom, (
        "No /odom received from Gazebo. Is gazebo_boot.sh running and the "
        "burger spawned on the same ROS_DOMAIN_ID?"
    )
    x, y = adapter.position
    # Spawned at origin.
    assert abs(x) < 0.5 and abs(y) < 0.5, f"expected start ≈ origin, got ({x}, {y})"


@pytest.mark.integration
def test_capabilities_report_cmd_vel_navigation(robot):
    """With /cmd_vel + /odom live (no Nav2), NAVIGATE is declared."""
    from apyrobo.core.schemas import CapabilityType

    caps = robot.capabilities()
    cap_types = {c.capability_type for c in caps.capabilities}
    assert CapabilityType.NAVIGATE in cap_types, (
        f"expected NAVIGATE via cmd_vel; got {cap_types}"
    )


@pytest.mark.integration
def test_move_command_drives_the_physics_robot(robot):
    """APYROBO move() → cmd_vel → the burger actually travels in Gazebo."""
    adapter = robot._adapter
    x0, y0 = adapter.position

    # Drive forward ~1 m. move() blocks on the proportional controller; give
    # the sim a moment to settle regardless of exactly where it stopped.
    robot.move(x=1.0, y=0.0)
    time.sleep(1.0)

    x1, y1 = adapter.position
    travelled = math.hypot(x1 - x0, y1 - y0)
    assert travelled > 0.3, (
        f"robot barely moved in Gazebo (Δ={travelled:.2f} m from "
        f"({x0:.2f},{y0:.2f}) to ({x1:.2f},{y1:.2f})) — the cmd_vel→physics "
        f"loop is not closing"
    )
    # Progress should be toward +x, not sideways/backwards.
    assert (x1 - x0) > 0.2, f"expected forward (+x) progress, Δx={x1 - x0:.2f}"


@pytest.mark.integration
def test_stop_is_safe_and_leaves_robot_queryable(robot):
    """stop() must always work and keep the adapter queryable (safety path)."""
    robot.move(x=0.5, y=0.5)
    robot.stop()          # must not raise, in any state
    robot.stop()          # idempotent
    # Still queryable after stop.
    pos = robot.get_position()
    assert isinstance(pos, tuple) and len(pos) == 2
    health = robot._adapter.get_health()
    assert isinstance(health, dict)

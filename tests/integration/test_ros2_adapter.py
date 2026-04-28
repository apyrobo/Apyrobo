"""
Integration tests for the ros2:// adapter.

Requires rclpy and nav2_msgs — run inside the Docker integration container:

    docker compose -f docker/docker-compose.yml --profile integration up \
        --abort-on-container-exit

Or locally if ROS 2 Humble is installed:

    ROS_DOMAIN_ID=42 pytest -m integration tests/integration/ -v

Env vars
--------
FAKE_ROBOT_EXTERNAL=1
    Set this when the fake robot is already running as a separate process
    (i.e. in Docker where it is a separate service).  The fixture then skips
    launching a subprocess and just waits for the action server to appear.

ROS_DOMAIN_ID
    All participants must share the same domain ID. Default: 42.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Marker — every test in this module is opt-in
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.integration

_FAKE_ROBOT_SCRIPT = Path(__file__).parent / "fake_turtlebot4.py"

# How long (seconds) to wait for the fake robot to come up before connecting
_STARTUP_WAIT_SEC = 5.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fake_robot():
    """
    Ensure the fake TurtleBot4 node is running.

    When FAKE_ROBOT_EXTERNAL=1, the fake robot is assumed to already be up
    (the Docker 'fake-robot' service).  Otherwise a subprocess is started.
    """
    if os.environ.get("FAKE_ROBOT_EXTERNAL"):
        # External process — just wait briefly for it to publish odom
        time.sleep(_STARTUP_WAIT_SEC)
        yield None
        return

    env = {**os.environ, "ROS_DOMAIN_ID": os.environ.get("ROS_DOMAIN_ID", "42")}
    proc = subprocess.Popen(
        [sys.executable, str(_FAKE_ROBOT_SCRIPT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # Wait for node to initialise and publish first odom
    time.sleep(_STARTUP_WAIT_SEC)
    if proc.poll() is not None:
        out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
        pytest.fail(f"fake_turtlebot4.py exited unexpectedly:\n{out}")

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def robot(fake_robot):
    """
    Connect to the fake TurtleBot4 via the ros2:// adapter.

    Uses generous timeouts so CI on slow machines doesn't flake.
    """
    from apyrobo.core.robot import Robot
    from apyrobo.core.ros2_bridge import _ROS2NodeManager

    bot = Robot.discover(
        "ros2://turtlebot4",
        nav2_server_wait_sec=15.0,
        odom_wait_sec=10.0,
    )
    yield bot

    # Clean up the singleton ROS 2 node after the test module finishes
    try:
        _ROS2NodeManager.shutdown()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_adapter_connects_and_has_odom(robot):
    """Adapter initialised → odometry is online and position is (0, 0)."""
    adapter = robot._adapter
    assert adapter._has_odom, "Adapter should have received at least one /odom message"
    x, y = adapter.position
    # Fake robot starts at origin
    assert abs(x) < 0.5
    assert abs(y) < 0.5


@pytest.mark.integration
def test_adapter_has_nav2(robot):
    """Nav2 action server is discovered within the wait window."""
    adapter = robot._adapter
    assert adapter._has_nav2, (
        "Nav2 action server 'navigate_to_pose' was not found. "
        "Check that fake_turtlebot4.py is running and on the same ROS_DOMAIN_ID."
    )


@pytest.mark.integration
def test_navigate_to_via_agent_completes(robot):
    """
    Run navigate_to (2.0, 1.0) via an apyrobo Agent and verify:
      1. The task completes with status 'completed'.
      2. The robot's reported position is approximately (2.0, 1.0).

    Uses provider='rule' — no LLM or API key needed.
    The RuleBasedProvider extracts coordinates from '(2.0, 1.0)' and
    maps 'navigate' to the navigate_to skill.
    """
    from apyrobo.skills.agent import Agent

    agent = Agent(provider="rule")
    result = agent.execute(task="navigate to (2.0, 1.0)", robot=robot)

    assert result.status == "completed", (
        f"Expected task status 'completed', got {result.status!r}. "
        f"Error: {result.error}"
    )
    assert result.steps_completed >= 1

    # Confirm the adapter tracked the position change via Nav2 feedback
    x, y = robot._adapter.position
    assert abs(x - 2.0) < 0.5, f"Expected x≈2.0, got {x:.3f}"
    assert abs(y - 1.0) < 0.5, f"Expected y≈1.0, got {y:.3f}"


@pytest.mark.integration
def test_direct_move_updates_position(robot):
    """
    Call robot.move() directly (no agent) to (3.0, 0.5) and confirm position.

    This exercises _move_nav2 in isolation, without the skill-planning layer.
    """
    robot.move(x=3.0, y=0.5)

    from apyrobo.core.ros2_bridge import NavState

    adapter = robot._adapter
    assert adapter._nav_state == NavState.SUCCEEDED, (
        f"Expected nav_state SUCCEEDED, got {adapter._nav_state!r}"
    )

    x, y = adapter.position
    assert abs(x - 3.0) < 0.5, f"Expected x≈3.0, got {x:.3f}"
    assert abs(y - 0.5) < 0.5, f"Expected y≈0.5, got {y:.3f}"


@pytest.mark.integration
def test_get_capabilities_reports_navigate(robot):
    """
    Capability introspection returns a NAVIGATE capability because Nav2 is up.
    """
    from apyrobo.core.schemas import CapabilityType

    caps = robot.capabilities(refresh=True)
    cap_types = {c.capability_type for c in caps.capabilities}
    assert CapabilityType.NAVIGATE in cap_types, (
        f"Expected NAVIGATE capability, got: {cap_types}"
    )
    assert caps.metadata.get("nav2_available") is True

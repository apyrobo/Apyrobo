#!/usr/bin/env python3
"""
APYROBO Integration Test — validates the full pipeline in Gazebo.

Run AFTER Gazebo + Nav2 are up (via scripts/launch.sh).

Tests:
    1. ROS 2 connectivity — can we see topics?
    2. Robot discovery — does gazebo:// adapter work?
    3. Capability detection — are sensors and nav detected?
    4. Odometry — is the robot reporting its position?
    5. Sensor pipeline — does lidar/camera data flow?
    6. Navigation — can the robot move to a goal?
    7. Safety enforcement — does speed clamping work?
    8. Skill execution — does a simple skill graph complete?
    9. Agent planning — can the agent plan and execute a task?

Usage:
    python3 scripts/integration_test.py
    python3 scripts/integration_test.py --timeout 60
"""

from __future__ import annotations

import argparse
import sys
import os
import time
import math
import logging

sys.path.insert(0, "/workspace")
os.environ.setdefault("PYTHONPATH", "/workspace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("integration_test")

# ---------------------------------------------------------------------------
# Test framework
# ---------------------------------------------------------------------------

passed = 0
failed = 0
skipped = 0


def test(name, fn, skip_reason=None):
    global passed, failed, skipped
    if skip_reason:
        print(f"  \033[33mSKIP\033[0m  {name}: {skip_reason}")
        skipped += 1
        return
    try:
        fn()
        print(f"  \033[32mPASS\033[0m  {name}")
        passed += 1
    except Exception as e:
        print(f"  \033[31mFAIL\033[0m  {name}: {e}")
        failed += 1


def section(title):
    print(f"\n\033[1m{'='*60}\033[0m")
    print(f"\033[1m  {title}\033[0m")
    print(f"\033[1m{'='*60}\033[0m")


# ---------------------------------------------------------------------------
# Check ROS 2 availability
# ---------------------------------------------------------------------------

try:
    import rclpy
    from rclpy.node import Node
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    print("\033[31mERROR: rclpy not found. Are you inside the Docker container?\033[0m")
    print("  Run: docker compose -f docker/docker-compose.yml exec apyrobo bash")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="APYROBO integration test")
parser.add_argument("--timeout", type=float, default=30.0,
                    help="Timeout for navigation tests (seconds)")
args = parser.parse_args()


# ===================================================================
section("1. ROS 2 Connectivity")
# ===================================================================

def test_ros2_init():
    """Verify rclpy initialises."""
    if not rclpy.ok():
        rclpy.init()
    assert rclpy.ok(), "rclpy.init() failed"

def test_ros2_topics():
    """Verify ROS 2 topics are visible (Gazebo must be running)."""
    node = rclpy.create_node("_apyrobo_test_probe")
    try:
        topics = node.get_topic_names_and_types()
        topic_names = [name for name, _ in topics]
        logger.info("Found %d topics", len(topic_names))
        assert len(topic_names) > 0, "No ROS 2 topics found — is Gazebo running?"
        # Check for essential topics
        essential = ["/odom", "/scan", "/cmd_vel"]
        for t in essential:
            if t not in topic_names:
                logger.warning("Missing expected topic: %s", t)
    finally:
        node.destroy_node()

test("rclpy init", test_ros2_init)
test("ROS 2 topics visible", test_ros2_topics)


# ===================================================================
section("2. Robot Discovery")
# ===================================================================

from apyrobo.core.robot import Robot

robot = None

def test_discover_gazebo():
    global robot
    robot = Robot.discover("gazebo://turtlebot4")
    assert robot is not None
    assert robot.robot_id == "turtlebot4"

test("Discover gazebo://turtlebot4", test_discover_gazebo)


# ===================================================================
section("3. Capability Detection")
# ===================================================================

caps = None

def test_capabilities():
    global caps
    assert robot is not None, "Robot not discovered"
    caps = robot.capabilities()
    logger.info("Robot: %s", caps.name)
    logger.info("Capabilities: %s", [c.name for c in caps.capabilities])
    logger.info("Sensors: %s", [f"{s.sensor_id} ({s.sensor_type.value})" for s in caps.sensors])
    logger.info("Max speed: %s", caps.max_speed)
    assert len(caps.capabilities) > 0, "No capabilities detected"

def test_nav_capability():
    from apyrobo.core.schemas import CapabilityType
    assert caps is not None
    cap_types = {c.capability_type for c in caps.capabilities}
    assert CapabilityType.NAVIGATE in cap_types, "Navigate capability not detected"

def test_sensors_detected():
    assert caps is not None
    assert len(caps.sensors) > 0, "No sensors detected — check topic names"

test("Capabilities query", test_capabilities)
test("Navigate capability present", test_nav_capability)
test("Sensors detected", test_sensors_detected)


# ===================================================================
section("4. Odometry")
# ===================================================================

def test_odom():
    """Check that the robot has a valid position from odometry."""
    assert robot is not None
    adapter = robot._adapter
    pos = adapter.position
    logger.info("Robot position: (%.2f, %.2f)", *pos)
    logger.info("Robot orientation: %.1f°", math.degrees(adapter.orientation))
    # Position should be finite
    assert math.isfinite(pos[0]) and math.isfinite(pos[1]), f"Invalid position: {pos}"

test("Odometry position", test_odom)


# ===================================================================
section("5. Sensor Pipeline")
# ===================================================================

from apyrobo.sensors.pipeline import SensorPipeline
from apyrobo.sensors.ros2_subscribers import ROS2SensorBridge

pipeline = None
sensor_bridge = None

def test_sensor_pipeline():
    global pipeline, sensor_bridge
    assert robot is not None
    from apyrobo.core.ros2_bridge import _ROS2NodeManager
    node = _ROS2NodeManager.get_node()
    pipeline = SensorPipeline()
    sensor_bridge = ROS2SensorBridge(node, pipeline)
    assert sensor_bridge.subscriber_count > 0, "No sensor subscribers created"
    logger.info("Sensor bridge: %d subscribers", sensor_bridge.subscriber_count)

def test_sensor_data_flowing():
    """Wait a few seconds and check that data has arrived."""
    assert pipeline is not None
    logger.info("Waiting 3s for sensor data...")
    time.sleep(3.0)
    count = pipeline.reading_count
    logger.info("Sensor readings received: %d", count)
    assert count > 0, "No sensor data received after 3s"

def test_world_state():
    assert pipeline is not None
    world = pipeline.get_world_state()
    logger.info("World state: %s", world)
    logger.info("  Position: (%.2f, %.2f)", *world.robot_position)
    logger.info("  Obstacles: %d", len(world.obstacles))
    logger.info("  Objects: %d", len(world.detected_objects))

test("Sensor pipeline + ROS 2 bridge", test_sensor_pipeline)
test("Sensor data flowing", test_sensor_data_flowing)
test("World state built", test_world_state)


# ===================================================================
section("6. Navigation")
# ===================================================================

def test_move_short():
    """Command the robot to move a short distance."""
    assert robot is not None
    adapter = robot._adapter
    start_pos = adapter.position
    logger.info("Start: (%.2f, %.2f)", *start_pos)

    # Move 1 meter forward from current position
    target_x = start_pos[0] + 1.0
    target_y = start_pos[1]
    logger.info("Target: (%.2f, %.2f)", target_x, target_y)

    robot.move(x=target_x, y=target_y, speed=0.3)

    end_pos = adapter.position
    logger.info("End: (%.2f, %.2f)", *end_pos)

    # Check we moved at least somewhat
    dist_moved = math.sqrt(
        (end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2
    )
    logger.info("Distance moved: %.2f m", dist_moved)
    assert dist_moved > 0.1, f"Robot didn't move enough ({dist_moved:.2f}m)"

def test_stop():
    """Stop the robot and verify it halts."""
    assert robot is not None
    robot.stop()
    time.sleep(0.5)
    assert not robot._adapter.is_moving, "Robot still moving after stop()"

test("Navigate short distance", test_move_short)
test("Stop command", test_stop)


# ===================================================================
section("7. Safety Enforcement")
# ===================================================================

from apyrobo.safety.enforcer import SafetyEnforcer, SafetyPolicy, SafetyViolation

def test_safety_speed_clamp():
    assert robot is not None
    enforcer = SafetyEnforcer(robot, policy=SafetyPolicy(max_speed=0.5))
    adapter = robot._adapter
    start = adapter.position
    # Request high speed — should be clamped
    enforcer.move(x=start[0] + 0.5, y=start[1], speed=5.0)
    assert len(enforcer.interventions) == 1
    assert enforcer.interventions[0]["type"] == "speed_clamped"
    enforcer.stop()

def test_safety_collision_zone():
    assert robot is not None
    adapter = robot._adapter
    pos = adapter.position
    # Create a zone right where we want to go
    enforcer = SafetyEnforcer(robot, policy=SafetyPolicy(
        collision_zones=[{
            "x_min": pos[0] + 5, "x_max": pos[0] + 10,
            "y_min": pos[1] - 5, "y_max": pos[1] + 5,
        }]
    ))
    try:
        enforcer.move(x=pos[0] + 7, y=pos[1])
        assert False, "Should have raised SafetyViolation"
    except SafetyViolation:
        pass
    assert len(enforcer.violations) == 1

test("Safety: speed clamping", test_safety_speed_clamp)
test("Safety: collision zone rejection", test_safety_collision_zone)


# ===================================================================
section("8. Skill Execution")
# ===================================================================

from apyrobo.skills.skill import BUILTIN_SKILLS
from apyrobo.skills.executor import SkillGraph, SkillExecutor, SkillStatus

def test_skill_navigate():
    assert robot is not None
    adapter = robot._adapter
    start = adapter.position
    executor = SkillExecutor(robot)
    events = []
    executor.on_event(lambda e: events.append(e))
    nav = BUILTIN_SKILLS["navigate_to"]
    status = executor.execute_skill(nav, {"x": start[0] + 0.5, "y": start[1], "speed": 0.3})
    logger.info("Skill result: %s, events: %d", status.value, len(events))
    assert status == SkillStatus.COMPLETED
    assert len(events) >= 2
    robot.stop()

def test_skill_graph():
    assert robot is not None
    adapter = robot._adapter
    pos = adapter.position
    graph = SkillGraph()
    graph.add_skill(BUILTIN_SKILLS["navigate_to"],
                    parameters={"x": pos[0] + 0.3, "y": pos[1], "speed": 0.3})
    graph.add_skill(BUILTIN_SKILLS["report_status"],
                    depends_on=["navigate_to"])
    executor = SkillExecutor(robot)
    from apyrobo.core.schemas import TaskStatus
    result = executor.execute_graph(graph)
    logger.info("Graph result: %s (%d/%d steps)", result.status.value,
                result.steps_completed, result.steps_total)
    assert result.status == TaskStatus.COMPLETED
    robot.stop()

test("Skill: navigate_to", test_skill_navigate)
test("Skill graph: navigate → report", test_skill_graph)


# ===================================================================
section("9. Agent Planning + Execution")
# ===================================================================

from apyrobo.skills.agent import Agent
from apyrobo.core.schemas import TaskStatus

def test_agent_execute():
    assert robot is not None
    adapter = robot._adapter
    pos = adapter.position
    agent = Agent(provider="rule")
    events = []
    result = agent.execute(
        task=f"go to ({pos[0] + 0.5:.1f}, {pos[1]:.1f})",
        robot=robot,
        on_event=lambda e: events.append(e),
    )
    logger.info("Agent result: %s, events: %d", result.status.value, len(events))
    for e in events:
        logger.info("  %s %s: %s", e.status.value, e.skill_id, e.message)
    assert result.status == TaskStatus.COMPLETED
    robot.stop()

test("Agent: plan + execute navigation", test_agent_execute)


# ===================================================================
# Summary
# ===================================================================

robot.stop()  # always stop at end

print(f"\n{'='*60}")
total = passed + failed + skipped
print(f"\033[1m  Integration Test Results\033[0m")
print(f"  \033[32mPassed:  {passed}\033[0m")
if failed:
    print(f"  \033[31mFailed:  {failed}\033[0m")
if skipped:
    print(f"  \033[33mSkipped: {skipped}\033[0m")
print(f"  Total:   {total}")
print(f"{'='*60}")

if failed:
    print(f"\n\033[31m  SOME TESTS FAILED — check Gazebo/Nav2 are running\033[0m")
    sys.exit(1)
print(f"\n\033[32m  ALL INTEGRATION TESTS PASSED ✓\033[0m")
print(f"  Your pipeline works end-to-end in Gazebo!")

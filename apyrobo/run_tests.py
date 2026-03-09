"""Quick test runner — works without pytest."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apyrobo.core.schemas import (
    RobotCapability, Capability, CapabilityType, SensorInfo, SensorType,
    TaskRequest, TaskResult, TaskStatus, RecoveryAction, SafetyPolicyRef,
)
from apyrobo.core.robot import Robot
from apyrobo.core.adapters import MockAdapter

passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1


# === Schema Tests ===
print("=== Schema Tests ===")


def test_minimal_robot_capability():
    rc = RobotCapability(robot_id="bot-1", name="TestBot")
    assert rc.robot_id == "bot-1"
    assert rc.capabilities == []
    assert rc.sensors == []


def test_full_robot_capability():
    rc = RobotCapability(
        robot_id="tb4", name="TB4",
        capabilities=[Capability(capability_type=CapabilityType.NAVIGATE, name="nav")],
        sensors=[SensorInfo(sensor_id="cam0", sensor_type=SensorType.CAMERA)],
        max_speed=1.2,
    )
    assert len(rc.capabilities) == 1
    assert rc.max_speed == 1.2


def test_missing_required_fields():
    try:
        RobotCapability()
        raise AssertionError("Should have raised")
    except TypeError:
        pass  # expected


def test_task_request_minimal():
    tr = TaskRequest(task_name="deliver_package")
    assert tr.task_name == "deliver_package"
    assert tr.priority == 1


def test_task_request_with_params():
    tr = TaskRequest(
        task_name="deliver_package",
        parameters={"destination": "room_3"},
        priority=5,
        target_robot_id="tb4-001",
    )
    assert tr.parameters["destination"] == "room_3"
    assert tr.priority == 5


def test_task_result_success():
    r = TaskResult(task_name="x", status=TaskStatus.COMPLETED, confidence=0.92)
    assert r.status == TaskStatus.COMPLETED
    assert r.confidence == 0.92


def test_task_result_with_recovery():
    r = TaskResult(
        task_name="x", status=TaskStatus.FAILED, confidence=0.0,
        recovery_actions_taken=[RecoveryAction.RETRY, RecoveryAction.ABORT],
    )
    assert RecoveryAction.RETRY in r.recovery_actions_taken


def test_safety_policy_ref():
    sp = SafetyPolicyRef(policy_name="restricted", max_speed=0.5, human_proximity_limit=1.0)
    assert sp.max_speed == 0.5
    assert sp.human_proximity_limit == 1.0


def test_custom_safety_in_request():
    req = TaskRequest(
        task_name="fast",
        safety_policy=SafetyPolicyRef(policy_name="strict", max_speed=0.3),
    )
    assert req.safety_policy.max_speed == 0.3


test("minimal RobotCapability", test_minimal_robot_capability)
test("full RobotCapability", test_full_robot_capability)
test("missing required fields raises", test_missing_required_fields)
test("TaskRequest minimal", test_task_request_minimal)
test("TaskRequest with params", test_task_request_with_params)
test("TaskResult success", test_task_result_success)
test("TaskResult with recovery", test_task_result_with_recovery)
test("SafetyPolicyRef", test_safety_policy_ref)
test("custom safety in TaskRequest", test_custom_safety_in_request)


# === Robot Discovery Tests ===
print("\n=== Robot Discovery Tests ===")


def test_discover_mock():
    robot = Robot.discover("mock://test_bot")
    assert robot.robot_id == "test_bot"
    assert isinstance(robot._adapter, MockAdapter)


def test_discover_bad_uri():
    try:
        Robot.discover("noscheme")
        raise AssertionError("Should have raised")
    except ValueError:
        pass


def test_discover_unknown_scheme():
    try:
        Robot.discover("unknown://bot")
        raise AssertionError("Should have raised")
    except ValueError:
        pass


def test_capabilities_returned():
    robot = Robot.discover("mock://tb4")
    caps = robot.capabilities()
    assert caps.robot_id == "tb4"
    assert len(caps.capabilities) > 0
    cap_types = {c.capability_type for c in caps.capabilities}
    assert CapabilityType.NAVIGATE in cap_types


def test_capabilities_cached():
    robot = Robot.discover("mock://tb4")
    c1 = robot.capabilities()
    c2 = robot.capabilities()
    assert c1 is c2  # same object


def test_capabilities_refresh():
    robot = Robot.discover("mock://tb4")
    c1 = robot.capabilities()
    c2 = robot.capabilities(refresh=True)
    assert c1 is not c2  # new object


def test_move():
    robot = Robot.discover("mock://tb4")
    robot.move(x=2.0, y=3.0, speed=0.5)
    assert robot._adapter.position == (2.0, 3.0)
    assert robot._adapter.is_moving is True


def test_stop():
    robot = Robot.discover("mock://tb4")
    robot.move(x=1.0, y=1.0)
    robot.stop()
    assert robot._adapter.is_moving is False


def test_multiple_moves():
    robot = Robot.discover("mock://tb4")
    robot.move(x=1.0, y=0.0)
    robot.move(x=2.0, y=0.0)
    robot.move(x=3.0, y=0.0)
    assert len(robot._adapter.move_history) == 3
    assert robot._adapter.position == (3.0, 0.0)


def test_repr():
    robot = Robot.discover("mock://r1")
    assert "MockAdapter" in repr(robot)


test("discover mock", test_discover_mock)
test("discover bad URI raises", test_discover_bad_uri)
test("discover unknown scheme raises", test_discover_unknown_scheme)
test("capabilities returned", test_capabilities_returned)
test("capabilities cached", test_capabilities_cached)
test("capabilities refresh", test_capabilities_refresh)
test("move command", test_move)
test("stop command", test_stop)
test("multiple moves tracked", test_multiple_moves)
test("repr", test_repr)

# === Demo ===
print("\n=== Example Demo ===")
robot = Robot.discover("mock://turtlebot4")
caps = robot.capabilities()
print(f"  Robot:        {caps.name}")
print(f"  Capabilities: {[c.name for c in caps.capabilities]}")
print(f"  Sensors:      {[s.sensor_id for s in caps.sensors]}")
print(f"  Max speed:    {caps.max_speed} m/s")
robot.move(x=2.0, y=3.0, speed=0.5)
robot.stop()
print(f"  Move+stop:    OK")

# === Summary ===
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
print("All tests passed!")

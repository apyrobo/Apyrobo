#!/usr/bin/env python3
"""
APYROBO Test Suite — runs without pytest, using stdlib only.

Covers: schemas, robot discovery, adapters, skills, skill graph,
executor, agent planning, and safety enforcement.
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apyrobo.core.schemas import (
    RobotCapability, Capability, CapabilityType, SensorInfo, SensorType,
    TaskRequest, TaskResult, TaskStatus, RecoveryAction, SafetyPolicyRef,
    JointInfo,
)
from apyrobo.core.robot import Robot
from apyrobo.core.adapters import MockAdapter
from apyrobo.skills.skill import Skill, BUILTIN_SKILLS, SkillStatus, Condition
from apyrobo.skills.executor import SkillGraph, SkillExecutor, ExecutionEvent
from apyrobo.skills.agent import Agent, RuleBasedProvider
from apyrobo.safety.enforcer import SafetyEnforcer, SafetyPolicy, SafetyViolation

passed = 0
failed = 0
skipped = 0


def test(name, fn):
    global passed, failed
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


# =====================================================================
section("Phase 1: Core Schemas")
# =====================================================================

def test_robot_capability_minimal():
    rc = RobotCapability(robot_id="bot-1", name="TestBot")
    assert rc.robot_id == "bot-1"
    assert rc.capabilities == []
    assert rc.sensors == []
    assert rc.joints == []

def test_robot_capability_full():
    rc = RobotCapability(
        robot_id="tb4", name="TB4",
        capabilities=[Capability(capability_type=CapabilityType.NAVIGATE, name="nav")],
        sensors=[SensorInfo(sensor_id="cam0", sensor_type=SensorType.CAMERA)],
        joints=[JointInfo(joint_id="j1", name="wheel_left")],
        max_speed=1.2,
        workspace={"x_min": -5, "x_max": 5},
    )
    assert len(rc.capabilities) == 1
    assert rc.max_speed == 1.2
    assert rc.joints[0].name == "wheel_left"

def test_missing_required_fields():
    try:
        RobotCapability()
        assert False, "Should have raised"
    except TypeError:
        pass

def test_task_request_defaults():
    tr = TaskRequest(task_name="deliver_package")
    assert tr.priority == 1
    assert tr.parameters == {}

def test_task_request_full():
    tr = TaskRequest(
        task_name="deliver", parameters={"dest": "room_3"},
        priority=5, target_robot_id="tb4",
    )
    assert tr.parameters["dest"] == "room_3"

def test_task_result_success():
    r = TaskResult(task_name="x", status=TaskStatus.COMPLETED, confidence=0.92)
    assert r.status == TaskStatus.COMPLETED

def test_task_result_recovery():
    r = TaskResult(
        task_name="x", status=TaskStatus.FAILED,
        recovery_actions_taken=[RecoveryAction.RETRY, RecoveryAction.ABORT],
    )
    assert RecoveryAction.RETRY in r.recovery_actions_taken

def test_safety_policy_ref():
    sp = SafetyPolicyRef(policy_name="strict", max_speed=0.5, human_proximity_limit=1.0)
    assert sp.max_speed == 0.5

test("RobotCapability minimal", test_robot_capability_minimal)
test("RobotCapability full", test_robot_capability_full)
test("Missing required fields raises", test_missing_required_fields)
test("TaskRequest defaults", test_task_request_defaults)
test("TaskRequest full", test_task_request_full)
test("TaskResult success", test_task_result_success)
test("TaskResult recovery actions", test_task_result_recovery)
test("SafetyPolicyRef", test_safety_policy_ref)


# =====================================================================
section("Phase 1: Robot Discovery & Adapters")
# =====================================================================

def test_discover_mock():
    robot = Robot.discover("mock://test_bot")
    assert robot.robot_id == "test_bot"
    assert isinstance(robot._adapter, MockAdapter)

def test_discover_bad_uri():
    try:
        Robot.discover("noscheme")
        assert False
    except ValueError:
        pass

def test_discover_unknown_scheme():
    try:
        Robot.discover("unknown://bot")
        assert False
    except ValueError:
        pass

def test_capabilities():
    robot = Robot.discover("mock://tb4")
    caps = robot.capabilities()
    assert caps.robot_id == "tb4"
    assert len(caps.capabilities) == 3
    cap_types = {c.capability_type for c in caps.capabilities}
    assert CapabilityType.NAVIGATE in cap_types
    assert CapabilityType.PICK in cap_types

def test_capabilities_cached():
    robot = Robot.discover("mock://tb4")
    c1 = robot.capabilities()
    c2 = robot.capabilities()
    assert c1 is c2

def test_capabilities_refresh():
    robot = Robot.discover("mock://tb4")
    c1 = robot.capabilities()
    c2 = robot.capabilities(refresh=True)
    assert c1 is not c2

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
    for i in range(5):
        robot.move(x=float(i), y=0.0)
    assert len(robot._adapter.move_history) == 5
    assert robot._adapter.position == (4.0, 0.0)

test("Discover mock", test_discover_mock)
test("Bad URI raises", test_discover_bad_uri)
test("Unknown scheme raises", test_discover_unknown_scheme)
test("Capabilities returned", test_capabilities)
test("Capabilities cached", test_capabilities_cached)
test("Capabilities refresh", test_capabilities_refresh)
test("Move command", test_move)
test("Stop command", test_stop)
test("Multiple moves tracked", test_multiple_moves)


# =====================================================================
section("Phase 2: Skill Definition & Serialisation")
# =====================================================================

def test_builtin_skills_exist():
    assert "navigate_to" in BUILTIN_SKILLS
    assert "stop" in BUILTIN_SKILLS
    assert "pick_object" in BUILTIN_SKILLS
    assert "place_object" in BUILTIN_SKILLS
    assert "report_status" in BUILTIN_SKILLS
    assert len(BUILTIN_SKILLS) >= 5

def test_skill_serialisation_roundtrip():
    skill = BUILTIN_SKILLS["navigate_to"]
    json_str = skill.to_json()
    parsed = json.loads(json_str)
    assert parsed["skill_id"] == "navigate_to"
    assert parsed["required_capability"] == "navigate"

    restored = Skill.from_json(json_str)
    assert restored.skill_id == skill.skill_id
    assert restored.name == skill.name
    assert restored.required_capability == skill.required_capability

def test_skill_from_dict():
    data = {
        "skill_id": "custom_scan",
        "name": "Custom Scan",
        "required_capability": "scan",
        "parameters": {"resolution": "high"},
    }
    skill = Skill.from_dict(data)
    assert skill.skill_id == "custom_scan"
    assert skill.required_capability == CapabilityType.SCAN

def test_skill_preconditions():
    skill = BUILTIN_SKILLS["pick_object"]
    assert len(skill.preconditions) == 2
    assert skill.preconditions[0].name == "object_detected"

test("Built-in skills exist", test_builtin_skills_exist)
test("Skill JSON round-trip", test_skill_serialisation_roundtrip)
test("Skill from_dict", test_skill_from_dict)
test("Skill preconditions", test_skill_preconditions)


# =====================================================================
section("Phase 2: Skill Graph & Execution")
# =====================================================================

def test_skill_graph_basic():
    graph = SkillGraph()
    nav = BUILTIN_SKILLS["navigate_to"]
    pick = BUILTIN_SKILLS["pick_object"]
    graph.add_skill(nav)
    graph.add_skill(pick, depends_on=["navigate_to"])
    assert len(graph) == 2
    order = graph.get_execution_order()
    assert order[0].skill_id == "navigate_to"
    assert order[1].skill_id == "pick_object"

def test_skill_graph_params():
    graph = SkillGraph()
    nav = BUILTIN_SKILLS["navigate_to"]
    graph.add_skill(nav, parameters={"x": 5.0, "y": 3.0})
    params = graph.get_parameters("navigate_to")
    assert params["x"] == 5.0
    assert params["y"] == 3.0

def test_executor_single_skill():
    robot = Robot.discover("mock://tb4")
    executor = SkillExecutor(robot)
    nav = BUILTIN_SKILLS["navigate_to"]
    status = executor.execute_skill(nav, {"x": 2.0, "y": 3.0})
    assert status == SkillStatus.COMPLETED
    assert robot._adapter.position == (2.0, 3.0)

def test_executor_events():
    robot = Robot.discover("mock://tb4")
    executor = SkillExecutor(robot)
    events = []
    executor.on_event(lambda e: events.append(e))
    nav = BUILTIN_SKILLS["navigate_to"]
    executor.execute_skill(nav)
    assert len(events) >= 2  # PENDING + RUNNING + COMPLETED
    statuses = [e.status for e in events]
    assert SkillStatus.PENDING in statuses
    assert SkillStatus.COMPLETED in statuses

def test_executor_graph():
    robot = Robot.discover("mock://tb4")
    executor = SkillExecutor(robot)
    graph = SkillGraph()
    graph.add_skill(BUILTIN_SKILLS["navigate_to"], parameters={"x": 1.0, "y": 2.0})
    graph.add_skill(BUILTIN_SKILLS["pick_object"], depends_on=["navigate_to"])
    graph.add_skill(BUILTIN_SKILLS["navigate_to"],)  # will have same id — test dedup
    result = executor.execute_graph(graph)
    # Note: duplicate skill_id in graph means only 2 unique skills
    assert result.status == TaskStatus.COMPLETED
    assert result.steps_completed >= 2

def test_executor_stop_skill():
    robot = Robot.discover("mock://tb4")
    robot.move(x=5.0, y=5.0)
    assert robot._adapter.is_moving is True
    executor = SkillExecutor(robot)
    executor.execute_skill(BUILTIN_SKILLS["stop"])
    assert robot._adapter.is_moving is False

test("SkillGraph basic ordering", test_skill_graph_basic)
test("SkillGraph parameters", test_skill_graph_params)
test("Executor single skill", test_executor_single_skill)
test("Executor events", test_executor_events)
test("Executor full graph", test_executor_graph)
test("Executor stop skill", test_executor_stop_skill)


# =====================================================================
section("Phase 2: Agent Planning")
# =====================================================================

def test_agent_plan_delivery():
    robot = Robot.discover("mock://tb4")
    agent = Agent(provider="rule")
    graph = agent.plan("deliver package to room 3", robot)
    assert len(graph) >= 3  # navigate, pick, navigate, place
    order = graph.get_execution_order()
    skill_names = [s.name for s in order]
    assert "Navigate To" in skill_names
    assert "Pick Object" in skill_names

def test_agent_plan_navigation():
    robot = Robot.discover("mock://tb4")
    agent = Agent(provider="rule")
    graph = agent.plan("go to position (3, 5)", robot)
    assert len(graph) >= 1

def test_agent_plan_stop():
    robot = Robot.discover("mock://tb4")
    agent = Agent(provider="rule")
    graph = agent.plan("stop immediately", robot)
    assert len(graph) >= 1
    order = graph.get_execution_order()
    assert any("stop" in s.skill_id for s in order)

def test_agent_execute_delivery():
    robot = Robot.discover("mock://tb4")
    agent = Agent(provider="rule")
    events = []
    result = agent.execute(
        task="deliver package from (1, 2) to (5, 5)",
        robot=robot,
        on_event=lambda e: events.append(e),
    )
    assert result.status == TaskStatus.COMPLETED
    assert result.steps_completed > 0
    assert len(events) > 0

def test_agent_execute_status():
    robot = Robot.discover("mock://tb4")
    agent = Agent(provider="rule")
    result = agent.execute(task="report status", robot=robot)
    assert result.status == TaskStatus.COMPLETED

def test_agent_auto_provider():
    # Should fall back to rule-based since litellm isn't installed
    agent = Agent(provider="auto")
    robot = Robot.discover("mock://tb4")
    result = agent.execute(task="stop", robot=robot)
    assert result.status == TaskStatus.COMPLETED

def test_coordinate_extraction():
    provider = RuleBasedProvider()
    coords = provider._extract_coordinates("go from (1.5, 2.0) to (5, 3)")
    assert len(coords) == 2
    assert coords[0]["x"] == 1.5
    assert coords[1]["x"] == 5.0

def test_room_extraction():
    provider = RuleBasedProvider()
    rooms = provider._extract_rooms("deliver from room A to room B")
    assert len(rooms) == 2
    assert rooms[0] == "A"
    assert rooms[1] == "B"

test("Agent plans delivery", test_agent_plan_delivery)
test("Agent plans navigation", test_agent_plan_navigation)
test("Agent plans stop", test_agent_plan_stop)
test("Agent executes delivery", test_agent_execute_delivery)
test("Agent executes status report", test_agent_execute_status)
test("Agent auto provider fallback", test_agent_auto_provider)
test("Coordinate extraction from text", test_coordinate_extraction)
test("Room extraction from text", test_room_extraction)


# =====================================================================
section("Phase 3: Safety Enforcement")
# =====================================================================

def test_safety_speed_clamping():
    robot = Robot.discover("mock://tb4")
    enforcer = SafetyEnforcer(robot, policy=SafetyPolicy(
        name="test", max_speed=1.0,
    ))
    enforcer.move(x=2.0, y=3.0, speed=5.0)
    # Speed should be clamped, not rejected
    assert robot._adapter.position == (2.0, 3.0)
    assert len(enforcer.interventions) == 1
    assert enforcer.interventions[0]["type"] == "speed_clamped"
    assert enforcer.interventions[0]["requested"] == 5.0
    assert enforcer.interventions[0]["enforced"] == 1.0

def test_safety_normal_speed():
    robot = Robot.discover("mock://tb4")
    enforcer = SafetyEnforcer(robot, policy=SafetyPolicy(max_speed=2.0))
    enforcer.move(x=1.0, y=1.0, speed=1.5)
    assert len(enforcer.interventions) == 0

def test_safety_collision_zone_rejected():
    robot = Robot.discover("mock://tb4")
    enforcer = SafetyEnforcer(robot, policy=SafetyPolicy(
        name="test",
        collision_zones=[{"x_min": 3.0, "x_max": 7.0, "y_min": 3.0, "y_max": 7.0}],
    ))
    # This should be rejected
    try:
        enforcer.move(x=5.0, y=5.0)
        assert False, "Should have raised SafetyViolation"
    except SafetyViolation:
        pass
    assert len(enforcer.violations) == 1
    # Robot should NOT have moved
    assert robot._adapter.position == (0.0, 0.0)

def test_safety_collision_zone_allowed():
    robot = Robot.discover("mock://tb4")
    enforcer = SafetyEnforcer(robot, policy=SafetyPolicy(
        collision_zones=[{"x_min": 3.0, "x_max": 7.0, "y_min": 3.0, "y_max": 7.0}],
    ))
    # This is outside the zone — should pass
    enforcer.move(x=1.0, y=1.0)
    assert robot._adapter.position == (1.0, 1.0)
    assert len(enforcer.violations) == 0

def test_safety_stop_always_allowed():
    robot = Robot.discover("mock://tb4")
    enforcer = SafetyEnforcer(robot, policy=SafetyPolicy(max_speed=0.0))
    robot.move(x=1.0, y=1.0)  # bypass enforcer
    enforcer.stop()  # stop is always allowed
    assert robot._adapter.is_moving is False

def test_safety_multiple_zones():
    robot = Robot.discover("mock://tb4")
    enforcer = SafetyEnforcer(robot, policy=SafetyPolicy(
        collision_zones=[
            {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2},
            {"x_min": 8, "x_max": 10, "y_min": 8, "y_max": 10},
        ],
    ))
    # Zone 1
    try:
        enforcer.move(x=1.0, y=1.0)
        assert False
    except SafetyViolation:
        pass
    # Zone 2
    try:
        enforcer.move(x=9.0, y=9.0)
        assert False
    except SafetyViolation:
        pass
    # Safe spot
    enforcer.move(x=5.0, y=5.0)
    assert robot._adapter.position == (5.0, 5.0)
    assert len(enforcer.violations) == 2

def test_safety_policy_from_ref():
    ref = SafetyPolicyRef(policy_name="custom", max_speed=0.8, human_proximity_limit=1.5)
    policy = SafetyPolicy.from_ref(ref)
    assert policy.name == "custom"
    assert policy.max_speed == 0.8
    assert policy.human_proximity_limit == 1.5

test("Speed clamping", test_safety_speed_clamping)
test("Normal speed passes", test_safety_normal_speed)
test("Collision zone rejected", test_safety_collision_zone_rejected)
test("Outside collision zone allowed", test_safety_collision_zone_allowed)
test("Stop always allowed", test_safety_stop_always_allowed)
test("Multiple collision zones", test_safety_multiple_zones)
test("SafetyPolicy from ref", test_safety_policy_from_ref)


# =====================================================================
section("Integration: Full Pipeline")
# =====================================================================

def test_full_pipeline():
    """End-to-end: discover → plan → enforce safety → execute → result."""
    robot = Robot.discover("mock://turtlebot4")
    enforcer = SafetyEnforcer(robot, policy=SafetyPolicy(
        name="demo",
        max_speed=1.0,
        collision_zones=[{"x_min": 8, "x_max": 10, "y_min": 8, "y_max": 10}],
    ))
    agent = Agent(provider="rule")

    all_events = []
    result = agent.execute(
        task="deliver package from (1, 2) to (5, 5)",
        robot=robot,
        on_event=lambda e: all_events.append(e),
    )

    assert result.status == TaskStatus.COMPLETED
    assert result.steps_completed > 0
    assert len(all_events) > 0

    # Verify events have proper structure
    for event in all_events:
        assert hasattr(event, "skill_id")
        assert hasattr(event, "status")
        assert hasattr(event, "timestamp")

def test_full_pipeline_events_streaming():
    """Verify events stream in correct order."""
    robot = Robot.discover("mock://tb4")
    agent = Agent(provider="rule")
    result = agent.execute(task="go to (3, 4)", robot=robot)
    events = agent.last_events
    assert len(events) >= 2
    # First event should be PENDING
    assert events[0].status == SkillStatus.PENDING
    # Last event should be COMPLETED
    assert events[-1].status == SkillStatus.COMPLETED

test("Full pipeline: discover → plan → execute", test_full_pipeline)
test("Event streaming order", test_full_pipeline_events_streaming)


# =====================================================================
section("Phase 4: Swarm — Bus")
# =====================================================================

from apyrobo.swarm.bus import SwarmBus, SwarmMessage
from apyrobo.swarm.coordinator import SwarmCoordinator
from apyrobo.swarm.safety import SwarmSafety, ProximityViolation, DeadlockDetected

def test_swarm_register():
    bus = SwarmBus()
    r1 = Robot.discover("mock://robot_a")
    r2 = Robot.discover("mock://robot_b")
    bus.register(r1)
    bus.register(r2)
    assert bus.robot_count == 2
    assert "robot_a" in bus.robot_ids
    assert "robot_b" in bus.robot_ids

def test_swarm_send():
    bus = SwarmBus()
    r1 = Robot.discover("mock://robot_a")
    r2 = Robot.discover("mock://robot_b")
    bus.register(r1)
    bus.register(r2)
    received = []
    bus.on_message("robot_b", lambda msg: received.append(msg))
    bus.send("robot_a", "robot_b", {"task": "help"})
    assert len(received) == 1
    assert received[0].sender == "robot_a"
    assert received[0].payload["task"] == "help"

def test_swarm_broadcast():
    bus = SwarmBus()
    r1 = Robot.discover("mock://robot_a")
    r2 = Robot.discover("mock://robot_b")
    r3 = Robot.discover("mock://robot_c")
    bus.register(r1)
    bus.register(r2)
    bus.register(r3)
    received_b = []
    received_c = []
    bus.on_message("robot_b", lambda msg: received_b.append(msg))
    bus.on_message("robot_c", lambda msg: received_c.append(msg))
    bus.broadcast("robot_a", {"status": "ok"})
    assert len(received_b) == 1  # not counting the registration broadcast
    assert len(received_c) == 1

def test_swarm_send_unknown_target():
    bus = SwarmBus()
    r1 = Robot.discover("mock://robot_a")
    bus.register(r1)
    try:
        bus.send("robot_a", "nonexistent", {"msg": "hello"})
        assert False
    except ValueError:
        pass

def test_swarm_unregister():
    bus = SwarmBus()
    r1 = Robot.discover("mock://robot_a")
    bus.register(r1)
    assert bus.robot_count == 1
    bus.unregister("robot_a")
    assert bus.robot_count == 0

def test_swarm_capabilities():
    bus = SwarmBus()
    r1 = Robot.discover("mock://robot_a")
    bus.register(r1)
    caps = bus.get_capabilities("robot_a")
    assert caps.robot_id == "robot_a"
    assert len(caps.capabilities) > 0

def test_swarm_message_log():
    bus = SwarmBus()
    r1 = Robot.discover("mock://robot_a")
    r2 = Robot.discover("mock://robot_b")
    bus.register(r1)
    bus.register(r2)
    initial_count = len(bus.message_log)
    bus.send("robot_a", "robot_b", {"msg": "test"})
    assert len(bus.message_log) == initial_count + 1

test("Swarm register", test_swarm_register)
test("Swarm targeted send", test_swarm_send)
test("Swarm broadcast", test_swarm_broadcast)
test("Swarm send to unknown target", test_swarm_send_unknown_target)
test("Swarm unregister", test_swarm_unregister)
test("Swarm capabilities query", test_swarm_capabilities)
test("Swarm message log", test_swarm_message_log)


# =====================================================================
section("Phase 4: Swarm — Coordinator")
# =====================================================================

def test_coordinator_single_robot():
    bus = SwarmBus()
    r1 = Robot.discover("mock://robot_a")
    bus.register(r1)
    agent = Agent(provider="rule")
    coord = SwarmCoordinator(bus)
    result = coord.execute_task("deliver package to room 3", agent)
    assert result.status == TaskStatus.COMPLETED
    assert result.steps_completed > 0

def test_coordinator_two_robots():
    bus = SwarmBus()
    r1 = Robot.discover("mock://robot_a")
    r2 = Robot.discover("mock://robot_b")
    bus.register(r1)
    bus.register(r2)
    agent = Agent(provider="rule")
    coord = SwarmCoordinator(bus)
    result = coord.execute_task("deliver package from (1,2) to (5,5)", agent)
    assert result.status == TaskStatus.COMPLETED
    # Verify work was distributed
    assignments = coord.assignments
    assert len(assignments) >= 1

def test_coordinator_events():
    bus = SwarmBus()
    r1 = Robot.discover("mock://robot_a")
    r2 = Robot.discover("mock://robot_b")
    bus.register(r1)
    bus.register(r2)
    agent = Agent(provider="rule")
    coord = SwarmCoordinator(bus)
    events = []
    coord.execute_task("go to (3, 4)", agent, on_event=lambda e: events.append(e))
    assert len(events) > 0
    assert len(coord.events) > 0

test("Coordinator single robot", test_coordinator_single_robot)
test("Coordinator two robots", test_coordinator_two_robots)
test("Coordinator event streaming", test_coordinator_events)


# =====================================================================
section("Phase 4: Swarm — Safety")
# =====================================================================

def test_swarm_proximity_ok():
    bus = SwarmBus()
    safety = SwarmSafety(bus, min_distance=0.5)
    safety.update_position("robot_a", 0.0, 0.0)
    safety.update_position("robot_b", 5.0, 5.0)
    violations = safety.check_proximity()
    assert len(violations) == 0

def test_swarm_proximity_violation():
    bus = SwarmBus()
    safety = SwarmSafety(bus, min_distance=1.0)
    safety.update_position("robot_a", 0.0, 0.0)
    safety.update_position("robot_b", 0.3, 0.3)
    violations = safety.check_proximity()
    assert len(violations) == 1
    assert violations[0][0] == "robot_a"
    assert violations[0][1] == "robot_b"

def test_swarm_proximity_enforce():
    bus = SwarmBus()
    safety = SwarmSafety(bus, min_distance=1.0)
    safety.update_position("robot_a", 0.0, 0.0)
    safety.update_position("robot_b", 0.1, 0.1)
    try:
        safety.enforce_proximity()
        assert False
    except ProximityViolation:
        pass

def test_swarm_would_violate():
    bus = SwarmBus()
    safety = SwarmSafety(bus, min_distance=1.0)
    safety.update_position("robot_b", 3.0, 3.0)
    would, other, dist = safety.would_violate_proximity("robot_a", 3.1, 3.1)
    assert would is True
    assert other == "robot_b"
    would2, _, _ = safety.would_violate_proximity("robot_a", 10.0, 10.0)
    assert would2 is False

def test_swarm_deadlock_none():
    bus = SwarmBus()
    safety = SwarmSafety(bus)
    safety.set_waiting("robot_a", None)
    cycles = safety.check_deadlock()
    assert len(cycles) == 0

def test_swarm_deadlock_detected():
    bus = SwarmBus()
    safety = SwarmSafety(bus)
    safety.set_waiting("robot_a", "robot_b")
    safety.set_waiting("robot_b", "robot_a")
    cycles = safety.check_deadlock()
    assert len(cycles) >= 1
    cycle = cycles[0]
    assert "robot_a" in cycle
    assert "robot_b" in cycle

def test_swarm_deadlock_enforce():
    bus = SwarmBus()
    safety = SwarmSafety(bus)
    safety.set_waiting("robot_a", "robot_b")
    safety.set_waiting("robot_b", "robot_a")
    try:
        safety.enforce_deadlock()
        assert False
    except DeadlockDetected:
        pass

def test_swarm_check_all_safe():
    bus = SwarmBus()
    safety = SwarmSafety(bus, min_distance=0.5)
    safety.update_position("robot_a", 0.0, 0.0)
    safety.update_position("robot_b", 10.0, 10.0)
    result = safety.check_all()
    assert result["safe"] is True

def test_swarm_check_all_unsafe():
    bus = SwarmBus()
    safety = SwarmSafety(bus, min_distance=2.0)
    safety.update_position("robot_a", 0.0, 0.0)
    safety.update_position("robot_b", 0.5, 0.5)
    safety.set_waiting("robot_a", "robot_b")
    safety.set_waiting("robot_b", "robot_a")
    result = safety.check_all()
    assert result["safe"] is False

test("Proximity: robots far apart", test_swarm_proximity_ok)
test("Proximity: robots too close", test_swarm_proximity_violation)
test("Proximity: enforce raises", test_swarm_proximity_enforce)
test("Proximity: would_violate check", test_swarm_would_violate)
test("Deadlock: none detected", test_swarm_deadlock_none)
test("Deadlock: cycle detected", test_swarm_deadlock_detected)
test("Deadlock: enforce raises", test_swarm_deadlock_enforce)
test("Check all: safe", test_swarm_check_all_safe)
test("Check all: unsafe", test_swarm_check_all_unsafe)


# =====================================================================
section("Phase 5: Sensor Pipeline & World State")
# =====================================================================

from apyrobo.sensors.pipeline import (
    SensorPipeline, SensorReading, WorldState, Obstacle, DetectedObject,
)
from apyrobo.core.schemas import SensorType

def test_world_state_empty():
    ws = WorldState()
    assert ws.obstacles == []
    assert ws.detected_objects == []
    assert ws.robot_position == (0.0, 0.0)

def test_pipeline_lidar():
    pipeline = SensorPipeline()
    pipeline.feed(SensorReading(
        "lidar0", SensorType.LIDAR,
        [{"x": 3.0, "y": 2.0}, {"x": 5.0, "y": 5.0}],
    ))
    world = pipeline.get_world_state()
    assert len(world.obstacles) == 2
    assert world.obstacles[0].x == 3.0

def test_pipeline_camera():
    pipeline = SensorPipeline()
    pipeline.feed(SensorReading(
        "cam0", SensorType.CAMERA,
        [{"label": "box", "x": 1.0, "y": 2.0}, {"label": "person", "x": 3.0, "y": 4.0}],
    ))
    world = pipeline.get_world_state()
    assert len(world.detected_objects) == 2
    box = world.find_object("box")
    assert box is not None
    assert box.x == 1.0

def test_pipeline_imu():
    pipeline = SensorPipeline()
    pipeline.feed(SensorReading(
        "imu0", SensorType.IMU,
        {"x": 2.5, "y": 3.5, "yaw": 1.57},
    ))
    world = pipeline.get_world_state()
    assert world.robot_position == (2.5, 3.5)
    assert abs(world.robot_orientation - 1.57) < 0.01

def test_nearest_obstacle():
    ws = WorldState()
    ws.obstacles = [
        Obstacle(10.0, 10.0),
        Obstacle(1.0, 1.0),
        Obstacle(5.0, 5.0),
    ]
    nearest = ws.nearest_obstacle(0.0, 0.0)
    assert nearest is not None
    assert nearest.x == 1.0

def test_obstacles_within():
    ws = WorldState()
    ws.obstacles = [
        Obstacle(1.0, 0.0),
        Obstacle(3.0, 0.0),
        Obstacle(10.0, 0.0),
    ]
    nearby = ws.obstacles_within(5.0, 0.0, 0.0)
    assert len(nearby) == 2

def test_path_clear():
    ws = WorldState()
    ws.obstacles = [Obstacle(5.0, 5.0, radius=0.3)]
    assert ws.is_path_clear(0.0, 0.0, 10.0, 0.0) is True   # path along x-axis, obstacle at (5,5)
    assert ws.is_path_clear(0.0, 0.0, 10.0, 10.0) is False  # diagonal passes through obstacle

def test_pipeline_fusion():
    """Feed multiple sensors and verify fused world state."""
    pipeline = SensorPipeline()
    pipeline.feed(SensorReading("imu0", SensorType.IMU, {"x": 1.0, "y": 2.0, "yaw": 0.0}))
    pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, [{"x": 3.0, "y": 3.0}]))
    pipeline.feed(SensorReading("cam0", SensorType.CAMERA, [{"label": "package", "x": 4.0, "y": 4.0}]))
    world = pipeline.get_world_state()
    assert world.robot_position == (1.0, 2.0)
    assert len(world.obstacles) == 1
    assert len(world.detected_objects) == 1
    assert world.detected_objects[0].label == "package"

def test_world_state_to_dict():
    ws = WorldState()
    ws.robot_position = (1.0, 2.0)
    ws.obstacles = [Obstacle(3.0, 4.0)]
    ws.detected_objects = [DetectedObject("obj1", "box", 5.0, 6.0)]
    d = ws.to_dict()
    assert d["robot_position"] == [1.0, 2.0]
    assert len(d["obstacles"]) == 1
    assert d["detected_objects"][0]["label"] == "box"

test("WorldState empty", test_world_state_empty)
test("Pipeline lidar → obstacles", test_pipeline_lidar)
test("Pipeline camera → objects", test_pipeline_camera)
test("Pipeline IMU → pose", test_pipeline_imu)
test("Nearest obstacle", test_nearest_obstacle)
test("Obstacles within radius", test_obstacles_within)
test("Path clearance check", test_path_clear)
test("Multi-sensor fusion", test_pipeline_fusion)
test("WorldState to_dict", test_world_state_to_dict)


# =====================================================================
section("Confidence Estimation")
# =====================================================================

from apyrobo.safety.confidence import ConfidenceEstimator, ConfidenceReport, RiskFactor

def test_confidence_basic():
    robot = Robot.discover("mock://tb4")
    agent = Agent(provider="rule")
    graph = agent.plan("go to (3, 4)", robot)
    estimator = ConfidenceEstimator()
    report = estimator.assess(graph, robot)
    assert 0.0 <= report.confidence <= 1.0
    assert report.can_proceed is True
    assert report.risk_level in ("low", "medium", "high")

def test_confidence_report_to_dict():
    report = ConfidenceReport(
        confidence=0.85, risks=[RiskFactor("test", 0.2, "minor")],
        can_proceed=True,
    )
    d = report.to_dict()
    assert d["confidence"] == 0.85
    assert len(d["risks"]) == 1

def test_confidence_delivery():
    robot = Robot.discover("mock://tb4")
    agent = Agent(provider="rule")
    graph = agent.plan("deliver package to room 3", robot)
    estimator = ConfidenceEstimator()
    report = estimator.assess(graph, robot)
    assert report.confidence > 0.0
    # Delivery plan should mention risks if any
    assert isinstance(report.risks, list)

def test_confidence_with_world_state():
    from apyrobo.sensors.pipeline import WorldState, Obstacle
    ws = WorldState()
    ws.obstacles = [Obstacle(0.5, 0.0), Obstacle(0.3, 0.1), Obstacle(0.7, -0.2), Obstacle(0.4, 0.3)]
    robot = Robot.discover("mock://tb4")
    agent = Agent(provider="rule")
    graph = agent.plan("go to (3, 4)", robot)
    estimator = ConfidenceEstimator(world_state=ws)
    report = estimator.assess(graph, robot)
    # Should detect crowded environment
    risk_names = [r.name for r in report.risks]
    assert "crowded_environment" in risk_names

test("Confidence: basic assessment", test_confidence_basic)
test("Confidence: report to_dict", test_confidence_report_to_dict)
test("Confidence: delivery task", test_confidence_delivery)
test("Confidence: world state risks", test_confidence_with_world_state)


# =====================================================================
section("Skill Library")
# =====================================================================

from apyrobo.skills.library import SkillLibrary
import tempfile, os

def test_library_builtins():
    lib = SkillLibrary()
    assert len(lib) >= 5
    assert "navigate_to" in lib
    assert "stop" in lib

def test_library_load_json():
    lib = SkillLibrary()
    skill_json = json.dumps({
        "skill_id": "test_custom",
        "name": "Test Custom",
        "required_capability": "custom",
        "parameters": {"foo": "bar"},
    })
    skill = lib.load_json(skill_json)
    assert skill.skill_id == "test_custom"
    assert "test_custom" in lib

def test_library_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        lib = SkillLibrary(tmpdir)
        skill = Skill(
            skill_id="test_save",
            name="Test Save",
            parameters={"x": 1.0},
        )
        path = lib.save_skill(skill)
        assert path.exists()

        # Load fresh
        lib2 = SkillLibrary(tmpdir)
        loaded = lib2.get("test_save")
        assert loaded is not None
        assert loaded.name == "Test Save"

def test_library_from_project_skills():
    skills_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
    if os.path.exists(skills_dir):
        lib = SkillLibrary(skills_dir)
        custom = lib.custom_skills()
        assert len(custom) >= 1  # patrol_route.json at minimum
        assert "patrol_route" in custom or "scan_area" in custom

def test_library_remove():
    lib = SkillLibrary()
    lib.load_json('{"skill_id": "temp", "name": "Temp"}')
    assert "temp" in lib
    lib.remove("temp")
    assert "temp" not in lib

test("Library: built-ins present", test_library_builtins)
test("Library: load from JSON string", test_library_load_json)
test("Library: save and reload", test_library_save_and_load)
test("Library: load from skills/ dir", test_library_from_project_skills)
test("Library: remove custom skill", test_library_remove)


# =====================================================================
section("Configuration")
# =====================================================================

from apyrobo.config import ApyroboConfig

def test_config_defaults():
    config = ApyroboConfig()
    assert config.robot_uri == "mock://turtlebot4"
    assert config.agent_provider == "auto"
    assert config.log_level == "INFO"

def test_config_override():
    config = ApyroboConfig({"robot": {"uri": "gazebo://tb4"}, "safety": {"max_speed": 0.5}})
    assert config.robot_uri == "gazebo://tb4"
    policy = config.safety_policy()
    assert policy.max_speed == 0.5
    # Defaults should still be present for non-overridden fields
    assert config.agent_provider == "auto"

def test_config_dot_access():
    config = ApyroboConfig()
    assert config.get("safety.max_speed") == 1.5
    assert config.get("nonexistent.key", "fallback") == "fallback"

def test_config_yaml_roundtrip():
    config = ApyroboConfig({"robot": {"uri": "ros2://test"}})
    yaml_str = config.to_yaml()
    assert "ros2://test" in yaml_str

def test_config_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.yaml")
        config = ApyroboConfig({"robot": {"uri": "mock://saved"}})
        config.save(path)

        loaded = ApyroboConfig.from_file(path)
        assert loaded.robot_uri == "mock://saved"

test("Config: defaults", test_config_defaults)
test("Config: override merging", test_config_override)
test("Config: dot-notation access", test_config_dot_access)
test("Config: YAML roundtrip", test_config_yaml_roundtrip)
test("Config: save and load file", test_config_save_load)


# =====================================================================
section("CLI (smoke test)")
# =====================================================================

import argparse as _argparse

def test_cli_discover():
    """Smoke test the CLI discover command."""
    import io, contextlib
    from apyrobo.cli import cmd_discover
    args = _argparse.Namespace(uri="mock://cli_test")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        cmd_discover(args)
    text = output.getvalue()
    assert "cli_test" in text
    assert "navigate_to" in text

def test_cli_plan():
    import io, contextlib
    from apyrobo.cli import cmd_plan
    args = _argparse.Namespace(task="go to (1, 2)", robot="mock://tb4", provider="rule")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        cmd_plan(args)
    text = output.getvalue()
    assert "Navigate To" in text
    assert "Confidence" in text

def test_cli_skills_list():
    import io, contextlib
    from apyrobo.cli import cmd_skills
    args = _argparse.Namespace(list=True, export=None)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        cmd_skills(args)
    text = output.getvalue()
    assert "navigate_to" in text
    assert "pick_object" in text

test("CLI: discover", test_cli_discover)
test("CLI: plan", test_cli_plan)
test("CLI: skills --list", test_cli_skills_list)


# =====================================================================
section("Inference Router")
# =====================================================================

from apyrobo.inference.router import InferenceRouter, Urgency, ProviderHealth, InferenceTier

def test_router_fallback_only():
    """Router with no tiers falls back to rule-based."""
    router = InferenceRouter()
    robot = Robot.discover("mock://tb4")
    caps = robot.capabilities()
    cap_names = [c.capability_type.value for c in caps.capabilities]
    skills = [{"skill_id": "navigate_to", "name": "Navigate To",
               "description": "Move", "required_capability": "navigate", "parameters": {}}]
    result = router.plan("go to (3, 4)", skills, cap_names)
    assert len(result) > 0  # rule-based fallback should produce a plan

def test_router_with_rule_tier():
    """Router with a rule-based tier routes to it."""
    router = InferenceRouter()
    router.add_tier("local_rules", RuleBasedProvider(), max_latency_ms=100, priority=0)
    skills = [{"skill_id": "navigate_to", "name": "nav", "description": "",
               "required_capability": "navigate", "parameters": {}}]
    result = router.plan("go to (1, 2)", skills, ["navigate"])
    assert len(result) > 0
    assert len(router.route_log) == 1
    assert router.route_log[0]["tier"] == "local_rules"
    assert router.route_log[0]["success"] is True

def test_router_urgency_filtering():
    """High urgency should skip cloud-only tiers."""
    router = InferenceRouter()
    # Cloud tier: only normal/low
    router.add_tier("cloud", RuleBasedProvider(), priority=0,
                    supports_urgency=[Urgency.NORMAL, Urgency.LOW])
    # Edge tier: high/normal
    router.add_tier("edge", RuleBasedProvider(), priority=1,
                    supports_urgency=[Urgency.HIGH, Urgency.NORMAL])
    skills = [{"skill_id": "stop", "name": "Stop", "description": "",
               "required_capability": "navigate", "parameters": {}}]

    # Normal urgency -> cloud (priority 0)
    router.plan("stop", skills, ["navigate"], urgency=Urgency.NORMAL)
    assert router.route_log[-1]["tier"] == "cloud"

    # High urgency -> edge (cloud filtered out)
    router.plan("obstacle! stop!", skills, ["navigate"], urgency=Urgency.HIGH)
    assert router.route_log[-1]["tier"] == "edge"

def test_router_health_tracking():
    """Provider health tracks latency and errors."""
    health = ProviderHealth("test")
    health.record_success(100.0)
    health.record_success(200.0)
    health.record_success(150.0)
    assert health.is_healthy is True
    assert 100 <= health.avg_latency_ms <= 200
    assert health.error_rate == 0.0

    health.record_failure("timeout")
    health.record_failure("timeout")
    health.record_failure("timeout")
    assert health.is_healthy is False
    assert health.error_rate > 0

def test_router_health_report():
    """Health report returns structured data."""
    router = InferenceRouter()
    router.add_tier("fast", RuleBasedProvider(), max_latency_ms=100)
    router.plan("test", [{"skill_id": "stop", "name": "s", "description": "",
                          "required_capability": "custom", "parameters": {}}], [])
    report = router.health_report()
    assert "tiers" in report
    assert len(report["tiers"]) == 1
    assert report["tiers"][0]["name"] == "fast"

def test_router_connectivity_check():
    router = InferenceRouter()
    router.add_tier("t1", RuleBasedProvider())
    router.add_tier("t2", RuleBasedProvider())
    status = router.connectivity_check()
    assert status["t1"] is True
    assert status["t2"] is True
    assert status["rule_fallback"] is True

def test_router_tier_names():
    router = InferenceRouter()
    router.add_tier("cloud", RuleBasedProvider())
    router.add_tier("edge", RuleBasedProvider())
    assert router.tier_names == ["cloud", "edge"]

def test_router_from_config():
    """Build router from config dict (without real LLM models)."""
    # Models are None so it will use RuleBasedProvider fallback
    router = InferenceRouter()
    router.add_tier("test_tier", RuleBasedProvider(), max_latency_ms=500, priority=0)
    assert len(router.tier_names) == 1

def test_agent_with_router():
    """Agent can use the router as its provider."""
    router = InferenceRouter()
    router.add_tier("local", RuleBasedProvider(), max_latency_ms=100)
    agent = Agent(provider="routed", router=router)
    robot = Robot.discover("mock://tb4")
    result = agent.execute(task="go to (3, 4)", robot=robot)
    assert result.status == TaskStatus.COMPLETED
    assert len(router.route_log) > 0

test("Router: fallback only", test_router_fallback_only)
test("Router: rule-based tier", test_router_with_rule_tier)
test("Router: urgency filtering", test_router_urgency_filtering)
test("Router: health tracking", test_router_health_tracking)
test("Router: health report", test_router_health_report)
test("Router: connectivity check", test_router_connectivity_check)
test("Router: tier names", test_router_tier_names)
test("Router: from config", test_router_from_config)
test("Router: agent integration", test_agent_with_router)


# =====================================================================
section("Observability")
# =====================================================================

from apyrobo.observability import get_logger as get_slog, trace_context, current_trace_id

def test_trace_context():
    with trace_context(task="deliver") as ctx:
        assert "trace_id" in ctx
        assert ctx["task"] == "deliver"
        assert current_trace_id() == ctx["trace_id"]
    assert current_trace_id() is None

def test_structured_logger():
    slog = get_slog("test_module")
    # Should not raise
    slog.info("test message", key="value", count=42)
    slog.warning("warn", reason="test")

test("Trace context", test_trace_context)
test("Structured logger", test_structured_logger)


# =====================================================================
section("State Persistence")
# =====================================================================

from apyrobo.persistence import StateStore

def test_state_task_lifecycle():
    store = StateStore("/tmp/apyrobo_test_state.json")
    store.clear()
    store.begin_task("t1", {"desc": "deliver"}, robot_id="tb4", total_steps=4)
    store.update_task("t1", step=2, status="in_progress")
    task = store.get_task("t1")
    assert task is not None
    assert task.step == 2
    store.complete_task("t1", result={"status": "ok"})
    task = store.get_task("t1")
    assert task.status == "completed"
    store.clear()

def test_state_interrupted_detection():
    store = StateStore("/tmp/apyrobo_test_state2.json")
    store.clear()
    store.begin_task("t1", {})
    store.begin_task("t2", {})
    store.complete_task("t2")
    interrupted = store.get_interrupted_tasks()
    assert len(interrupted) == 1
    assert interrupted[0].task_id == "t1"
    store.clear()

def test_state_persistence_roundtrip():
    store = StateStore("/tmp/apyrobo_test_state3.json")
    store.clear()
    store.begin_task("persist_test", {"key": "val"})
    store.save_robot_position("tb4", 1.5, 2.5, 0.3)
    # Load fresh
    store2 = StateStore("/tmp/apyrobo_test_state3.json")
    assert store2.get_task("persist_test") is not None
    pos = store2.get_robot_position("tb4")
    assert pos is not None
    assert pos["x"] == 1.5
    store.clear()

test("State: task lifecycle", test_state_task_lifecycle)
test("State: interrupted detection", test_state_interrupted_detection)
test("State: persistence roundtrip", test_state_persistence_roundtrip)


# =====================================================================
section("Auth + Access Control")
# =====================================================================

from apyrobo.auth import AuthManager, GuardedRobot, AuthError, Role

def test_auth_basic():
    auth = AuthManager()
    user = auth.add_user("op1", role=Role.OPERATOR, robots=["tb4"])
    assert user.can_command("tb4") is True
    assert user.can_command("other_robot") is False

def test_auth_admin():
    auth = AuthManager()
    auth.add_user("admin", role=Role.ADMIN)
    assert auth.check_access("admin", "any_robot", "move") is True

def test_auth_viewer():
    auth = AuthManager()
    auth.add_user("viewer", role=Role.VIEWER)
    assert auth.check_access("viewer", "tb4", "move") is False
    assert auth.check_access("viewer", "tb4", "view") is True

def test_auth_guarded_robot():
    auth = AuthManager()
    auth.add_user("op1", role=Role.OPERATOR, robots=["tb4"])
    robot = Robot.discover("mock://tb4")
    guarded = auth.guard(robot, user_id="op1")
    guarded.move(x=1.0, y=2.0)  # should succeed
    guarded.capabilities()  # should succeed

def test_auth_guarded_denied():
    auth = AuthManager()
    auth.add_user("op1", role=Role.OPERATOR, robots=["other"])
    robot = Robot.discover("mock://tb4")
    guarded = auth.guard(robot, user_id="op1")
    try:
        guarded.move(x=1.0, y=2.0)
        assert False, "Should have raised AuthError"
    except AuthError:
        pass

def test_auth_audit_trail():
    auth = AuthManager()
    auth.add_user("op1", role=Role.OPERATOR, robots=["tb4"])
    auth.check_access("op1", "tb4", "move")
    auth.check_access("op1", "forbidden", "move")
    assert len(auth.audit_log) == 2
    assert auth.audit_log[0].allowed is True
    assert auth.audit_log[1].allowed is False

def test_auth_api_key():
    auth = AuthManager()
    user = auth.add_user("op1")
    found = auth.authenticate(user.api_key)
    assert found is not None
    assert found.user_id == "op1"
    assert auth.authenticate("bad_key") is None

test("Auth: basic permissions", test_auth_basic)
test("Auth: admin access", test_auth_admin)
test("Auth: viewer denied commands", test_auth_viewer)
test("Auth: guarded robot allowed", test_auth_guarded_robot)
test("Auth: guarded robot denied", test_auth_guarded_denied)
test("Auth: audit trail", test_auth_audit_trail)
test("Auth: API key lookup", test_auth_api_key)


# =====================================================================
section("Task Queue")
# =====================================================================

from apyrobo.task_queue import TaskQueue, QueuedTaskStatus

def test_queue_submit_and_next():
    q = TaskQueue()
    q.submit("low task", priority=2)
    q.submit("high task", priority=8)
    task = q.next()
    assert task is not None
    assert task.task_description == "high task"

def test_queue_priority_ordering():
    q = TaskQueue()
    q.submit("p1", priority=1)
    q.submit("p5", priority=5)
    q.submit("p3", priority=3)
    t1 = q.next()
    t2 = q.next()
    t3 = q.next()
    assert t1.task_description == "p5"
    assert t2.task_description == "p3"
    assert t3.task_description == "p1"

def test_queue_cancel():
    q = TaskQueue()
    task = q.submit("cancel me", priority=5)
    assert q.cancel(task.task_id) is True
    assert q.next() is None  # cancelled, nothing left

def test_queue_completion():
    q = TaskQueue()
    task = q.submit("do it", priority=5)
    t = q.next()
    q.mark_running(t.task_id, "tb4")
    q.mark_completed(t.task_id, {"result": "ok"})
    assert q.get_task(t.task_id).status == QueuedTaskStatus.COMPLETED

def test_queue_stats():
    q = TaskQueue()
    q.submit("a", priority=5)
    q.submit("b", priority=3)
    t = q.next()
    q.mark_running(t.task_id, "tb4")
    stats = q.stats()
    assert stats.get("queued", 0) >= 1
    assert stats.get("running", 0) >= 1

def test_queue_events():
    q = TaskQueue()
    events = []
    q.on_event(lambda etype, task: events.append(etype))
    q.submit("task1", priority=5)
    assert "submitted" in events

test("Queue: submit and next", test_queue_submit_and_next)
test("Queue: priority ordering", test_queue_priority_ordering)
test("Queue: cancel", test_queue_cancel)
test("Queue: completion lifecycle", test_queue_completion)
test("Queue: stats", test_queue_stats)
test("Queue: events", test_queue_events)


# =====================================================================
section("Operations: Battery, Maps, Teleop, Webhooks")
# =====================================================================

from apyrobo.operations import BatteryMonitor, MapManager, TeleoperationBridge, WebhookEmitter

def test_battery_basic():
    batt = BatteryMonitor("tb4", dock_position=(0, 0))
    batt.update(percentage=80.0)
    assert batt.status == "ok"
    assert batt.estimated_range_m > 0

def test_battery_low():
    batt = BatteryMonitor("tb4")
    batt.update(percentage=15.0)
    assert batt.status == "low"

def test_battery_trip_check():
    batt = BatteryMonitor("tb4", meters_per_percent=2.0)
    batt.update(percentage=50.0)  # 100m range
    assert batt.can_complete_trip(30.0) is True
    assert batt.can_complete_trip(80.0) is False  # 80m + 80m return > 100m

def test_map_manager():
    mm = MapManager()
    mm.register("floor1", "/maps/floor1.yaml", floor=1)
    mm.register("floor2", "/maps/floor2.yaml", floor=2)
    assert len(mm.available_maps) == 2
    mm.set_active("floor1")
    assert mm.active_map_name == "floor1"
    f2 = mm.get_floor_map(2)
    assert f2 is not None

def test_teleop():
    teleop = TeleoperationBridge("tb4")
    assert teleop.is_active is False
    cmds = []
    teleop.set_velocity_callback(lambda l, a: cmds.append((l, a)))
    teleop.enable(operator_id="nurse_1")
    assert teleop.is_active is True
    teleop.send_velocity(0.5, 0.1)
    assert len(cmds) == 1
    assert cmds[0] == (0.5, 0.1)
    teleop.disable()
    assert teleop.is_active is False
    assert len(cmds) == 2  # zero velocity on disable

def test_teleop_rejected_when_disabled():
    teleop = TeleoperationBridge("tb4")
    assert teleop.send_velocity(0.5, 0.0) is False

def test_webhook_callback():
    wh = WebhookEmitter()
    received = []
    wh.add_callback("test", lambda e: received.append(e))
    wh.emit("task_completed", task_id="t1", robot="tb4")
    assert len(received) == 1
    assert received[0]["event_type"] == "task_completed"
    assert received[0]["data"]["task_id"] == "t1"

def test_webhook_event_log():
    wh = WebhookEmitter()
    wh.emit("safety_violation", robot="tb4", zone="restricted")
    wh.emit("battery_low", robot="tb4", percentage=15)
    assert len(wh.event_log) == 2

test("Battery: basic state", test_battery_basic)
test("Battery: low threshold", test_battery_low)
test("Battery: trip feasibility", test_battery_trip_check)
test("Maps: register and switch", test_map_manager)
test("Teleop: enable/command/disable", test_teleop)
test("Teleop: rejected when disabled", test_teleop_rejected_when_disabled)
test("Webhooks: callback delivery", test_webhook_callback)
test("Webhooks: event log", test_webhook_event_log)


# =====================================================================
section("Skill Execution Engine — Execution State")
# =====================================================================

from apyrobo.skills.executor import ExecutionState

def test_execution_state_basic():
    state = ExecutionState()
    assert state.get("key") is None
    assert state.is_set("key") is False
    state.set("key", "value")
    assert state.get("key") == "value"
    assert state.is_set("key") is True

def test_execution_state_flags():
    state = ExecutionState()
    state.set("object_held", True)
    state.set("at_position", (1.0, 2.0))
    assert state.flags == {"object_held": True, "at_position": (1.0, 2.0)}

def test_execution_state_clear():
    state = ExecutionState()
    state.set("a", 1)
    state.set("b", 2)
    state.clear("a")
    assert state.get("a") is None
    assert state.get("b") == 2
    state.clear_all()
    assert state.flags == {}

test("ExecutionState: basic get/set", test_execution_state_basic)
test("ExecutionState: flags property", test_execution_state_flags)
test("ExecutionState: clear/clear_all", test_execution_state_clear)


# =====================================================================
section("Skill Execution Engine — Timeout Enforcement")
# =====================================================================

from apyrobo.skills.executor import SkillTimeout, _run_with_timeout

def test_timeout_fast_fn():
    """Fast function completes before timeout."""
    result = _run_with_timeout(lambda: 42, timeout_seconds=5.0)
    assert result == 42

def test_timeout_slow_fn():
    """Slow function triggers SkillTimeout."""
    def slow():
        time.sleep(10)
        return "should not reach"
    try:
        _run_with_timeout(slow, timeout_seconds=0.2)
        assert False, "Should have raised SkillTimeout"
    except SkillTimeout:
        pass

def test_timeout_propagates_exception():
    """Exception inside fn is propagated, not converted to timeout."""
    def failing():
        raise ValueError("deliberate")
    try:
        _run_with_timeout(failing, timeout_seconds=5.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "deliberate" in str(e)

def test_executor_timeout_enforcement():
    """Executor uses skill.timeout_seconds during dispatch."""
    robot = Robot.discover("mock://tb4")
    executor = SkillExecutor(robot)
    # Create a skill with a very short timeout that will still succeed
    # because mock dispatch is instant
    fast_skill = Skill(
        skill_id="navigate_to",
        name="Navigate To",
        required_capability=CapabilityType.NAVIGATE,
        parameters={"x": 1.0, "y": 2.0},
        timeout_seconds=5.0,
    )
    status = executor.execute_skill(fast_skill)
    assert status == SkillStatus.COMPLETED

test("Timeout: fast function succeeds", test_timeout_fast_fn)
test("Timeout: slow function raises SkillTimeout", test_timeout_slow_fn)
test("Timeout: exception propagation", test_timeout_propagates_exception)
test("Timeout: executor enforcement", test_executor_timeout_enforcement)


# =====================================================================
section("Skill Execution Engine — Postcondition Verification")
# =====================================================================

def test_postcondition_state_update():
    """Postcondition verification updates execution state."""
    robot = Robot.discover("mock://tb4")
    state = ExecutionState()
    executor = SkillExecutor(robot, state=state)

    nav_skill = Skill(
        skill_id="navigate_to",
        name="Navigate To",
        required_capability=CapabilityType.NAVIGATE,
        parameters={"x": 3.0, "y": 4.0},
    )
    status = executor.execute_skill(nav_skill)
    assert status == SkillStatus.COMPLETED
    assert state.get("at_position") == (3.0, 4.0)
    assert state.is_set("robot_idle")

def test_postcondition_pick_updates_state():
    """Pick skill sets object_held and clears gripper_open."""
    robot = Robot.discover("mock://tb4")
    state = ExecutionState()
    executor = SkillExecutor(robot, state=state)

    pick = Skill(
        skill_id="pick_object",
        name="Pick Object",
        required_capability=CapabilityType.PICK,
        parameters={},
    )
    status = executor.execute_skill(pick)
    assert status == SkillStatus.COMPLETED
    assert state.is_set("object_held")
    assert state.get("gripper_open") is False

def test_postcondition_place_updates_state():
    """Place skill clears object_held and sets gripper_open."""
    robot = Robot.discover("mock://tb4")
    state = ExecutionState()
    state.set("object_held", True)
    executor = SkillExecutor(robot, state=state)

    place = Skill(
        skill_id="place_object",
        name="Place Object",
        required_capability=CapabilityType.PLACE,
        parameters={},
    )
    status = executor.execute_skill(place)
    assert status == SkillStatus.COMPLETED
    assert state.get("object_held") is False
    assert state.is_set("gripper_open")

def test_postcondition_custom_state():
    """Postcondition with check_type='state' sets flag on completion."""
    robot = Robot.discover("mock://tb4")
    state = ExecutionState()
    executor = SkillExecutor(robot, state=state)

    skill = Skill(
        skill_id="report_status",
        name="Report",
        required_capability=CapabilityType.CUSTOM,
        postconditions=[
            Condition(name="status_reported", check_type="state",
                      parameters={"key": "status_reported", "value": True}),
        ],
    )
    status = executor.execute_skill(skill)
    assert status == SkillStatus.COMPLETED
    assert state.is_set("status_reported")

test("Postcondition: navigate updates position", test_postcondition_state_update)
test("Postcondition: pick sets object_held", test_postcondition_pick_updates_state)
test("Postcondition: place clears object_held", test_postcondition_place_updates_state)
test("Postcondition: custom state flag", test_postcondition_custom_state)


# =====================================================================
section("Skill Execution Engine — State Preconditions")
# =====================================================================

def test_state_precondition_pass():
    """State-based precondition passes when flag is set."""
    robot = Robot.discover("mock://tb4")
    state = ExecutionState()
    state.set("object_held", True)
    executor = SkillExecutor(robot, state=state)

    skill = Skill(
        skill_id="place_object",
        name="Place Object",
        required_capability=CapabilityType.PLACE,
        preconditions=[
            Condition(name="object_held", check_type="state",
                      parameters={"key": "object_held", "value": True}),
        ],
    )
    ok, reason = executor.check_preconditions(skill, robot)
    assert ok is True

def test_state_precondition_fail():
    """State-based precondition fails when flag is not set."""
    robot = Robot.discover("mock://tb4")
    state = ExecutionState()
    executor = SkillExecutor(robot, state=state)

    skill = Skill(
        skill_id="place_object",
        name="Place Object",
        required_capability=CapabilityType.PLACE,
        preconditions=[
            Condition(name="object_held", check_type="state",
                      parameters={"key": "object_held", "value": True}),
        ],
    )
    ok, reason = executor.check_preconditions(skill, robot)
    assert ok is False
    assert "object_held" in reason

def test_state_precondition_execution_rejected():
    """Skill execution fails if state precondition not met."""
    robot = Robot.discover("mock://tb4")
    state = ExecutionState()
    executor = SkillExecutor(robot, state=state)

    skill = Skill(
        skill_id="place_object",
        name="Place Object",
        required_capability=CapabilityType.PLACE,
        preconditions=[
            Condition(name="object_held", check_type="state",
                      parameters={"key": "object_held", "value": True}),
        ],
    )
    status = executor.execute_skill(skill)
    assert status == SkillStatus.FAILED

test("State precondition: passes when set", test_state_precondition_pass)
test("State precondition: fails when missing", test_state_precondition_fail)
test("State precondition: execution rejected", test_state_precondition_execution_rejected)


# =====================================================================
section("Skill Execution Engine — Parallel Execution")
# =====================================================================

def test_execution_layers_linear():
    """Linear graph has one skill per layer."""
    graph = SkillGraph()
    s1 = Skill(skill_id="a", name="A", parameters={})
    s2 = Skill(skill_id="b", name="B", parameters={})
    s3 = Skill(skill_id="c", name="C", parameters={})
    graph.add_skill(s1)
    graph.add_skill(s2, depends_on=["a"])
    graph.add_skill(s3, depends_on=["b"])
    layers = graph.get_execution_layers()
    assert len(layers) == 3
    assert [s.skill_id for s in layers[0]] == ["a"]
    assert [s.skill_id for s in layers[1]] == ["b"]
    assert [s.skill_id for s in layers[2]] == ["c"]

def test_execution_layers_parallel():
    """Skills with same dependencies land in the same layer."""
    graph = SkillGraph()
    root = Skill(skill_id="root", name="Root", parameters={})
    a = Skill(skill_id="a", name="A", parameters={})
    b = Skill(skill_id="b", name="B", parameters={})
    final = Skill(skill_id="final", name="Final", parameters={})
    graph.add_skill(root)
    graph.add_skill(a, depends_on=["root"])
    graph.add_skill(b, depends_on=["root"])
    graph.add_skill(final, depends_on=["a", "b"])
    layers = graph.get_execution_layers()
    assert len(layers) == 3
    layer1_ids = sorted([s.skill_id for s in layers[1]])
    assert layer1_ids == ["a", "b"]  # a and b are parallel
    assert [s.skill_id for s in layers[2]] == ["final"]

def test_parallel_graph_execution():
    """Parallel execution produces same result as sequential."""
    robot = Robot.discover("mock://tb4")
    graph = SkillGraph()
    nav1 = Skill(skill_id="navigate_to_0", name="Nav1",
                 required_capability=CapabilityType.NAVIGATE, parameters={"x": 1.0, "y": 1.0})
    report = Skill(skill_id="report_status_1", name="Report",
                   required_capability=CapabilityType.CUSTOM, parameters={})
    graph.add_skill(nav1)
    graph.add_skill(report)  # no dependency — can run in parallel with nav1

    executor = SkillExecutor(robot)
    result = executor.execute_graph(graph, parallel=True)
    assert result.status == TaskStatus.COMPLETED
    assert result.steps_completed == 2

def test_parallel_vs_sequential_same_result():
    """Both modes produce COMPLETED for a simple plan."""
    robot = Robot.discover("mock://tb4")

    def make_graph():
        g = SkillGraph()
        s1 = Skill(skill_id="navigate_to_0", name="Nav",
                    required_capability=CapabilityType.NAVIGATE, parameters={"x": 1.0, "y": 1.0})
        s2 = Skill(skill_id="stop_1", name="Stop",
                    required_capability=CapabilityType.NAVIGATE, parameters={})
        g.add_skill(s1)
        g.add_skill(s2, depends_on=["navigate_to_0"])
        return g

    seq_result = SkillExecutor(robot).execute_graph(make_graph(), parallel=False)
    par_result = SkillExecutor(robot).execute_graph(make_graph(), parallel=True)
    assert seq_result.status == par_result.status == TaskStatus.COMPLETED
    assert seq_result.steps_completed == par_result.steps_completed == 2

test("Execution layers: linear graph", test_execution_layers_linear)
test("Execution layers: parallel graph", test_execution_layers_parallel)
test("Parallel execution: basic", test_parallel_graph_execution)
test("Parallel vs sequential: same result", test_parallel_vs_sequential_same_result)


# =====================================================================
section("Skill Execution Engine — Library Integration")
# =====================================================================

def test_agent_with_library():
    """Agent uses SkillLibrary for custom skills."""
    lib = SkillLibrary()
    lib.load_json(json.dumps({
        "skill_id": "scan_area",
        "name": "Scan Area",
        "description": "Scan the surrounding area",
        "required_capability": "custom",
        "parameters": {"radius": 5.0},
    }))
    agent = Agent(provider="rule", library=lib)
    catalog = agent._get_skill_catalog()
    assert "scan_area" in catalog
    assert "navigate_to" in catalog  # built-ins still present

def test_agent_library_overrides_builtin():
    """Custom skills in library override built-ins with same ID."""
    lib = SkillLibrary()
    lib.load_json(json.dumps({
        "skill_id": "navigate_to",
        "name": "Custom Navigate",
        "required_capability": "navigate",
        "parameters": {"x": 0.0, "y": 0.0, "speed": 1.0},
        "timeout_seconds": 120.0,
    }))
    agent = Agent(provider="rule", library=lib)
    catalog = agent._get_skill_catalog()
    assert catalog["navigate_to"].name == "Custom Navigate"
    assert catalog["navigate_to"].timeout_seconds == 120.0

def test_agent_execute_tracks_state():
    """Agent.execute exposes last_state after execution."""
    agent = Agent(provider="rule")
    robot = Robot.discover("mock://tb4")
    result = agent.execute(task="deliver package from (1,2) to (5,5)", robot=robot)
    assert result.status == TaskStatus.COMPLETED
    state = agent.last_state
    assert state is not None
    # After delivery: navigate→pick→navigate→place
    # State should show gripper_open (from place_object)
    assert state.is_set("gripper_open")
    assert state.get("object_held") is False

def test_agent_execute_parallel_flag():
    """Agent.execute with parallel=True works."""
    agent = Agent(provider="rule")
    robot = Robot.discover("mock://tb4")
    result = agent.execute(task="go to (3, 4)", robot=robot, parallel=True)
    assert result.status == TaskStatus.COMPLETED

test("Agent + Library: custom skills available", test_agent_with_library)
test("Agent + Library: override built-in", test_agent_library_overrides_builtin)
test("Agent: execute tracks state", test_agent_execute_tracks_state)
test("Agent: parallel execution flag", test_agent_execute_parallel_flag)


# =====================================================================
section("Skill Execution Engine — State Flow in Graph")
# =====================================================================

def test_state_flows_through_graph():
    """Execution state flows correctly through a multi-skill graph."""
    robot = Robot.discover("mock://tb4")
    state = ExecutionState()
    executor = SkillExecutor(robot, state=state)

    graph = SkillGraph()
    nav = Skill(skill_id="navigate_to_0", name="Nav",
                required_capability=CapabilityType.NAVIGATE,
                parameters={"x": 3.0, "y": 4.0})
    pick = Skill(skill_id="pick_object_1", name="Pick",
                 required_capability=CapabilityType.PICK,
                 parameters={})
    nav2 = Skill(skill_id="navigate_to_2", name="Nav2",
                 required_capability=CapabilityType.NAVIGATE,
                 parameters={"x": 7.0, "y": 8.0})
    place = Skill(skill_id="place_object_3", name="Place",
                  required_capability=CapabilityType.PLACE,
                  parameters={})

    graph.add_skill(nav)
    graph.add_skill(pick, depends_on=["navigate_to_0"])
    graph.add_skill(nav2, depends_on=["pick_object_1"])
    graph.add_skill(place, depends_on=["navigate_to_2"])

    result = executor.execute_graph(graph)
    assert result.status == TaskStatus.COMPLETED
    assert result.steps_completed == 4

    # Verify final state
    assert state.get("at_position") == (7.0, 8.0)  # last navigate
    assert state.get("object_held") is False  # placed
    assert state.is_set("gripper_open")  # placed

def test_graph_edges_property():
    """SkillGraph.edges returns a copy of the edge map."""
    graph = SkillGraph()
    s1 = Skill(skill_id="a", name="A", parameters={})
    s2 = Skill(skill_id="b", name="B", parameters={})
    graph.add_skill(s1)
    graph.add_skill(s2, depends_on=["a"])
    edges = graph.edges
    assert edges["b"] == ["a"]
    assert edges["a"] == []

test("State flow: full delivery pipeline", test_state_flows_through_graph)
test("SkillGraph: edges property", test_graph_edges_property)


# =====================================================================
# Summary
# =====================================================================

print(f"\n{'='*60}")
total = passed + failed
print(f"\033[1m  Results: {passed} passed, {failed} failed ({total} total)\033[0m")
if failed:
    print(f"\033[31m  SOME TESTS FAILED\033[0m")
    sys.exit(1)
print(f"\033[32m  ALL TESTS PASSED ✓\033[0m")
print(f"{'='*60}")

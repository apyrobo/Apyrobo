"""
Comprehensive coverage tests for apyrobo/core/schemas.py.

Covers every enum, dataclass, and model defined in that module,
including all enum values, optional/required fields, and any
serialisation helpers.
"""

from __future__ import annotations

import pytest

from apyrobo.core.schemas import (
    AdapterState,
    Capability,
    CapabilityType,
    JointInfo,
    RecoveryAction,
    RobotCapability,
    SafetyPolicyRef,
    SensorInfo,
    SensorType,
    TaskRequest,
    TaskResult,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# TaskStatus enum
# ---------------------------------------------------------------------------

class TestTaskStatus:
    def test_all_values_exist(self) -> None:
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.STARTED == "started"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.ABORTED == "aborted"

    def test_enum_membership(self) -> None:
        all_values = {s.value for s in TaskStatus}
        assert "pending" in all_values
        assert "started" in all_values
        assert "in_progress" in all_values
        assert "completed" in all_values
        assert "failed" in all_values
        assert "aborted" in all_values

    def test_string_equality(self) -> None:
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"

    def test_count(self) -> None:
        assert len(TaskStatus) == 6


# ---------------------------------------------------------------------------
# CapabilityType enum
# ---------------------------------------------------------------------------

class TestCapabilityType:
    def test_all_values(self) -> None:
        expected = {
            "navigate", "rotate", "pick", "place", "scan",
            "speak", "manipulate", "dock", "custom",
        }
        actual = {c.value for c in CapabilityType}
        assert expected == actual

    def test_individual_values(self) -> None:
        assert CapabilityType.NAVIGATE == "navigate"
        assert CapabilityType.ROTATE == "rotate"
        assert CapabilityType.PICK == "pick"
        assert CapabilityType.PLACE == "place"
        assert CapabilityType.SCAN == "scan"
        assert CapabilityType.SPEAK == "speak"
        assert CapabilityType.MANIPULATE == "manipulate"
        assert CapabilityType.DOCK == "dock"
        assert CapabilityType.CUSTOM == "custom"

    def test_from_string(self) -> None:
        assert CapabilityType("navigate") is CapabilityType.NAVIGATE
        assert CapabilityType("custom") is CapabilityType.CUSTOM


# ---------------------------------------------------------------------------
# SensorType enum
# ---------------------------------------------------------------------------

class TestSensorType:
    def test_all_values(self) -> None:
        expected = {"camera", "lidar", "imu", "depth", "force_torque", "gps"}
        actual = {s.value for s in SensorType}
        assert expected == actual

    def test_individual_values(self) -> None:
        assert SensorType.CAMERA == "camera"
        assert SensorType.LIDAR == "lidar"
        assert SensorType.IMU == "imu"
        assert SensorType.DEPTH == "depth"
        assert SensorType.FORCE_TORQUE == "force_torque"
        assert SensorType.GPS == "gps"

    def test_from_string(self) -> None:
        assert SensorType("lidar") is SensorType.LIDAR


# ---------------------------------------------------------------------------
# AdapterState enum
# ---------------------------------------------------------------------------

class TestAdapterState:
    def test_all_values(self) -> None:
        expected = {"disconnected", "connecting", "connected", "error"}
        actual = {s.value for s in AdapterState}
        assert expected == actual

    def test_individual_values(self) -> None:
        assert AdapterState.DISCONNECTED == "disconnected"
        assert AdapterState.CONNECTING == "connecting"
        assert AdapterState.CONNECTED == "connected"
        assert AdapterState.ERROR == "error"


# ---------------------------------------------------------------------------
# RecoveryAction enum
# ---------------------------------------------------------------------------

class TestRecoveryAction:
    def test_all_values(self) -> None:
        expected = {"retry", "reroute", "escalate", "abort"}
        actual = {r.value for r in RecoveryAction}
        assert expected == actual

    def test_individual_values(self) -> None:
        assert RecoveryAction.RETRY == "retry"
        assert RecoveryAction.REROUTE == "reroute"
        assert RecoveryAction.ESCALATE == "escalate"
        assert RecoveryAction.ABORT == "abort"


# ---------------------------------------------------------------------------
# JointInfo
# ---------------------------------------------------------------------------

class TestJointInfo:
    def test_create_minimal(self) -> None:
        j = JointInfo(joint_id="joint_1", name="shoulder")
        assert j.joint_id == "joint_1"
        assert j.name == "shoulder"
        assert j.min_position is None
        assert j.max_position is None
        assert j.max_velocity is None

    def test_create_full(self) -> None:
        j = JointInfo(
            joint_id="elbow",
            name="Elbow Joint",
            min_position=-1.57,
            max_position=1.57,
            max_velocity=0.5,
        )
        assert j.joint_id == "elbow"
        assert j.name == "Elbow Joint"
        assert j.min_position == pytest.approx(-1.57)
        assert j.max_position == pytest.approx(1.57)
        assert j.max_velocity == pytest.approx(0.5)

    def test_joint_id_required(self) -> None:
        with pytest.raises((TypeError, Exception)):
            JointInfo(name="no_id")  # type: ignore[call-arg]

    def test_name_required(self) -> None:
        with pytest.raises((TypeError, Exception)):
            JointInfo(joint_id="j1")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# SensorInfo
# ---------------------------------------------------------------------------

class TestSensorInfo:
    def test_create_minimal(self) -> None:
        s = SensorInfo(sensor_id="cam0", sensor_type=SensorType.CAMERA)
        assert s.sensor_id == "cam0"
        assert s.sensor_type == SensorType.CAMERA
        assert s.topic is None
        assert s.frame_id is None
        assert s.hz is None

    def test_create_full(self) -> None:
        s = SensorInfo(
            sensor_id="lidar0",
            sensor_type=SensorType.LIDAR,
            topic="/scan",
            frame_id="laser_frame",
            hz=10.0,
        )
        assert s.sensor_id == "lidar0"
        assert s.sensor_type == SensorType.LIDAR
        assert s.topic == "/scan"
        assert s.frame_id == "laser_frame"
        assert s.hz == pytest.approx(10.0)

    def test_all_sensor_types(self) -> None:
        for st in SensorType:
            s = SensorInfo(sensor_id=f"sensor_{st.value}", sensor_type=st)
            assert s.sensor_type == st

    def test_sensor_id_required(self) -> None:
        with pytest.raises((TypeError, Exception)):
            SensorInfo(sensor_type=SensorType.IMU)  # type: ignore[call-arg]

    def test_sensor_type_required(self) -> None:
        with pytest.raises((TypeError, Exception)):
            SensorInfo(sensor_id="s1")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------

class TestCapability:
    def test_create_minimal(self) -> None:
        c = Capability(capability_type=CapabilityType.NAVIGATE, name="navigate_to")
        assert c.capability_type == CapabilityType.NAVIGATE
        assert c.name == "navigate_to"
        assert c.parameters == {}
        assert c.description == ""

    def test_create_full(self) -> None:
        c = Capability(
            capability_type=CapabilityType.PICK,
            name="pick_object",
            parameters={"max_payload_kg": 5.0},
            description="Pick up an object",
        )
        assert c.capability_type == CapabilityType.PICK
        assert c.name == "pick_object"
        assert c.parameters == {"max_payload_kg": 5.0}
        assert c.description == "Pick up an object"

    def test_all_capability_types(self) -> None:
        for ct in CapabilityType:
            cap = Capability(capability_type=ct, name=f"cap_{ct.value}")
            assert cap.capability_type == ct

    def test_capability_type_required(self) -> None:
        with pytest.raises((TypeError, Exception)):
            Capability(name="no_type")  # type: ignore[call-arg]

    def test_name_required(self) -> None:
        with pytest.raises((TypeError, Exception)):
            Capability(capability_type=CapabilityType.DOCK)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# RobotCapability
# ---------------------------------------------------------------------------

class TestRobotCapability:
    def test_create_minimal(self) -> None:
        rc = RobotCapability(robot_id="tb4-01", name="TurtleBot4-alpha")
        assert rc.robot_id == "tb4-01"
        assert rc.name == "TurtleBot4-alpha"
        assert rc.capabilities == []
        assert rc.joints == []
        assert rc.sensors == []
        assert rc.max_speed is None
        assert rc.workspace == {}
        assert rc.metadata == {}

    def test_create_with_all_fields(self) -> None:
        sensor = SensorInfo(sensor_id="cam0", sensor_type=SensorType.CAMERA, hz=30.0)
        joint = JointInfo(joint_id="j1", name="wrist", max_velocity=1.0)
        cap = Capability(capability_type=CapabilityType.NAVIGATE, name="nav")
        rc = RobotCapability(
            robot_id="arm-01",
            name="RobotArm",
            capabilities=[cap],
            joints=[joint],
            sensors=[sensor],
            max_speed=0.5,
            workspace={"x_min": -1.0, "x_max": 1.0},
            metadata={"firmware": "v2.1"},
        )
        assert rc.robot_id == "arm-01"
        assert len(rc.capabilities) == 1
        assert len(rc.joints) == 1
        assert len(rc.sensors) == 1
        assert rc.max_speed == pytest.approx(0.5)
        assert rc.workspace["x_min"] == -1.0
        assert rc.metadata["firmware"] == "v2.1"

    def test_multiple_sensors(self) -> None:
        sensors = [
            SensorInfo(sensor_id=f"s{i}", sensor_type=SensorType.LIDAR)
            for i in range(3)
        ]
        rc = RobotCapability(robot_id="r1", name="MultiSensor", sensors=sensors)
        assert len(rc.sensors) == 3

    def test_multiple_capabilities(self) -> None:
        caps = [
            Capability(capability_type=ct, name=ct.value)
            for ct in [CapabilityType.NAVIGATE, CapabilityType.PICK, CapabilityType.PLACE]
        ]
        rc = RobotCapability(robot_id="r2", name="Multi", capabilities=caps)
        assert len(rc.capabilities) == 3

    def test_robot_id_required(self) -> None:
        with pytest.raises((TypeError, Exception)):
            RobotCapability(name="no_id")  # type: ignore[call-arg]

    def test_name_required(self) -> None:
        with pytest.raises((TypeError, Exception)):
            RobotCapability(robot_id="r1")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# SafetyPolicyRef
# ---------------------------------------------------------------------------

class TestSafetyPolicyRef:
    def test_create_defaults(self) -> None:
        ref = SafetyPolicyRef()
        assert ref.policy_name == "default"
        assert ref.max_speed is None
        assert ref.collision_zones == []
        assert ref.human_proximity_limit is None

    def test_create_with_values(self) -> None:
        zone = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
        ref = SafetyPolicyRef(
            policy_name="strict",
            max_speed=0.5,
            collision_zones=[zone],
            human_proximity_limit=1.0,
        )
        assert ref.policy_name == "strict"
        assert ref.max_speed == pytest.approx(0.5)
        assert len(ref.collision_zones) == 1
        assert ref.human_proximity_limit == pytest.approx(1.0)

    def test_collision_zones_is_list(self) -> None:
        ref = SafetyPolicyRef()
        assert isinstance(ref.collision_zones, list)


# ---------------------------------------------------------------------------
# TaskRequest
# ---------------------------------------------------------------------------

class TestTaskRequest:
    def test_create_minimal(self) -> None:
        req = TaskRequest(task_name="deliver_package")
        assert req.task_name == "deliver_package"
        assert req.parameters == {}
        assert req.priority == 1
        assert req.target_robot_id is None

    def test_create_full(self) -> None:
        policy = SafetyPolicyRef(policy_name="strict", max_speed=0.3)
        req = TaskRequest(
            task_name="scan_area",
            parameters={"zone": "A"},
            priority=5,
            safety_policy=policy,
            target_robot_id="tb4-01",
        )
        assert req.task_name == "scan_area"
        assert req.parameters == {"zone": "A"}
        assert req.priority == 5
        assert req.safety_policy.policy_name == "strict"
        assert req.target_robot_id == "tb4-01"

    def test_default_safety_policy(self) -> None:
        req = TaskRequest(task_name="navigate")
        assert isinstance(req.safety_policy, SafetyPolicyRef)
        assert req.safety_policy.policy_name == "default"

    def test_task_name_required(self) -> None:
        with pytest.raises((TypeError, Exception)):
            TaskRequest()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------

class TestTaskResult:
    def test_create_minimal(self) -> None:
        result = TaskResult(task_name="navigate", status=TaskStatus.COMPLETED)
        assert result.task_name == "navigate"
        assert result.status == TaskStatus.COMPLETED
        assert result.confidence == pytest.approx(0.0)
        assert result.steps_completed == 0
        assert result.steps_total == 0
        assert result.error is None
        assert result.recovery_actions_taken == []
        assert result.metadata == {}

    def test_create_full(self) -> None:
        result = TaskResult(
            task_name="pick_and_place",
            status=TaskStatus.FAILED,
            confidence=0.75,
            steps_completed=3,
            steps_total=5,
            error="Gripper timeout",
            recovery_actions_taken=[RecoveryAction.RETRY, RecoveryAction.ESCALATE],
            metadata={"robot": "arm-01"},
        )
        assert result.status == TaskStatus.FAILED
        assert result.confidence == pytest.approx(0.75)
        assert result.steps_completed == 3
        assert result.steps_total == 5
        assert result.error == "Gripper timeout"
        assert len(result.recovery_actions_taken) == 2
        assert result.metadata["robot"] == "arm-01"

    def test_all_task_statuses(self) -> None:
        for status in TaskStatus:
            r = TaskResult(task_name="t", status=status)
            assert r.status == status

    def test_all_recovery_actions(self) -> None:
        actions = list(RecoveryAction)
        r = TaskResult(
            task_name="t",
            status=TaskStatus.ABORTED,
            recovery_actions_taken=actions,
        )
        assert len(r.recovery_actions_taken) == len(actions)

    def test_status_pending(self) -> None:
        r = TaskResult(task_name="t", status=TaskStatus.PENDING)
        assert r.status == "pending"

    def test_status_started(self) -> None:
        r = TaskResult(task_name="t", status=TaskStatus.STARTED)
        assert r.status == "started"

    def test_status_in_progress(self) -> None:
        r = TaskResult(task_name="t", status=TaskStatus.IN_PROGRESS)
        assert r.status == "in_progress"

    def test_task_name_required(self) -> None:
        with pytest.raises((TypeError, Exception)):
            TaskResult(status=TaskStatus.COMPLETED)  # type: ignore[call-arg]

    def test_status_required(self) -> None:
        with pytest.raises((TypeError, Exception)):
            TaskResult(task_name="t")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Cross-model integration
# ---------------------------------------------------------------------------

class TestCrossModelIntegration:
    """Verify that models compose correctly."""

    def test_task_request_with_collision_zones(self) -> None:
        zone = {"x_min": 0.0, "x_max": 2.0, "y_min": 0.0, "y_max": 2.0}
        policy = SafetyPolicyRef(collision_zones=[zone])
        req = TaskRequest(task_name="navigate", safety_policy=policy)
        assert len(req.safety_policy.collision_zones) == 1

    def test_robot_capability_sensor_types(self) -> None:
        sensors = [SensorInfo(sensor_id=st.value, sensor_type=st) for st in SensorType]
        rc = RobotCapability(robot_id="r", name="AllSensors", sensors=sensors)
        sensor_types = {s.sensor_type for s in rc.sensors}
        assert sensor_types == set(SensorType)

    def test_task_result_with_all_recovery_actions(self) -> None:
        all_actions = list(RecoveryAction)
        r = TaskResult(
            task_name="complex_task",
            status=TaskStatus.FAILED,
            recovery_actions_taken=all_actions,
        )
        assert RecoveryAction.ABORT in r.recovery_actions_taken
        assert RecoveryAction.RETRY in r.recovery_actions_taken

"""Tests for APYROBO core schemas — validates the capability data model."""

import pytest

from apyrobo.core.schemas import (
    Capability,
    CapabilityType,
    RecoveryAction,
    RobotCapability,
    SafetyPolicyRef,
    SensorInfo,
    SensorType,
    TaskRequest,
    TaskResult,
    TaskStatus,
    _USE_PYDANTIC,
)

# If pydantic is available, import ValidationError for stricter tests
if _USE_PYDANTIC:
    from pydantic import ValidationError


# ---------------------------------------------------------------------------
# RobotCapability
# ---------------------------------------------------------------------------

class TestRobotCapability:
    """Validates the RobotCapability schema."""

    def test_minimal_valid(self) -> None:
        cap = RobotCapability(robot_id="bot-1", name="TestBot")
        assert cap.robot_id == "bot-1"
        assert cap.capabilities == []
        assert cap.sensors == []

    def test_full_valid(self) -> None:
        cap = RobotCapability(
            robot_id="tb4-001",
            name="TurtleBot4-Alpha",
            capabilities=[
                Capability(
                    capability_type=CapabilityType.NAVIGATE,
                    name="navigate_to",
                    description="Move to 2D position",
                ),
                Capability(
                    capability_type=CapabilityType.PICK,
                    name="pick_object",
                ),
            ],
            sensors=[
                SensorInfo(
                    sensor_id="cam0",
                    sensor_type=SensorType.CAMERA,
                    topic="/camera/image_raw",
                    hz=30.0,
                ),
            ],
            max_speed=1.2,
            workspace={"x_min": -5, "x_max": 5, "y_min": -5, "y_max": 5},
        )
        assert len(cap.capabilities) == 2
        assert cap.capabilities[0].capability_type == CapabilityType.NAVIGATE
        assert cap.sensors[0].sensor_type == SensorType.CAMERA
        assert cap.max_speed == 1.2

    def test_missing_required_fields(self) -> None:
        with pytest.raises((TypeError, Exception)):
            RobotCapability()  # type: ignore[call-arg]

    @pytest.mark.skipif(not _USE_PYDANTIC, reason="Validation requires pydantic")
    def test_negative_max_speed_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RobotCapability(robot_id="x", name="X", max_speed=-1.0)

    @pytest.mark.skipif(not _USE_PYDANTIC, reason="Validation requires pydantic")
    def test_negative_sensor_hz_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SensorInfo(sensor_id="s", sensor_type=SensorType.LIDAR, hz=-5)


# ---------------------------------------------------------------------------
# TaskRequest
# ---------------------------------------------------------------------------

class TestTaskRequest:
    """Validates the TaskRequest schema."""

    def test_minimal(self) -> None:
        req = TaskRequest(task_name="deliver_package")
        assert req.task_name == "deliver_package"
        assert req.priority == 1

    def test_with_parameters(self) -> None:
        req = TaskRequest(
            task_name="deliver_package",
            parameters={"destination": "room_3", "fragile": True},
            priority=5,
            target_robot_id="tb4-001",
        )
        assert req.parameters["destination"] == "room_3"
        assert req.priority == 5

    @pytest.mark.skipif(not _USE_PYDANTIC, reason="Validation requires pydantic")
    def test_priority_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            TaskRequest(task_name="x", priority=0)
        with pytest.raises(ValidationError):
            TaskRequest(task_name="x", priority=11)

    def test_custom_safety_policy(self) -> None:
        req = TaskRequest(
            task_name="fast_delivery",
            safety_policy=SafetyPolicyRef(
                policy_name="restricted",
                max_speed=0.5,
                human_proximity_limit=1.0,
            ),
        )
        assert req.safety_policy.max_speed == 0.5
        assert req.safety_policy.human_proximity_limit == 1.0


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------

class TestTaskResult:
    """Validates the TaskResult schema."""

    def test_success(self) -> None:
        result = TaskResult(
            task_name="deliver_package",
            status=TaskStatus.COMPLETED,
            confidence=0.92,
            steps_completed=4,
            steps_total=4,
        )
        assert result.status == TaskStatus.COMPLETED
        assert result.confidence == 0.92

    def test_failure_with_recovery(self) -> None:
        result = TaskResult(
            task_name="deliver_package",
            status=TaskStatus.FAILED,
            confidence=0.0,
            steps_completed=2,
            steps_total=4,
            error="grasp_failed after max retries",
            recovery_actions_taken=[RecoveryAction.RETRY, RecoveryAction.ABORT],
        )
        assert result.status == TaskStatus.FAILED
        assert RecoveryAction.RETRY in result.recovery_actions_taken

    @pytest.mark.skipif(not _USE_PYDANTIC, reason="Validation requires pydantic")
    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            TaskResult(task_name="x", status=TaskStatus.COMPLETED, confidence=1.5)
        with pytest.raises(ValidationError):
            TaskResult(task_name="x", status=TaskStatus.COMPLETED, confidence=-0.1)

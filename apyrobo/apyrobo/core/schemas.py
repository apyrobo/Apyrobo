"""
Robot capability schemas — the semantic data model at the heart of APYROBO.

These models define what a 'robot' looks like to the rest of the framework.
They are the contract between AI agents and physical hardware: agents plan
against capabilities, not ROS 2 topics.

Note: When pydantic is available (production), these use Pydantic BaseModel
for validation.  The fallback uses dataclasses for zero-dependency development.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any

try:
    from pydantic import BaseModel, Field

    _USE_PYDANTIC = True
except ImportError:
    _USE_PYDANTIC = False

    _SENTINEL = object()

    class _DefaultFactory:
        """Wraps a factory callable for deferred default construction."""
        def __init__(self, factory):
            self.factory = factory

    # Lightweight fallback so everything works without pydantic installed
    class BaseModel:  # type: ignore[no-redef]
        """Minimal BaseModel stand-in using dataclass-style init."""

        def __init_subclass__(cls, **kwargs: Any) -> None:
            super().__init_subclass__(**kwargs)

        def __init__(self, **kwargs: Any) -> None:
            # Walk MRO to collect all annotations
            all_hints: dict[str, Any] = {}
            for klass in reversed(type(self).__mro__):
                all_hints.update(getattr(klass, "__annotations__", {}))
            for name in all_hints:
                if name.startswith("_"):
                    continue
                # Check kwargs first
                if name in kwargs:
                    setattr(self, name, kwargs[name])
                    continue
                # Check class-level default
                default = _SENTINEL
                for klass in type(self).__mro__:
                    if name in klass.__dict__:
                        default = klass.__dict__[name]
                        break
                if isinstance(default, _DefaultFactory):
                    setattr(self, name, default.factory())
                elif default is not _SENTINEL:
                    setattr(self, name, default)
                else:
                    raise TypeError(f"Missing required field: {name}")

        def __repr__(self) -> str:
            own = getattr(type(self), "__annotations__", {})
            fields = ", ".join(f"{k}={getattr(self, k, '?')!r}" for k in own)
            return f"{type(self).__name__}({fields})"

    def Field(default: Any = _SENTINEL, *, default_factory: Any = None, **kwargs: Any) -> Any:  # type: ignore[misc] # noqa: N802
        if default_factory is not None:
            return _DefaultFactory(default_factory)
        if default is _SENTINEL or default is Ellipsis:
            return _SENTINEL
        return default


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SensorType(str, Enum):
    """Sensor modalities that APYROBO understands."""

    CAMERA = "camera"
    LIDAR = "lidar"
    IMU = "imu"
    DEPTH = "depth"
    FORCE_TORQUE = "force_torque"
    GPS = "gps"


class CapabilityType(str, Enum):
    """High-level capabilities a robot can expose."""

    NAVIGATE = "navigate"
    PICK = "pick"
    PLACE = "place"
    SCAN = "scan"
    SPEAK = "speak"
    MANIPULATE = "manipulate"
    DOCK = "dock"
    CUSTOM = "custom"


class TaskStatus(str, Enum):
    """Lifecycle states for task execution."""

    PENDING = "pending"
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class RecoveryAction(str, Enum):
    """Recovery actions the execution engine can take on failure."""

    RETRY = "retry"
    REROUTE = "reroute"
    ESCALATE = "escalate"
    ABORT = "abort"


# ---------------------------------------------------------------------------
# Capability description
# ---------------------------------------------------------------------------

class SensorInfo(BaseModel):
    """Describes a single sensor available on a robot."""

    sensor_id: str = Field(..., description="Unique identifier for this sensor")
    sensor_type: SensorType
    topic: str | None = Field(None, description="ROS 2 topic this sensor publishes on")
    frame_id: str | None = Field(None, description="TF frame for this sensor")
    hz: float | None = Field(None, ge=0, description="Publishing rate in Hz")


class JointInfo(BaseModel):
    """Describes a single actuated joint."""

    joint_id: str
    name: str
    min_position: float | None = None
    max_position: float | None = None
    max_velocity: float | None = None


class Capability(BaseModel):
    """A single capability that a robot can perform."""

    capability_type: CapabilityType
    name: str = Field(..., description="Human-readable name, e.g. 'navigate_to'")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Accepted parameters for this capability",
    )
    description: str = ""


class RobotCapability(BaseModel):
    """
    Complete capability profile for a robot.

    This is what APYROBO discovers about a robot and exposes to AI agents.
    Agents plan against this — not raw ROS 2 topics.
    """

    robot_id: str = Field(..., description="Unique identifier for this robot")
    name: str = Field(..., description="Human-friendly name, e.g. 'TurtleBot4-alpha'")
    capabilities: list[Capability] = Field(default_factory=list)
    joints: list[JointInfo] = Field(default_factory=list)
    sensors: list[SensorInfo] = Field(default_factory=list)
    max_speed: float | None = Field(None, ge=0, description="Max speed in m/s")
    workspace: dict[str, Any] = Field(
        default_factory=dict,
        description="Bounding box or polygon describing reachable workspace",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Vendor info, firmware version, etc.",
    )


# ---------------------------------------------------------------------------
# Task request / result
# ---------------------------------------------------------------------------

class SafetyPolicyRef(BaseModel):
    """Reference to a safety policy that governs task execution."""

    policy_name: str = "default"
    max_speed: float | None = None
    collision_zones: list[dict[str, Any]] = Field(default_factory=list)
    human_proximity_limit: float | None = Field(
        None, ge=0, description="Minimum distance to humans in metres"
    )


class TaskRequest(BaseModel):
    """A task that an agent or user submits to APYROBO for execution."""

    task_name: str = Field(..., description="What to do, e.g. 'deliver_package'")
    parameters: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=1, ge=1, le=10)
    safety_policy: SafetyPolicyRef = Field(default_factory=SafetyPolicyRef)
    target_robot_id: str | None = Field(
        None, description="Specific robot, or None for auto-assignment"
    )


class TaskResult(BaseModel):
    """Outcome of a completed (or failed) task execution."""

    task_name: str
    status: TaskStatus
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    steps_completed: int = 0
    steps_total: int = 0
    error: str | None = None
    recovery_actions_taken: list[RecoveryAction] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

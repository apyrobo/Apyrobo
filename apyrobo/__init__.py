"""
APYROBO — Open-source AI orchestration layer for robotics.

Built on ROS 2. Model-agnostic. Hardware-agnostic.

    from apyrobo import Agent, Robot, SafetyEnforcer

    robot = Robot.discover("mock://turtlebot4")
    agent = Agent(provider="rule")
    result = agent.execute(task="deliver package to room 3", robot=robot)
"""

from apyrobo.__version__ import __version__

from apyrobo.core.robot import Robot
from apyrobo.core.health import ConnectionHealth
from apyrobo.core.adapters import (
    CapabilityAdapter, MockAdapter, GazeboAdapter, MQTTAdapter, HTTPAdapter,
    list_adapters, register_adapter, register_adapter_class,
)
from apyrobo.core.schemas import RobotCapability, TaskRequest, TaskResult, AdapterState
from apyrobo.skills.agent import (
    Agent, AgentProvider, RuleBasedProvider, LLMProvider,
    ToolCallingProvider, MultiTurnProvider, ClarificationNeeded,
    build_constrained_prompt,
)
from apyrobo.skills.skill import Skill, BUILTIN_SKILLS
from apyrobo.skills.decorators import skill, get_decorated_skills
from apyrobo.skills.executor import SkillGraph, SkillExecutor, ExecutionState, SkillTimeout
from apyrobo.safety.enforcer import (
    SafetyEnforcer, SafetyPolicy, SafetyViolation, EscalationTimeout,
    SpeedProfile, SafetyAuditEntry, FormalConstraintExporter,
    POLICY_REGISTRY,
)
from apyrobo.safety.confidence import (
    ConfidenceEstimator, ConfidenceReport, LowConfidenceError,
)
from apyrobo.swarm.bus import SwarmBus, SwarmMessage
from apyrobo.swarm.coordinator import SwarmCoordinator
from apyrobo.swarm.safety import SwarmSafety, ProximityViolation, DeadlockDetected
from apyrobo.sensors.pipeline import SensorPipeline, WorldState, SensorReading
from apyrobo.skills.library import SkillLibrary
from apyrobo.skills.package import SkillPackage
from apyrobo.skills.registry import SkillRegistry, PackageConflict, DependencyError
from apyrobo.config import ApyroboConfig
from apyrobo.inference.router import (
    InferenceRouter, Urgency, CircuitState,
    TokenBudget, PlanCache, ProviderHealth,
)
from apyrobo.observability import get_logger, trace_context, configure_logging
from apyrobo.persistence import StateStore
from apyrobo.costmap import CostmapChecker, MockCostmapChecker
from apyrobo.auth import AuthManager, GuardedRobot, AuthError
from apyrobo.task_queue import TaskQueue, QueuedTask
from apyrobo.operations import (
    BatteryMonitor, MapManager, TeleoperationBridge, WebhookEmitter,
    ScheduledTaskRunner, OperationsApiServer, FleetDashboard,
)
from apyrobo.operations import BatteryMonitor, MapManager, TeleoperationBridge, WebhookEmitter
from apyrobo.sim import (
    GazeboNativeAdapter, MuJoCoAdapter, IsaacSimAdapter,
    DomainRandomizationConfig, DomainRandomizer, RealityGapCalibrator,
    SimToRealTransferPipeline,
)

# Registers vda5050:// (imports cleanly without paho-mqtt; construction
# raises a helpful ImportError only when paho is missing AND no transport
# was injected).
from apyrobo.core import vda5050_adapter as _vda5050_adapter  # noqa: F401

# Ensure ROS 2 adapter is registered (import triggers @register_adapter).
# When rclpy is missing, ros2_bridge emits a warnings.warn and _HAS_ROS2=False,
# so the real ROS2Adapter is not registered. We register a stub instead so that
# Robot.discover("ros2://...") raises a clear RuntimeError rather than a cryptic
# "No adapter registered for scheme 'ros2'" message.
import warnings as _warnings
_ros2_loaded = False
with _warnings.catch_warnings():
    _warnings.simplefilter("ignore")  # suppress ros2_bridge's own rclpy ImportWarning
    try:
        from apyrobo.core import ros2_bridge as _ros2_bridge  # noqa: F401
        _ros2_loaded = True
    except Exception:
        pass

if not _ros2_loaded:
    _warnings.warn(
        "ROS 2 adapter not available (rclpy not installed). "
        "Install inside a ROS 2 environment or use the Docker image. "
        "Available without ROS 2: mock://, gazebo://, gazebo_native://",
        stacklevel=2,
    )

from apyrobo.core.adapters import _ADAPTER_REGISTRY as _REG, CapabilityAdapter as _CA

if "ros2" not in _REG:
    # Surface the real import failure (e.g. a missing message package), not
    # a blanket "rclpy missing" guess — ros2_bridge can import fine yet still
    # leave the adapter unregistered when any one of its ROS imports failed.
    _ros2_cause = (
        getattr(_ros2_bridge, "_ROS2_IMPORT_ERROR", None) if _ros2_loaded
        else "the apyrobo.core.ros2_bridge module failed to import"
    ) or "unknown import failure"

    class _ROS2Unavailable(_CA):
        """Stub adapter that raises a helpful error when ROS 2 imports failed."""

        def __init__(self, robot_name: str, **kwargs: object) -> None:
            raise RuntimeError(
                "The ros2:// adapter is unavailable because a ROS 2 import "
                f"failed: {_ros2_cause}\n\n"
                "ROS 2 (rclpy and friends) is only available inside the "
                "APYROBO Docker container.\n\n"
                "Quick fix:\n"
                "  docker compose -f docker/docker-compose.yml exec apyrobo bash\n\n"
                "Without Docker, use mock:// for testing or gazebo:// for sim.\n"
                "See docs/QUICKSTART.md for the full setup guide."
            )

        def get_capabilities(self):  # type: ignore[override]
            raise NotImplementedError

        def move(self, x: float, y: float, speed=None) -> None:  # type: ignore[override]
            raise NotImplementedError

        def stop(self) -> None:
            raise NotImplementedError

    _REG["ros2"] = _ROS2Unavailable

del _REG, _CA

__all__ = [
    "__version__",
    "Robot",
    "Agent",
    "skill",
    "get_decorated_skills",
    "SkillLibrary",
    "RobotCapability",
    "TaskRequest",
    "TaskResult",
    "AdapterState",
    "CapabilityAdapter",
    "MockAdapter",
    "GazeboAdapter",
    "MQTTAdapter",
    "HTTPAdapter",
    "GazeboNativeAdapter",
    "MuJoCoAdapter",
    "IsaacSimAdapter",
    "list_adapters",
    "register_adapter",
    "register_adapter_class",
    "Skill",
    "BUILTIN_SKILLS",
    "SkillGraph",
    "SkillExecutor",
    "SafetyEnforcer",
    "SafetyPolicy",
    "SafetyViolation",
    "ScheduledTaskRunner",
    "OperationsApiServer",
    "FleetDashboard",
    "CostmapChecker",
    "MockCostmapChecker",
]

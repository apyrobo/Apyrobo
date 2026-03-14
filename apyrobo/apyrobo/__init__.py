"""
APYROBO — Open-source AI orchestration layer for robotics.

Built on ROS 2. Model-agnostic. Hardware-agnostic.

    from apyrobo import Agent, Robot, SafetyEnforcer

    robot = Robot.discover("mock://turtlebot4")
    agent = Agent(provider="rule")
    result = agent.execute(task="deliver package to room 3", robot=robot)
"""

__version__ = "0.1.0-dev"

from apyrobo.core.robot import Robot
from apyrobo.core.adapters import (
    CapabilityAdapter, MockAdapter, GazeboAdapter, MQTTAdapter, HTTPAdapter,
    list_adapters, register_adapter,
)
from apyrobo.core.schemas import RobotCapability, TaskRequest, TaskResult, AdapterState
from apyrobo.skills.agent import (
    Agent, AgentProvider, RuleBasedProvider, LLMProvider,
    ToolCallingProvider, MultiTurnProvider, ClarificationNeeded,
    build_constrained_prompt,
)
from apyrobo.skills.skill import Skill, BUILTIN_SKILLS
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
from apyrobo.auth import AuthManager, GuardedRobot, AuthError
from apyrobo.task_queue import TaskQueue, QueuedTask
from apyrobo.operations import BatteryMonitor, MapManager, TeleoperationBridge, WebhookEmitter
from apyrobo.sim import (
    GazeboNativeAdapter, MuJoCoAdapter, IsaacSimAdapter,
    DomainRandomizationConfig, DomainRandomizer, RealityGapCalibrator,
    SimToRealTransferPipeline,
)

# Ensure ROS 2 adapter is registered (import triggers @register_adapter)
try:
    from apyrobo.core import ros2_bridge as _  # noqa: F401
except Exception:
    pass  # ROS 2 not available — that's fine

__all__ = [
    "__version__",
    "Robot",
    "Agent",
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
    "Skill",
    "BUILTIN_SKILLS",
    "SkillGraph",
    "SkillExecutor",
    "SafetyEnforcer",
    "SafetyPolicy",
    "SafetyViolation",
]

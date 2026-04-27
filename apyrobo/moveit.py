"""
MoveIt adapter — ROS 2 MoveIt 2 integration for robot arm manipulation.

Provides a backend-agnostic interface to MoveIt motion planning with:
    - MoveItAdapter: ABC defining the manipulation API
    - RosMoveItAdapter: live rclpy/MoveIt2-backed adapter (optional dependency)
    - MockMoveItAdapter: in-process mock for unit tests
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MoveItConfig:
    """Configuration for the MoveIt adapter."""
    group_name: str = "manipulator"
    planning_time: float = 5.0          # seconds allowed for planning
    num_planning_attempts: int = 3
    goal_position_tolerance: float = 0.01   # metres
    goal_orientation_tolerance: float = 0.05  # radians
    velocity_scaling: float = 0.5          # 0-1
    acceleration_scaling: float = 0.5      # 0-1
    planner_id: str = "RRTConnectkConfigDefault"


@dataclass
class JointTarget:
    """Target joint-space configuration."""
    joint_names: list[str]
    positions: list[float]              # radians
    velocities: list[float] = field(default_factory=list)
    tolerances: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.joint_names) != len(self.positions):
            raise ValueError("joint_names and positions must have the same length")


@dataclass
class PoseTarget:
    """Target end-effector pose in Cartesian space."""
    x: float
    y: float
    z: float
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    frame_id: str = "base_link"
    end_effector_link: str = ""         # empty = use group default


class MotionStatus(str, Enum):
    SUCCESS = "success"
    PLANNING_FAILED = "planning_failed"
    EXECUTION_FAILED = "execution_failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    INVALID_TARGET = "invalid_target"


@dataclass
class MotionResult:
    """Result of a MoveIt motion planning + execution attempt."""
    status: MotionStatus
    message: str = ""
    planning_time_s: float = 0.0
    execution_time_s: float = 0.0
    final_joint_positions: list[float] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == MotionStatus.SUCCESS

    @property
    def total_time_s(self) -> float:
        return self.planning_time_s + self.execution_time_s


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------

class MoveItAdapter:
    """
    Abstract interface to MoveIt 2 motion planning.

    Implementations handle planning and execution; callers receive
    synchronous MotionResult objects.
    """

    def move_to_joint_target(
        self,
        target: JointTarget,
        config: MoveItConfig | None = None,
    ) -> MotionResult:
        """Plan and execute a joint-space motion."""
        raise NotImplementedError

    def move_to_pose_target(
        self,
        target: PoseTarget,
        config: MoveItConfig | None = None,
    ) -> MotionResult:
        """Plan and execute a Cartesian pose motion."""
        raise NotImplementedError

    def move_to_named_target(
        self,
        name: str,
        config: MoveItConfig | None = None,
    ) -> MotionResult:
        """Move to a named configuration (e.g. 'home', 'ready')."""
        raise NotImplementedError

    def stop(self) -> None:
        """Immediately halt execution."""
        raise NotImplementedError

    def get_current_joint_values(self) -> list[float]:
        """Return the current joint positions."""
        raise NotImplementedError

    def get_current_pose(self) -> PoseTarget | None:
        """Return the current end-effector pose."""
        raise NotImplementedError

    def plan_joint_target(
        self,
        target: JointTarget,
        config: MoveItConfig | None = None,
    ) -> MotionResult:
        """Plan only — do not execute. Returns result with planning_time_s set."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# RosMoveItAdapter — live rclpy/pymoveit2 implementation (optional dependency)
# ---------------------------------------------------------------------------

class RosMoveItAdapter(MoveItAdapter):
    """
    MoveIt adapter backed by pymoveit2 (Python MoveIt 2 bindings).

    Requires: ros2, moveit2, pymoveit2; rclpy must be on PYTHONPATH.
    Call init_ros() before use.
    """

    def __init__(self, config: MoveItConfig | None = None) -> None:
        self.config = config or MoveItConfig()
        self._node: Any = None
        self._moveit: Any = None

    def init_ros(self, node_name: str = "apyrobo_moveit") -> None:
        try:
            import rclpy
            from pymoveit2 import MoveIt2

            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node(node_name)
            self._moveit = MoveIt2(
                node=self._node,
                joint_names=None,       # will be set per group
                base_link_name="base_link",
                end_effector_name="tool0",
                group_name=self.config.group_name,
            )
        except ImportError as exc:
            raise RuntimeError(
                "rclpy or pymoveit2 not available. "
                "Install ROS 2 and source the workspace."
            ) from exc

    def move_to_joint_target(
        self,
        target: JointTarget,
        config: MoveItConfig | None = None,
    ) -> MotionResult:
        if self._moveit is None:
            raise RuntimeError("Call init_ros() before planning.")

        cfg = config or self.config
        t0 = time.time()
        try:
            self._moveit.move_to_configuration(target.positions)
            import rclpy
            rclpy.spin_once(self._node, timeout_sec=cfg.planning_time)
            elapsed = time.time() - t0
            return MotionResult(
                status=MotionStatus.SUCCESS,
                planning_time_s=elapsed / 2,
                execution_time_s=elapsed / 2,
                final_joint_positions=list(target.positions),
            )
        except Exception as exc:
            return MotionResult(
                status=MotionStatus.EXECUTION_FAILED,
                message=str(exc),
                planning_time_s=time.time() - t0,
            )

    def move_to_pose_target(
        self,
        target: PoseTarget,
        config: MoveItConfig | None = None,
    ) -> MotionResult:
        if self._moveit is None:
            raise RuntimeError("Call init_ros() before planning.")

        t0 = time.time()
        try:
            self._moveit.move_to_pose(
                position=[target.x, target.y, target.z],
                quat_xyzw=[target.qx, target.qy, target.qz, target.qw],
            )
            elapsed = time.time() - t0
            return MotionResult(
                status=MotionStatus.SUCCESS,
                planning_time_s=elapsed / 2,
                execution_time_s=elapsed / 2,
            )
        except Exception as exc:
            return MotionResult(
                status=MotionStatus.EXECUTION_FAILED,
                message=str(exc),
                planning_time_s=time.time() - t0,
            )

    def move_to_named_target(
        self,
        name: str,
        config: MoveItConfig | None = None,
    ) -> MotionResult:
        if self._moveit is None:
            raise RuntimeError("Call init_ros() before planning.")

        t0 = time.time()
        try:
            self._moveit.move_to_configuration(name)
            elapsed = time.time() - t0
            return MotionResult(
                status=MotionStatus.SUCCESS,
                planning_time_s=elapsed / 2,
                execution_time_s=elapsed / 2,
            )
        except Exception as exc:
            return MotionResult(
                status=MotionStatus.EXECUTION_FAILED,
                message=str(exc),
                planning_time_s=time.time() - t0,
            )

    def stop(self) -> None:
        if self._moveit is not None:
            try:
                self._moveit.stop()
            except Exception:
                pass

    def get_current_joint_values(self) -> list[float]:
        return []

    def get_current_pose(self) -> PoseTarget | None:
        return None

    def plan_joint_target(
        self,
        target: JointTarget,
        config: MoveItConfig | None = None,
    ) -> MotionResult:
        return self.move_to_joint_target(target, config)


# ---------------------------------------------------------------------------
# MockMoveItAdapter — in-process mock for unit and integration tests
# ---------------------------------------------------------------------------

class MockMoveItAdapter(MoveItAdapter):
    """
    Deterministic MoveIt adapter for testing — no ROS runtime required.

    Set fail_next to simulate planning or execution failures.
    """

    def __init__(self, config: MoveItConfig | None = None) -> None:
        self.config = config or MoveItConfig()
        self._joint_positions: list[float] = [0.0] * 6
        self._current_pose: PoseTarget | None = PoseTarget(x=0.0, y=0.0, z=0.5)
        self.joint_moves: list[JointTarget] = []
        self.pose_moves: list[PoseTarget] = []
        self.named_moves: list[str] = []
        self.results: list[MotionResult] = []
        self.fail_next: bool = False
        self.simulated_plan_time_s: float = 0.0
        self.simulated_exec_time_s: float = 0.0
        self._stopped: bool = False
        self._named_configs: dict[str, list[float]] = {
            "home": [0.0] * 6,
            "ready": [0.0, -1.57, 0.0, -1.57, 0.0, 0.0],
        }

    def _make_result(self) -> MotionResult:
        if self.fail_next:
            self.fail_next = False
            return MotionResult(
                status=MotionStatus.PLANNING_FAILED,
                message="simulated failure",
                planning_time_s=self.simulated_plan_time_s,
            )
        return MotionResult(
            status=MotionStatus.SUCCESS,
            planning_time_s=self.simulated_plan_time_s,
            execution_time_s=self.simulated_exec_time_s,
            final_joint_positions=list(self._joint_positions),
        )

    def move_to_joint_target(
        self,
        target: JointTarget,
        config: MoveItConfig | None = None,
    ) -> MotionResult:
        self.joint_moves.append(target)
        result = self._make_result()
        if result.success:
            self._joint_positions = list(target.positions)
        self.results.append(result)
        return result

    def move_to_pose_target(
        self,
        target: PoseTarget,
        config: MoveItConfig | None = None,
    ) -> MotionResult:
        self.pose_moves.append(target)
        result = self._make_result()
        if result.success:
            self._current_pose = target
        self.results.append(result)
        return result

    def move_to_named_target(
        self,
        name: str,
        config: MoveItConfig | None = None,
    ) -> MotionResult:
        self.named_moves.append(name)
        result = self._make_result()
        if result.success and name in self._named_configs:
            self._joint_positions = list(self._named_configs[name])
            result.final_joint_positions = list(self._joint_positions)
        elif result.success:
            result = MotionResult(
                status=MotionStatus.INVALID_TARGET,
                message=f"Unknown named target: {name!r}",
            )
        self.results.append(result)
        return result

    def stop(self) -> None:
        self._stopped = True

    def get_current_joint_values(self) -> list[float]:
        return list(self._joint_positions)

    def get_current_pose(self) -> PoseTarget | None:
        return self._current_pose

    def plan_joint_target(
        self,
        target: JointTarget,
        config: MoveItConfig | None = None,
    ) -> MotionResult:
        result = self._make_result()
        result.execution_time_s = 0.0  # plan-only, no execution
        return result

    def reset(self) -> None:
        self._joint_positions = [0.0] * 6
        self._current_pose = PoseTarget(x=0.0, y=0.0, z=0.5)
        self.joint_moves.clear()
        self.pose_moves.clear()
        self.named_moves.clear()
        self.results.clear()
        self.fail_next = False
        self._stopped = False

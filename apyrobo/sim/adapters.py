"""Simulation adapters and utilities for sim-first workflows."""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

from apyrobo.core.adapters import CapabilityAdapter, register_adapter
from apyrobo.core.schemas import AdapterState, Capability, CapabilityType, RobotCapability, SensorInfo, SensorType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GazeboNotRunningError(RuntimeError):
    """Raised when a Gazebo operation is attempted but Gazebo is unavailable."""


# ---------------------------------------------------------------------------
# Joint state dataclass
# ---------------------------------------------------------------------------

@dataclass
class JointState:
    """State of a single robot joint."""
    name: str
    position: float = 0.0
    velocity: float = 0.0
    effort: float = 0.0


@register_adapter("gazebo_native")
class GazeboNativeAdapter(CapabilityAdapter):
    """
    SIM-01: Gazebo-native adapter without ROS 2 dependency.

    Supports:
    - spawn_entity / despawn_entity for model management
    - get_joint_states / set_joint_state for joint control
    - apply_force for physics interactions
    - reset_world to restore initial simulation state
    - Graceful error handling when Gazebo is not running
    """

    def __init__(self, robot_name: str, **kwargs: Any) -> None:
        super().__init__(robot_name, **kwargs)
        self._position = (0.0, 0.0)
        self._orientation = 0.0
        self._moving = False
        self._topics: set[str] = {
            f"/{robot_name}/cmd_vel",
            f"/{robot_name}/odom",
            f"/{robot_name}/scan",
        }
        self._entities: dict[str, tuple[float, float]] = {robot_name: (0.0, 0.0)}
        self._world = kwargs.get("world", "default")
        self._state = AdapterState.CONNECTED
        self._gazebo_available: bool = kwargs.get("gazebo_available", True)
        # Joint states: joint_name -> JointState
        self._joint_states: dict[str, JointState] = {}
        # Applied forces: entity -> list of force records
        self._applied_forces: list[dict[str, Any]] = []
        # Initial entity poses for world reset
        self._initial_poses: dict[str, tuple[float, float]] = {robot_name: (0.0, 0.0)}

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Whether Gazebo is considered available."""
        return self._gazebo_available and self._state == AdapterState.CONNECTED

    def _check_available(self) -> None:
        """Raise GazeboNotRunningError if Gazebo is not available."""
        if not self._gazebo_available:
            raise GazeboNotRunningError(
                f"Gazebo is not running (world={self._world!r}). "
                "Start Gazebo before performing this operation."
            )
        if self._state != AdapterState.CONNECTED:
            raise GazeboNotRunningError(
                f"GazeboNativeAdapter is not connected (state={self._state.value})."
            )

    # ------------------------------------------------------------------
    # Entity management
    # ------------------------------------------------------------------

    def spawn_entity(
        self,
        entity_name: str,
        pose: tuple[float, float] = (0.0, 0.0),
    ) -> bool:
        """
        Spawn a model in the Gazebo world.

        Args:
            entity_name: Name of the entity/model to spawn.
            pose: (x, y) position in the world frame.

        Returns True on success.
        Raises GazeboNotRunningError if Gazebo is unavailable.
        """
        self._check_available()
        self._entities[entity_name] = pose
        self._initial_poses[entity_name] = pose
        self._topics.add(f"/{entity_name}/odom")
        self._topics.add(f"/{entity_name}/scan")
        if entity_name == self.robot_name:
            self._position = pose
        logger.info("GazeboNativeAdapter: spawned entity %r at %s", entity_name, pose)
        return True

    def despawn_entity(self, entity_name: str) -> bool:
        """
        Remove a model from the Gazebo world.

        The robot itself cannot be despawned.

        Returns True if the entity existed and was removed, False otherwise.
        Raises GazeboNotRunningError if Gazebo is unavailable.
        """
        self._check_available()
        if entity_name == self.robot_name:
            logger.warning(
                "GazeboNativeAdapter: cannot despawn the robot itself (%r)",
                entity_name,
            )
            return False
        if entity_name not in self._entities:
            logger.warning(
                "GazeboNativeAdapter: entity %r not found — cannot despawn",
                entity_name,
            )
            return False
        del self._entities[entity_name]
        self._initial_poses.pop(entity_name, None)
        self._topics.discard(f"/{entity_name}/odom")
        self._topics.discard(f"/{entity_name}/scan")
        logger.info("GazeboNativeAdapter: despawned entity %r", entity_name)
        return True

    def list_entities(self) -> list[str]:
        """Return names of all entities currently in the world."""
        return sorted(self._entities.keys())

    def list_topics(self) -> list[str]:
        """Return all active Gazebo topics."""
        return sorted(self._topics)

    # ------------------------------------------------------------------
    # Joint control
    # ------------------------------------------------------------------

    def get_joint_states(self) -> dict[str, JointState]:
        """
        Return current joint states keyed by joint name.

        Raises GazeboNotRunningError if Gazebo is unavailable.
        """
        self._check_available()
        return dict(self._joint_states)

    def set_joint_state(
        self,
        joint_name: str,
        position: float,
        velocity: float = 0.0,
        effort: float = 0.0,
    ) -> bool:
        """
        Set the state of a named joint.

        Args:
            joint_name: Name of the joint (e.g. ``"left_wheel_joint"``).
            position: Target position in radians or metres.
            velocity: Target velocity (rad/s or m/s).
            effort: Effort/torque in Nm.

        Returns True on success.
        Raises GazeboNotRunningError if Gazebo is unavailable.
        """
        self._check_available()
        self._joint_states[joint_name] = JointState(
            name=joint_name,
            position=position,
            velocity=velocity,
            effort=effort,
        )
        logger.info(
            "GazeboNativeAdapter: set joint %r → pos=%.3f vel=%.3f",
            joint_name, position, velocity,
        )
        return True

    # ------------------------------------------------------------------
    # Force application
    # ------------------------------------------------------------------

    def apply_force(
        self,
        entity_name: str,
        fx: float,
        fy: float,
        fz: float = 0.0,
        duration: float = 1.0,
    ) -> bool:
        """
        Apply a force vector to an entity in the world.

        Args:
            entity_name: Target entity name.
            fx, fy, fz: Force components in Newtons (world frame).
            duration: Duration of the force application in seconds.

        Returns True on success.
        Raises GazeboNotRunningError if Gazebo is unavailable.
        Raises ValueError if the entity does not exist.
        """
        self._check_available()
        if entity_name not in self._entities:
            raise ValueError(
                f"Entity {entity_name!r} not found in world {self._world!r}. "
                f"Spawn it first with spawn_entity()."
            )
        record = {
            "entity": entity_name,
            "fx": fx, "fy": fy, "fz": fz,
            "duration": duration,
            "timestamp": time.time(),
        }
        self._applied_forces.append(record)
        logger.info(
            "GazeboNativeAdapter: applied force (%.1f, %.1f, %.1f) N to %r for %.2fs",
            fx, fy, fz, entity_name, duration,
        )
        return True

    # ------------------------------------------------------------------
    # World reset
    # ------------------------------------------------------------------

    def reset_world(self) -> bool:
        """
        Reset the Gazebo world to its initial state.

        Restores all entity poses to their spawn positions, clears joint
        states, and clears applied forces.

        Raises GazeboNotRunningError if Gazebo is unavailable.
        """
        self._check_available()
        # Restore all entity poses
        for entity_name, pose in self._initial_poses.items():
            self._entities[entity_name] = pose
            if entity_name == self.robot_name:
                self._position = pose
        # Reset joint states and forces
        self._joint_states.clear()
        self._applied_forces.clear()
        self._orientation = 0.0
        self._moving = False
        logger.info("GazeboNativeAdapter: world %r reset to initial state", self._world)
        return True

    # ------------------------------------------------------------------
    # Smoke test / diagnostics
    # ------------------------------------------------------------------

    def smoke_test(self) -> dict[str, Any]:
        """Basic connectivity smoke test."""
        try:
            spawned = self.spawn_entity(self.robot_name)
        except GazeboNotRunningError:
            return {"available": False, "error": "Gazebo not running"}
        topics = self.list_topics()
        return {
            "available": True,
            "spawned": spawned,
            "has_odom_topic": f"/{self.robot_name}/odom" in topics,
            "topic_count": len(topics),
            "world": self._world,
        }

    # ------------------------------------------------------------------
    # CapabilityAdapter required interface
    # ------------------------------------------------------------------

    def get_capabilities(self) -> RobotCapability:
        return RobotCapability(
            robot_id=self.robot_name,
            name=f"GazeboNative-{self.robot_name}",
            capabilities=[
                Capability(capability_type=CapabilityType.NAVIGATE, name="navigate_to"),
                Capability(capability_type=CapabilityType.ROTATE, name="rotate"),
            ],
            sensors=[
                SensorInfo(
                    sensor_id="lidar0",
                    sensor_type=SensorType.LIDAR,
                    topic=f"/{self.robot_name}/scan",
                    hz=10.0,
                ),
                SensorInfo(
                    sensor_id="imu0",
                    sensor_type=SensorType.IMU,
                    topic=f"/{self.robot_name}/imu",
                    hz=100.0,
                ),
            ],
            metadata={"backend": "gazebo_native", "world": self._world},
            max_speed=1.0,
        )

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        self._moving = True
        dx = x - self._position[0]
        dy = y - self._position[1]
        if dx or dy:
            self._orientation = math.atan2(dy, dx)
        self._position = (x, y)
        self._moving = False

    def stop(self) -> None:
        self._moving = False

    def get_position(self) -> tuple[float, float]:
        return self._position

    def get_orientation(self) -> float:
        return self._orientation

    def get_health(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "adapter": "GazeboNativeAdapter",
            "robot": self.robot_name,
            "world": self._world,
            "gazebo_available": self._gazebo_available,
            "entities": len(self._entities),
            "joints": len(self._joint_states),
        }


@register_adapter("mujoco")
class MuJoCoAdapter(CapabilityAdapter):
    """SIM-03: lightweight MuJoCo adapter facade."""

    def __init__(self, robot_name: str, **kwargs: Any) -> None:
        super().__init__(robot_name, **kwargs)
        self._position = (0.0, 0.0)
        self._orientation = 0.0
        self._model = kwargs.get("model", "point_mass")
        self._state = AdapterState.CONNECTED

    def get_capabilities(self) -> RobotCapability:
        return RobotCapability(
            robot_id=self.robot_name,
            name=f"MuJoCo-{self.robot_name}",
            capabilities=[Capability(capability_type=CapabilityType.NAVIGATE, name="navigate_to")],
            metadata={"backend": "mujoco", "model": self._model},
            max_speed=2.0,
        )

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        self._position = (x, y)

    def stop(self) -> None:
        pass

    def get_position(self) -> tuple[float, float]:
        return self._position


@register_adapter("isaac")
class IsaacSimAdapter(CapabilityAdapter):
    """SIM-06: Isaac Sim adapter shell via Kit SDK integration points."""

    def __init__(self, robot_name: str, **kwargs: Any) -> None:
        super().__init__(robot_name, **kwargs)
        self._position = (0.0, 0.0)
        self._state = AdapterState.CONNECTED

    def get_capabilities(self) -> RobotCapability:
        return RobotCapability(
            robot_id=self.robot_name,
            name=f"Isaac-{self.robot_name}",
            capabilities=[Capability(capability_type=CapabilityType.NAVIGATE, name="navigate_to")],
            metadata={"backend": "isaac_sim", "rendering": "rtx"},
            max_speed=1.5,
        )

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        self._position = (x, y)

    def stop(self) -> None:
        pass


@dataclass
class DomainRandomizationConfig:
    lighting_range: tuple[float, float] = (0.6, 1.4)
    texture_pool: tuple[str, ...] = ("matte", "metal", "plastic")
    position_jitter_m: float = 0.5


class DomainRandomizer:
    """SIM-04: domain randomization API for sim-to-real robustness."""

    def __init__(self, config: DomainRandomizationConfig | None = None) -> None:
        self.config = config or DomainRandomizationConfig()

    def randomize(self, scene: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
        rng = random.Random(seed)
        randomized = dict(scene)
        randomized["lighting"] = rng.uniform(*self.config.lighting_range)
        randomized["texture"] = rng.choice(self.config.texture_pool)
        randomized["position_jitter_m"] = self.config.position_jitter_m
        return randomized


class RealityGapCalibrator:
    """SIM-05: compare sim and real metrics and report discrepancies."""

    def calibrate(self, sim_metrics: dict[str, float], real_metrics: dict[str, float]) -> dict[str, Any]:
        keys = sorted(set(sim_metrics) & set(real_metrics))
        discrepancies: dict[str, float] = {}
        for key in keys:
            s = sim_metrics[key]
            r = real_metrics[key]
            denom = abs(r) if abs(r) > 1e-6 else 1.0
            discrepancies[key] = abs(s - r) / denom

        avg = sum(discrepancies.values()) / len(discrepancies) if discrepancies else 0.0
        return {
            "discrepancies": discrepancies,
            "avg_gap": avg,
            "recommendation": "tune_sim" if avg > 0.15 else "within_tolerance",
        }


class SimToRealTransferPipeline:
    """SIM-07: lightweight train-in-sim / evaluate-in-real metric tracker."""

    def __init__(self) -> None:
        self._history: list[dict[str, Any]] = []

    def train_in_sim(self, policy_name: str, episodes: int = 100) -> dict[str, Any]:
        record = {
            "stage": "sim_train",
            "policy": policy_name,
            "episodes": episodes,
            "timestamp": time.time(),
        }
        self._history.append(record)
        return record

    def evaluate_on_real(self, policy_name: str, success_rate: float) -> dict[str, Any]:
        record = {
            "stage": "real_eval",
            "policy": policy_name,
            "success_rate": success_rate,
            "timestamp": time.time(),
        }
        self._history.append(record)
        return record

    def report(self) -> dict[str, Any]:
        sim = [r for r in self._history if r["stage"] == "sim_train"]
        real = [r for r in self._history if r["stage"] == "real_eval"]
        return {
            "sim_runs": len(sim),
            "real_runs": len(real),
            "history": list(self._history),
        }

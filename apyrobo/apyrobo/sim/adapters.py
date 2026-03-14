"""Simulation adapters and utilities for sim-first workflows."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any

from apyrobo.core.adapters import CapabilityAdapter, register_adapter
from apyrobo.core.schemas import AdapterState, Capability, CapabilityType, RobotCapability, SensorInfo, SensorType


@register_adapter("gazebo_native")
class GazeboNativeAdapter(CapabilityAdapter):
    """SIM-01: Gazebo-native adapter without ROS 2 dependency."""

    def __init__(self, robot_name: str, **kwargs: Any) -> None:
        super().__init__(robot_name, **kwargs)
        self._position = (0.0, 0.0)
        self._orientation = 0.0
        self._moving = False
        self._topics = {
            f"/{robot_name}/cmd_vel",
            f"/{robot_name}/odom",
            f"/{robot_name}/scan",
        }
        self._entities: set[str] = {robot_name}
        self._world = kwargs.get("world", "default")
        self._state = AdapterState.CONNECTED

    def spawn_entity(self, entity_name: str, pose: tuple[float, float] = (0.0, 0.0)) -> bool:
        self._entities.add(entity_name)
        self._topics.add(f"/{entity_name}/odom")
        self._topics.add(f"/{entity_name}/scan")
        if entity_name == self.robot_name:
            self._position = pose
        return True

    def list_topics(self) -> list[str]:
        return sorted(self._topics)

    def smoke_test(self) -> dict[str, Any]:
        spawned = self.spawn_entity(self.robot_name)
        topics = self.list_topics()
        return {
            "spawned": spawned,
            "has_odom_topic": f"/{self.robot_name}/odom" in topics,
            "topic_count": len(topics),
        }

    def get_capabilities(self) -> RobotCapability:
        return RobotCapability(
            robot_id=self.robot_name,
            name=f"GazeboNative-{self.robot_name}",
            capabilities=[
                Capability(capability_type=CapabilityType.NAVIGATE, name="navigate_to"),
                Capability(capability_type=CapabilityType.ROTATE, name="rotate"),
            ],
            sensors=[
                SensorInfo(sensor_id="lidar0", sensor_type=SensorType.LIDAR, topic=f"/{self.robot_name}/scan", hz=10.0),
                SensorInfo(sensor_id="imu0", sensor_type=SensorType.IMU, topic=f"/{self.robot_name}/imu", hz=100.0),
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

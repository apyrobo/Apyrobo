"""
Configuration — YAML-based settings for APYROBO.

Loads configuration from a YAML file and provides typed access.
Supports: robot adapters, safety policies, skill paths, agent provider,
and simulation settings.

Usage:
    config = ApyroboConfig.from_file("config.yaml")
    policy = config.safety_policy()
    provider = config.agent_provider
    
Default config is used when no file is specified.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from apyrobo.safety.enforcer import SafetyPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "robot": {
        "uri": "mock://turtlebot4",
        "config": {},
    },
    "agent": {
        "provider": "auto",
        "model": None,
    },
    "inference": {
        "routing_enabled": False,
        "tiers": {
            "cloud": {
                "model": None,
                "max_latency_ms": 5000,
                "priority": 0,
                "supports_urgency": ["normal", "low"],
            },
            "edge": {
                "model": None,
                "max_latency_ms": 1000,
                "priority": 1,
                "supports_urgency": ["high", "normal"],
            },
        },
    },
    "safety": {
        "policy_name": "default",
        "max_speed": 1.5,
        "collision_zones": [],
        "human_proximity_limit": 0.5,
    },
    "swarm": {
        "enabled": False,
        "min_distance": 0.5,
        "deadlock_timeout": 10.0,
        "bus_type": "memory",  # "memory" or "ros2"
    },
    "sensors": {
        "enabled": True,
        "lidar_subsample": 4,
        "lidar_max_range": 10.0,
        "topics": {
            "scan": "/scan",
            "camera": "/oakd/rgb/preview/image_raw",
            "imu": "/imu",
            "odom": "/odom",
        },
    },
    "skills": {
        "custom_dir": None,
        "registry_dir": None,  # defaults to ~/.apyrobo/registry
    },
    "logging": {
        "level": "INFO",
        "structured": False,
    },
    "simulation": {
        "world": "warehouse",
        "headless": False,
        "use_sim_time": True,
    },
}


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------

class ApyroboConfig:
    """
    Typed configuration object for APYROBO.

    Merges user YAML over defaults so you only need to specify
    what you want to change.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data = self._deep_merge(DEFAULT_CONFIG, data or {})

    @classmethod
    def from_file(cls, path: str | Path) -> ApyroboConfig:
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning("Config file %s not found — using defaults", path)
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        logger.info("Loaded config from %s", path)
        return cls(data)

    @classmethod
    def from_env(cls) -> ApyroboConfig:
        """Load config from APYROBO_CONFIG env var, or use defaults."""
        config_path = os.environ.get("APYROBO_CONFIG")
        if config_path:
            return cls.from_file(config_path)
        # Check common locations
        for candidate in ["apyrobo.yaml", "config.yaml", "config/apyrobo.yaml"]:
            if Path(candidate).exists():
                return cls.from_file(candidate)
        return cls()

    # ------------------------------------------------------------------
    # Typed accessors
    # ------------------------------------------------------------------

    @property
    def robot_uri(self) -> str:
        return self._data["robot"]["uri"]

    @property
    def robot_config(self) -> dict[str, Any]:
        return self._data["robot"].get("config", {})

    @property
    def agent_provider(self) -> str:
        return self._data["agent"]["provider"]

    @property
    def agent_model(self) -> str | None:
        return self._data["agent"].get("model")

    @property
    def inference_routing_enabled(self) -> bool:
        return self._data.get("inference", {}).get("routing_enabled", False)

    @property
    def inference_tiers(self) -> dict[str, Any]:
        return self._data.get("inference", {}).get("tiers", {})

    def build_inference_router(self) -> Any:
        """Build an InferenceRouter from config. Returns None if routing disabled."""
        if not self.inference_routing_enabled:
            return None
        from apyrobo.inference.router import InferenceRouter
        tiers = self.inference_tiers
        # Filter out tiers with no model configured
        active_tiers = {k: v for k, v in tiers.items() if v.get("model")}
        if not active_tiers:
            return None
        return InferenceRouter.from_config(active_tiers)

    def safety_policy(self) -> SafetyPolicy:
        """Build a SafetyPolicy from config."""
        s = self._data["safety"]
        return SafetyPolicy(
            name=s.get("policy_name", "default"),
            max_speed=s.get("max_speed", 1.5),
            collision_zones=s.get("collision_zones", []),
            human_proximity_limit=s.get("human_proximity_limit", 0.5),
        )

    @property
    def swarm_enabled(self) -> bool:
        return self._data["swarm"]["enabled"]

    @property
    def swarm_min_distance(self) -> float:
        return self._data["swarm"]["min_distance"]

    @property
    def swarm_bus_type(self) -> str:
        return self._data["swarm"]["bus_type"]

    @property
    def sensors_enabled(self) -> bool:
        return self._data["sensors"]["enabled"]

    @property
    def sensor_topics(self) -> dict[str, str]:
        return self._data["sensors"]["topics"]

    @property
    def skills_custom_dir(self) -> str | None:
        return self._data["skills"].get("custom_dir")

    @property
    def skills_registry_dir(self) -> str | None:
        return self._data["skills"].get("registry_dir")

    @property
    def log_level(self) -> str:
        return self._data["logging"]["level"]

    @property
    def sim_world(self) -> str:
        return self._data["simulation"]["world"]

    @property
    def sim_headless(self) -> bool:
        return self._data["simulation"]["headless"]

    # ------------------------------------------------------------------
    # Raw access
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-notation access: config.get('safety.max_speed')."""
        keys = key.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def to_dict(self) -> dict[str, Any]:
        """Return the full config as a dict."""
        return dict(self._data)

    def to_yaml(self) -> str:
        """Serialise to YAML string."""
        return yaml.dump(self._data, default_flow_style=False, sort_keys=False)

    def save(self, path: str | Path) -> None:
        """Write config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)
        logger.info("Saved config to %s", path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Deep-merge override into base (override wins)."""
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ApyroboConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return (
            f"<ApyroboConfig robot={self.robot_uri!r} "
            f"agent={self.agent_provider!r} "
            f"safety={self._data['safety']['policy_name']!r}>"
        )

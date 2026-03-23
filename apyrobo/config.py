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

# TOML support: tomllib is built-in from Python 3.11; fall back to tomli on 3.10
try:
    import tomllib as _tomllib  # type: ignore[import]
except ModuleNotFoundError:
    try:
        import tomli as _tomllib  # type: ignore[import,no-redef]
    except ModuleNotFoundError:
        _tomllib = None  # type: ignore[assignment]

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
# Minimal TOML serialiser (no external deps required)
# ---------------------------------------------------------------------------

def _simple_toml_dumps(data: dict, _prefix: str = "") -> str:
    """
    Minimal TOML serialiser for the config data types used by ApyroboConfig.

    Handles: str, int, float, bool, None (omitted), list of scalars,
    and nested dicts (rendered as TOML tables).  For complex structures
    install ``tomli_w``.
    """
    lines: list[str] = []
    deferred_tables: list[tuple[str, dict]] = []

    for key, value in data.items():
        full_key = f"{_prefix}.{key}" if _prefix else key
        if value is None:
            continue
        elif isinstance(value, bool):
            lines.append(f"{key} = {str(value).lower()}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value}")
        elif isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key} = "{escaped}"')
        elif isinstance(value, list):
            items = []
            for item in value:
                if isinstance(item, bool):
                    items.append(str(item).lower())
                elif isinstance(item, (int, float)):
                    items.append(str(item))
                elif isinstance(item, str):
                    escaped = item.replace("\\", "\\\\").replace('"', '\\"')
                    items.append(f'"{escaped}"')
                else:
                    items.append(str(item))
            lines.append(f"{key} = [{', '.join(items)}]")
        elif isinstance(value, dict):
            deferred_tables.append((full_key, value))
        else:
            lines.append(f'{key} = "{value}"')

    result = "\n".join(lines)
    for table_key, table_val in deferred_tables:
        section = _simple_toml_dumps(table_val, _prefix=table_key)
        header = f"\n[{table_key}]"
        result = result + header + "\n" + section if section else result + header
    return result


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
    def from_file(cls, path: str | Path) -> "ApyroboConfig":
        """
        Load configuration from a YAML or TOML file.

        The format is inferred from the file extension:
        ``.yaml`` / ``.yml`` → YAML, ``.toml`` → TOML.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Config file %s not found — using defaults", path)
            return cls()
        suffix = path.suffix.lower()
        if suffix == ".toml":
            return cls.from_toml_file(path)
        # Default: YAML
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        logger.info("Loaded config from %s", path)
        return cls(data)

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "ApyroboConfig":
        """Load configuration explicitly from a YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning("Config file %s not found — using defaults", path)
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        logger.info("Loaded YAML config from %s", path)
        return cls(data)

    @classmethod
    def from_toml_file(cls, path: str | Path) -> "ApyroboConfig":
        """
        Load configuration from a TOML file.

        Requires Python 3.11+ (built-in ``tomllib``) or the ``tomli`` package
        on Python 3.10.  Raises ``ImportError`` if neither is available.
        """
        if _tomllib is None:
            raise ImportError(
                "TOML support requires Python 3.11+ or the 'tomli' package. "
                "Install it with: pip install tomli"
            )
        path = Path(path)
        if not path.exists():
            logger.warning("Config file %s not found — using defaults", path)
            return cls()
        with open(path, "rb") as f:
            data = _tomllib.load(f)
        logger.info("Loaded TOML config from %s", path)
        return cls(data)

    @classmethod
    def from_env(cls) -> "ApyroboConfig":
        """Load config from APYROBO_CONFIG env var, or use defaults."""
        config_path = os.environ.get("APYROBO_CONFIG")
        if config_path:
            return cls.from_file(config_path)
        # Check common locations (YAML and TOML)
        for candidate in [
            "apyrobo.yaml", "apyrobo.yml", "apyrobo.toml",
            "config.yaml", "config.yml", "config.toml",
            "config/apyrobo.yaml", "config/apyrobo.toml",
        ]:
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

    def to_toml(self) -> str:
        """
        Serialise to TOML string.

        Uses ``tomli_w`` if installed; falls back to a simple hand-rolled
        serialiser for basic types (str, int, float, bool, list, dict).
        Raises ``ImportError`` if complex types prevent serialisation.
        """
        try:
            import tomli_w  # type: ignore[import]
            return tomli_w.dumps(self._data)
        except ModuleNotFoundError:
            return _simple_toml_dumps(self._data)

    def save(self, path: str | Path) -> None:
        """
        Write config to a file.  Format is inferred from the extension:
        ``.toml`` → TOML, anything else → YAML.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".toml":
            with open(path, "w") as f:
                f.write(self.to_toml())
        else:
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

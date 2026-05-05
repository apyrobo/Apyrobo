"""Hardware knowledge schema — structured per-robot spec files.

Spec files live in ``apyrobo/hardware/specs/*.yaml``.  Each file describes
one robot model's physical capabilities so the LLM planner can reason about
what the robot can and cannot do without hallucinating.
"""
from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SPECS_DIR = Path(__file__).parent / "specs"


@dataclass
class HardwareSpec:
    """Physical and capability specification for one robot model."""

    robot_id: str                       # unique slug, e.g. "ur5", "spot", "turtlebot4"
    model: str                          # human name, e.g. "UR5"
    manufacturer: str                   # e.g. "Universal Robots"
    dof: int = 0                        # degrees of freedom (0 = mobile base only)
    payload_kg: float = 0.0            # max payload in kg
    reach_m: float = 0.0               # max reach in metres (arm only)
    max_speed_ms: float = 0.0          # max linear speed (m/s)
    sensors: list[str] = field(default_factory=list)      # e.g. ["lidar", "rgb_camera"]
    skill_package: str = ""            # pip package name, e.g. "apyrobo-skills-ur"
    ros_node_patterns: list[str] = field(default_factory=list)  # glob patterns for discovery
    notes: str = ""


class HardwareRegistry:
    """Loads and indexes HardwareSpec objects from YAML files in *specs_dir*.

    Usage::

        reg = HardwareRegistry()
        spec = reg.lookup("ur5")          # by robot_id or model name
        spec = reg.detect_from_nodes(["/ur_hardware_interface", "/joint_state_broadcaster"])
    """

    def __init__(self, specs_dir: Path | str | None = None) -> None:
        self._specs_dir = Path(specs_dir) if specs_dir else _SPECS_DIR
        self._specs: dict[str, HardwareSpec] = {}
        self._load()

    def _load(self) -> None:
        if not self._specs_dir.exists():
            logger.debug("Hardware specs dir does not exist: %s", self._specs_dir)
            return
        try:
            import yaml  # type: ignore[import]
        except ImportError:
            logger.warning("PyYAML not installed — hardware specs will not be loaded")
            return

        for yaml_path in sorted(self._specs_dir.glob("*.yaml")):
            try:
                with yaml_path.open() as fh:
                    data: dict[str, Any] = yaml.safe_load(fh) or {}
                spec = HardwareSpec(
                    robot_id=data.get("robot_id", yaml_path.stem),
                    model=data.get("model", yaml_path.stem),
                    manufacturer=data.get("manufacturer", ""),
                    dof=int(data.get("dof", 0)),
                    payload_kg=float(data.get("payload_kg", 0.0)),
                    reach_m=float(data.get("reach_m", 0.0)),
                    max_speed_ms=float(data.get("max_speed_ms", 0.0)),
                    sensors=list(data.get("sensors", [])),
                    skill_package=data.get("skill_package", ""),
                    ros_node_patterns=list(data.get("ros_node_patterns", [])),
                    notes=data.get("notes", ""),
                )
                self._specs[spec.robot_id] = spec
                logger.debug("Loaded hardware spec: %s", spec.robot_id)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", yaml_path, exc)

    def all(self) -> list[HardwareSpec]:
        """Return all loaded specs."""
        return list(self._specs.values())

    def lookup(self, model_hint: str) -> HardwareSpec | None:
        """Find a spec by robot_id or model name (case-insensitive)."""
        hint_lower = model_hint.lower()
        # Exact robot_id match
        if hint_lower in self._specs:
            return self._specs[hint_lower]
        # Substring match in robot_id or model
        for spec in self._specs.values():
            if hint_lower in spec.robot_id.lower() or hint_lower in spec.model.lower():
                return spec
        return None

    def detect_from_nodes(self, node_names: list[str]) -> HardwareSpec | None:
        """Match a list of ROS node names against registered specs' patterns.

        Returns the first spec whose *ros_node_patterns* all match at least
        one node in *node_names*.  More specific (longer pattern) specs win.
        """
        matches: list[tuple[int, HardwareSpec]] = []
        for spec in self._specs.values():
            if not spec.ros_node_patterns:
                continue
            matched = 0
            for pattern in spec.ros_node_patterns:
                if any(fnmatch.fnmatch(n, pattern) for n in node_names):
                    matched += 1
            if matched == len(spec.ros_node_patterns):
                matches.append((matched, spec))

        if not matches:
            return None
        matches.sort(key=lambda x: x[0], reverse=True)
        return matches[0][1]

    def register(self, spec: HardwareSpec) -> None:
        """Programmatically register a spec (for testing or plugins)."""
        self._specs[spec.robot_id] = spec


def spec_to_planner_context(spec: HardwareSpec) -> str:
    """Format a HardwareSpec as a natural-language paragraph for LLM context injection."""
    parts = [
        f"Robot: {spec.model} by {spec.manufacturer} (id: {spec.robot_id}).",
    ]
    if spec.dof:
        parts.append(f"It has {spec.dof} degrees of freedom.")
    if spec.payload_kg:
        parts.append(f"Maximum payload: {spec.payload_kg} kg.")
    if spec.reach_m:
        parts.append(f"Maximum reach: {spec.reach_m} m.")
    if spec.max_speed_ms:
        parts.append(f"Maximum speed: {spec.max_speed_ms} m/s.")
    if spec.sensors:
        parts.append(f"Sensors: {', '.join(spec.sensors)}.")
    if spec.notes:
        parts.append(spec.notes)
    return " ".join(parts)

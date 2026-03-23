"""
Skill Discovery — runtime discovery of available skills matched to robot capabilities.

Classes:
    SkillManifest     — metadata descriptor for one skill
    SkillDiscovery    — scans skill library and matches to robot capabilities
    DiscoveryRegistry — cached registry refreshed on demand
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SkillManifest:
    """Metadata descriptor for a discoverable skill."""

    name: str
    version: str
    description: str
    parameters: dict[str, Any]
    requirements: list[str] = field(default_factory=list)  # e.g. ["camera", "arm"]
    ros_topics: list[str] = field(default_factory=list)

    def matches_capabilities(self, available: list[str]) -> bool:
        """Return True if all requirements are satisfied by *available*."""
        return all(req in available for req in self.requirements)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "parameters": self.parameters,
            "requirements": self.requirements,
            "ros_topics": self.ros_topics,
        }


# ---------------------------------------------------------------------------
# Built-in skill manifests — mirrors the handlers registered in builtins.py
# ---------------------------------------------------------------------------

_BUILTIN_MANIFESTS: list[SkillManifest] = [
    SkillManifest(
        name="navigate_to",
        version="1.0.0",
        description="Move the robot to target (x, y) coordinates.",
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "speed": {"type": "number", "default": 0.5},
            },
            "required": ["x", "y"],
        },
        requirements=["move"],
        ros_topics=["/cmd_vel", "/odom"],
    ),
    SkillManifest(
        name="rotate",
        version="1.0.0",
        description="Rotate the robot by angle_rad radians.",
        parameters={
            "type": "object",
            "properties": {
                "angle_rad": {"type": "number"},
                "speed": {"type": "number", "default": 0.3},
            },
            "required": ["angle_rad"],
        },
        requirements=["move"],
        ros_topics=["/cmd_vel"],
    ),
    SkillManifest(
        name="stop",
        version="1.0.0",
        description="Immediately stop all robot motion.",
        parameters={"type": "object", "properties": {}},
        requirements=[],
        ros_topics=["/cmd_vel"],
    ),
    SkillManifest(
        name="pick_object",
        version="1.0.0",
        description="Close gripper to pick up an object.",
        parameters={"type": "object", "properties": {}},
        requirements=["gripper"],
        ros_topics=["/gripper/command"],
    ),
    SkillManifest(
        name="place_object",
        version="1.0.0",
        description="Open gripper to place an object.",
        parameters={"type": "object", "properties": {}},
        requirements=["gripper"],
        ros_topics=["/gripper/command"],
    ),
    SkillManifest(
        name="report_status",
        version="1.0.0",
        description="Log current robot capabilities and status.",
        parameters={"type": "object", "properties": {}},
        requirements=[],
        ros_topics=[],
    ),
    SkillManifest(
        name="speak",
        version="1.0.0",
        description="Speak text aloud using the robot's voice adapter.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        requirements=["voice"],
        ros_topics=[],
    ),
]


class SkillDiscovery:
    """
    Scans the built-in skill library and filters by robot capabilities.
    """

    def __init__(self, extra_manifests: list[SkillManifest] | None = None) -> None:
        self._library: list[SkillManifest] = list(_BUILTIN_MANIFESTS)
        if extra_manifests:
            self._library.extend(extra_manifests)

    def scan_library(self) -> list[SkillManifest]:
        """Return all known skill manifests."""
        return list(self._library)

    def match_to_capabilities(self, available_caps: list[str]) -> list[SkillManifest]:
        """Return skills whose requirements are all present in *available_caps*."""
        return [m for m in self._library if m.matches_capabilities(available_caps)]

    def register(self, manifest: SkillManifest) -> None:
        """Add a custom skill manifest to the library."""
        self._library.append(manifest)


class DiscoveryRegistry:
    """
    Cached registry that refreshes the list of available skills on demand.

    Usage:
        registry = DiscoveryRegistry()
        registry.refresh(available_capabilities=["move", "gripper"])
        registry.available_skills()   # → matched skills
        registry.get("navigate_to")   # → SkillManifest | None
    """

    def __init__(self, discovery: SkillDiscovery | None = None) -> None:
        self._discovery = discovery or SkillDiscovery()
        self._cache: list[SkillManifest] = []
        self._last_caps: list[str] = []

    def refresh(self, available_capabilities: list[str] | None = None) -> None:
        """Re-scan and filter skills, optionally with a new capability list."""
        caps = available_capabilities if available_capabilities is not None else self._last_caps
        self._last_caps = caps
        self._cache = self._discovery.match_to_capabilities(caps)
        logger.debug(
            "DiscoveryRegistry: refreshed — %d skills available (caps=%s)",
            len(self._cache), caps,
        )

    def available_skills(self) -> list[SkillManifest]:
        """Return the most-recently refreshed list of available skills."""
        return list(self._cache)

    def get(self, name: str) -> SkillManifest | None:
        """Look up a skill by name from the cached list."""
        return next((s for s in self._cache if s.name == name), None)

    def all_skills(self) -> list[SkillManifest]:
        """Return all known skills regardless of capability filter."""
        return self._discovery.scan_library()

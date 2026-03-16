"""
Skill definitions — the building blocks of robot behaviour.

A Skill is a named, reusable action with preconditions and postconditions.
Skills can be chained into a SkillGraph (directed acyclic graph) that
represents a complex task plan.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any

from apyrobo.core.schemas import BaseModel, Field, CapabilityType

logger = logging.getLogger(__name__)


class SkillStatus(str, Enum):
    """Execution state of a skill."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Condition(BaseModel):
    """A precondition or postcondition for a skill."""
    name: str
    description: str = ""
    check_type: str = "capability"  # "capability", "state", "sensor", "custom"
    parameters: dict[str, Any] = Field(default_factory=dict)


class Skill(BaseModel):
    """
    A named, reusable robot action.

    Examples: navigate_to, pick_object, deliver_package, scan_area.
    Skills are the atoms of behaviour — the skill graph chains them.
    """
    skill_id: str
    name: str
    description: str = ""
    required_capability: CapabilityType = CapabilityType.CUSTOM
    preconditions: list[Condition] = Field(default_factory=list)
    postconditions: list[Condition] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = 60.0
    retry_count: int = 0
    handler_module: str | None = None
    handler_fn: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-compatible)."""
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "required_capability": self.required_capability.value,
            "preconditions": [
                {"name": c.name, "description": c.description,
                 "check_type": c.check_type, "parameters": c.parameters}
                for c in self.preconditions
            ],
            "postconditions": [
                {"name": c.name, "description": c.description,
                 "check_type": c.check_type, "parameters": c.parameters}
                for c in self.postconditions
            ],
            "parameters": self.parameters,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "handler_module": self.handler_module,
            "handler_fn": self.handler_fn,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Skill:
        """Deserialise from a plain dict."""
        return cls(
            skill_id=data["skill_id"],
            name=data["name"],
            description=data.get("description", ""),
            required_capability=CapabilityType(data.get("required_capability", "custom")),
            preconditions=[Condition(**c) for c in data.get("preconditions", [])],
            postconditions=[Condition(**c) for c in data.get("postconditions", [])],
            parameters=data.get("parameters", {}),
            timeout_seconds=data.get("timeout_seconds", 60.0),
            retry_count=data.get("retry_count", 0),
            handler_module=data.get("handler_module"),
            handler_fn=data.get("handler_fn"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> Skill:
        return cls.from_dict(json.loads(json_str))


# ---------------------------------------------------------------------------
# Built-in skills
# ---------------------------------------------------------------------------

BUILTIN_SKILLS: dict[str, Skill] = {}


def _register_builtin(skill: Skill) -> Skill:
    BUILTIN_SKILLS[skill.skill_id] = skill
    return skill


navigate_to = _register_builtin(Skill(
    skill_id="navigate_to",
    name="Navigate To",
    description="Move the robot to a specified (x, y) position",
    required_capability=CapabilityType.NAVIGATE,
    preconditions=[
        Condition(name="robot_idle", description="Robot is not currently executing another skill"),
    ],
    postconditions=[
        Condition(name="at_position", description="Robot is within tolerance of target position",
                  parameters={"tolerance_m": 0.3}),
    ],
    parameters={"x": 0.0, "y": 0.0, "speed": 0.5},
))

rotate = _register_builtin(Skill(
    skill_id="rotate",
    name="Rotate",
    description="Rotate the robot in place by a given angle (radians)",
    required_capability=CapabilityType.ROTATE,
    preconditions=[
        Condition(name="robot_idle", description="Robot is not currently executing another skill"),
    ],
    parameters={"angle_rad": 0.0, "speed": None},
    timeout_seconds=30.0,
))

stop = _register_builtin(Skill(
    skill_id="stop",
    name="Stop",
    description="Immediately halt all robot motion",
    required_capability=CapabilityType.NAVIGATE,
    parameters={},
    timeout_seconds=5.0,
))

pick_object = _register_builtin(Skill(
    skill_id="pick_object",
    name="Pick Object",
    description="Pick up an object at or near the robot's current position",
    required_capability=CapabilityType.PICK,
    preconditions=[
        Condition(name="object_detected", description="An object is within grasp range"),
        Condition(name="gripper_open", description="Gripper is not holding anything"),
    ],
    postconditions=[
        Condition(name="object_held", description="Object is securely grasped"),
    ],
    parameters={"object_id": None},
    retry_count=2,
))

place_object = _register_builtin(Skill(
    skill_id="place_object",
    name="Place Object",
    description="Place the held object at the current position",
    required_capability=CapabilityType.PLACE,
    preconditions=[
        Condition(name="object_held", description="Robot is holding an object"),
    ],
    postconditions=[
        Condition(name="gripper_open", description="Object has been released"),
    ],
    parameters={"target_x": 0.0, "target_y": 0.0},
))

report_status = _register_builtin(Skill(
    skill_id="report_status",
    name="Report Status",
    description="Report current robot status (position, battery, held objects)",
    required_capability=CapabilityType.CUSTOM,
    parameters={},
    timeout_seconds=5.0,
))

speak = _register_builtin(Skill(
    skill_id="speak",
    name="Speak",
    description="Speak a text message aloud via text-to-speech",
    required_capability=CapabilityType.SPEAK,
    parameters={"text": ""},
    timeout_seconds=30.0,
))

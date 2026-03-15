"""
Built-in skill handlers — the 6 core handlers that ship with APYROBO.

These are auto-registered on import via the @skill_handler decorator.
"""

from __future__ import annotations

import logging
from typing import Any

from apyrobo.skills.handlers import skill_handler

logger = logging.getLogger(__name__)


@skill_handler("navigate_to")
def _navigate_to(robot: Any, params: dict[str, Any]) -> bool:
    x = float(params.get("x", 0.0))
    y = float(params.get("y", 0.0))
    speed = params.get("speed")
    speed = float(speed) if speed is not None else None
    robot.move(x=x, y=y, speed=speed)
    return True


@skill_handler("rotate")
def _rotate(robot: Any, params: dict[str, Any]) -> bool:
    angle = float(params.get("angle_rad", 0.0))
    speed = params.get("speed")
    speed = float(speed) if speed is not None else None
    robot.rotate(angle_rad=angle, speed=speed)
    return True


@skill_handler("stop")
def _stop(robot: Any, params: dict[str, Any]) -> bool:
    robot.stop()
    return True


@skill_handler("pick_object")
def _pick_object(robot: Any, params: dict[str, Any]) -> bool:
    result = robot.gripper_close()
    logger.info("pick_object → gripper_close=%s", result)
    return result


@skill_handler("place_object")
def _place_object(robot: Any, params: dict[str, Any]) -> bool:
    result = robot.gripper_open()
    logger.info("place_object → gripper_open=%s", result)
    return result


@skill_handler("report_status")
def _report_status(robot: Any, params: dict[str, Any]) -> bool:
    caps = robot.capabilities()
    logger.info(
        "Status: robot=%s capabilities=%d sensors=%d",
        caps.name, len(caps.capabilities), len(caps.sensors),
    )
    return True

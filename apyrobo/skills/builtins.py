"""
Built-in skill handlers — the 6 core handlers that ship with APYROBO.

These are auto-registered on import via the @skill_handler decorator.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from apyrobo.skills.handlers import skill_handler

logger = logging.getLogger(__name__)


def _target_coordinate(params: dict[str, Any], key: str) -> float:
    """Return params[key] as a float, or raise ValueError naming the key."""
    if params.get(key) is None:
        raise ValueError(
            f"navigate_to needs a target: {key!r} is missing "
            "(pass both x and y; the robot did not move)"
        )
    value = params[key]
    not_a_number = ValueError(
        f"navigate_to {key!r} must be a finite number, got {value!r} "
        "(the robot did not move)"
    )
    if isinstance(value, bool):
        raise not_a_number
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise not_a_number from None
    if not math.isfinite(number):
        raise not_a_number
    return number


@skill_handler("navigate_to")
def _navigate_to(robot: Any, params: dict[str, Any]) -> bool:
    # No default target: a navigate_to without x or y is refused before the
    # robot moves, instead of driving to a made-up (0, 0).
    x = _target_coordinate(params, "x")
    y = _target_coordinate(params, "y")
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


@skill_handler("speak")
def _speak(robot: Any, params: dict[str, Any]) -> bool:
    """VC-01: Dispatch speak to a VoiceAdapter if available."""
    text = params.get("text", "")
    if not text:
        logger.warning("speak handler: no text provided")
        return False
    # Try robot's voice adapter if attached
    adapter = getattr(robot, "_voice_adapter", None)
    if adapter is not None:
        adapter.speak(text)
        return True
    logger.info("speak handler: no voice adapter, text=%r", text)
    return True


@skill_handler("run_policy")
def _run_policy(robot: Any, params: dict[str, Any]) -> bool:
    """Run a learned policy as a bounded control loop (apyrobo.skills.policy)."""
    from apyrobo.skills.policy import PolicyRunner

    policy = params.get("policy")
    if policy is None:
        logger.warning("run_policy handler: no 'policy' in parameters")
        return False
    runner_keys = (
        "hz", "max_duration_sec", "max_steps", "max_step_m",
        "speed", "success", "observe", "real_time",
    )
    kwargs = {k: params[k] for k in runner_keys if k in params}
    result = PolicyRunner(policy, robot, **kwargs).run()
    return result.success

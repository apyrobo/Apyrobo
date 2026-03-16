"""
Extended built-in skill handlers — additional handlers beyond the core 6.

Auto-registered via @skill_handler when imported.
"""

from __future__ import annotations

import logging
from typing import Any

from apyrobo.skills.handlers import skill_handler

logger = logging.getLogger(__name__)


@skill_handler("report_battery_status")
def _report_battery_status(robot: Any, params: dict[str, Any]) -> bool:
    health = robot.get_health()
    battery = health.get("battery_pct", "unknown")
    logger.info("Battery status: %s%%", battery)
    return True


@skill_handler("waypoint_tour")
def _waypoint_tour(robot: Any, params: dict[str, Any]) -> bool:
    waypoints = params.get("waypoints", [])
    speed = params.get("speed", 0.5)
    loops = int(params.get("loops", 1))
    for _ in range(loops):
        for wp in waypoints:
            x = float(wp.get("x", 0.0))
            y = float(wp.get("y", 0.0))
            robot.move(x=x, y=y, speed=speed)
    return True


@skill_handler("dock_to_charger")
def _dock_to_charger(robot: Any, params: dict[str, Any]) -> bool:
    dock_x = float(params.get("dock_x", 0.0))
    dock_y = float(params.get("dock_y", 0.0))
    robot.move(x=dock_x, y=dock_y, speed=0.3)
    robot.connect()
    return True


@skill_handler("patrol_route")
def _patrol_route(robot: Any, params: dict[str, Any]) -> bool:
    waypoints = params.get("waypoints", [])
    speed = params.get("speed", 0.5)
    loops = int(params.get("loops", 1))
    for _ in range(loops):
        for wp in waypoints:
            x = float(wp.get("x", 0.0))
            y = float(wp.get("y", 0.0))
            robot.move(x=x, y=y, speed=speed)
            # Scan at each waypoint
            robot.rotate(angle_rad=6.283, speed=0.3)
            caps = robot.capabilities()
            logger.info(
                "Patrol checkpoint (%.1f, %.1f): %d sensors active",
                x, y, len(caps.sensors),
            )
    return True


@skill_handler("scan_area")
def _scan_area(robot: Any, params: dict[str, Any]) -> bool:
    rotation_speed = float(params.get("rotation_speed", 0.3))
    full_rotations = int(params.get("full_rotations", 1))
    for _ in range(full_rotations):
        robot.rotate(angle_rad=6.283, speed=rotation_speed)
    caps = robot.capabilities()
    logger.info("Scan complete: %d sensors reported", len(caps.sensors))
    return True

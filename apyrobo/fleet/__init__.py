"""
Fleet Manager — centralized management of multi-robot fleets.

Handles robot registration, heartbeats, load-balanced task assignment,
and offline detection.
"""

from apyrobo.fleet.manager import FleetManager, RobotInfo

__all__ = ["FleetManager", "RobotInfo"]

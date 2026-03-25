"""
Fleet Manager — centralized management of multi-robot fleets.

Handles robot registration, heartbeats, load-balanced task assignment,
and offline detection.
"""

from apyrobo.fleet.manager import FleetManager, RobotInfo
from apyrobo.fleet.multisite import MultiSiteManager, SiteConfig, SiteStatus, MultiSiteError

__all__ = [
    "FleetManager",
    "RobotInfo",
    "MultiSiteManager",
    "SiteConfig",
    "SiteStatus",
    "MultiSiteError",
]

"""APYROBO skill wrappers for ROS 2 Nav2 navigation actions.

Each skill guards the rclpy import at call-time so the module loads cleanly
in environments without ROS 2.  When rclpy is missing the skill raises an
``ImportError`` with installation instructions rather than silently failing.

Typical usage::

    from apyrobo_skills_ros_nav.skills import navigate_to_pose
    result = navigate_to_pose(robot, x=2.0, y=1.5, yaw=0.0)
"""
from __future__ import annotations

import logging

from apyrobo import skill

logger = logging.getLogger(__name__)

_ROS2_MISSING_MSG = (
    "ROS 2 rclpy is required. Source your ROS 2 workspace first.\n"
    "  e.g.: source /opt/ros/humble/setup.bash && source install/setup.bash"
)


# ---------------------------------------------------------------------------
# navigate_to_pose
# ---------------------------------------------------------------------------

@skill(
    description="Send a Nav2 navigate_to_pose action goal",
    capability="navigate",
    timeout=35.0,
)
def navigate_to_pose(
    robot,
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
    frame_id: str = "map",
) -> bool:
    """Navigate to (x, y, yaw) in frame_id using Nav2 NavigateToPose action.

    Args:
        robot:    APYROBO Robot instance (passed automatically by the executor).
        x:        Target X position in *frame_id* (metres).
        y:        Target Y position in *frame_id* (metres).
        yaw:      Target heading in radians (default 0.0).
        frame_id: Coordinate frame for the goal (default ``"map"``).

    Returns:
        ``True`` on success, ``False`` on failure, cancel, or timeout.

    Raises:
        ImportError: If ``rclpy`` is not installed.
    """
    try:
        import rclpy  # noqa: F401  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(_ROS2_MISSING_MSG) from exc

    from apyrobo_skills_ros_nav.nav2_client import Nav2Client

    logger.info(
        "navigate_to_pose: navigating to (%.3f, %.3f, yaw=%.3f) in frame '%s'",
        x, y, yaw, frame_id,
    )
    with Nav2Client() as client:
        success = client.navigate_to_pose(x=x, y=y, yaw=yaw, frame_id=frame_id)

    if success:
        logger.info("navigate_to_pose: reached (%.3f, %.3f)", x, y)
    else:
        logger.warning("navigate_to_pose: failed to reach (%.3f, %.3f)", x, y)
    return success


# ---------------------------------------------------------------------------
# follow_path
# ---------------------------------------------------------------------------

@skill(
    description="Follow a list of (x, y) waypoints using Nav2",
    capability="navigate",
    timeout=125.0,
)
def follow_path(
    robot,
    waypoints: list | None = None,
) -> bool:
    """Follow a sequence of (x, y) waypoints using the Nav2 FollowWaypoints action.

    Args:
        robot:     APYROBO Robot instance.
        waypoints: List of ``[x, y]`` pairs (or tuples) in the ``map`` frame.
                   Pass ``None`` or ``[]`` to return immediately with ``True``.

    Returns:
        ``True`` if all waypoints were reached, ``False`` otherwise.

    Raises:
        ImportError: If ``rclpy`` is not installed.
    """
    try:
        import rclpy  # noqa: F401  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(_ROS2_MISSING_MSG) from exc

    from apyrobo_skills_ros_nav.nav2_client import Nav2Client

    if waypoints is None:
        waypoints = []

    # Normalise [[x, y], ...] or [(x, y), ...] to list of tuples
    normalised = [(float(wp[0]), float(wp[1])) for wp in waypoints]

    logger.info("follow_path: following %d waypoint(s)", len(normalised))
    with Nav2Client() as client:
        success = client.follow_path(waypoints=normalised)

    if success:
        logger.info("follow_path: all waypoints reached")
    else:
        logger.warning("follow_path: did not complete all waypoints")
    return success


# ---------------------------------------------------------------------------
# clear_costmaps
# ---------------------------------------------------------------------------

@skill(
    description="Clear Nav2 global and local costmaps",
    capability="navigate",
    timeout=15.0,
)
def clear_costmaps(robot) -> bool:
    """Call the Nav2 clear-costmap services for both global and local costmaps.

    Args:
        robot: APYROBO Robot instance.

    Returns:
        ``True`` if both costmaps were cleared successfully, ``False`` otherwise.

    Raises:
        ImportError: If ``rclpy`` is not installed.
    """
    try:
        import rclpy  # noqa: F401  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(_ROS2_MISSING_MSG) from exc

    from apyrobo_skills_ros_nav.nav2_client import Nav2Client

    logger.info("clear_costmaps: clearing global and local costmaps")
    with Nav2Client() as client:
        success = client.clear_costmaps()

    if success:
        logger.info("clear_costmaps: both costmaps cleared")
    else:
        logger.warning("clear_costmaps: one or more costmap clears failed")
    return success


# ---------------------------------------------------------------------------
# nav2_recover
# ---------------------------------------------------------------------------

@skill(
    description="Trigger Nav2 recovery behaviours",
    capability="navigate",
    timeout=20.0,
)
def nav2_recover(robot) -> bool:
    """Execute Nav2 recovery behaviours (backup manoeuvre).

    Sends a short backup motion via the Nav2 BackUp action as a recovery
    entry-point.  Operators can extend this to chain Spin or ClearCostmap
    recoveries as needed by their robot's nav2_params configuration.

    Args:
        robot: APYROBO Robot instance.

    Returns:
        ``True`` if recovery completed, ``False`` on failure or timeout.

    Raises:
        ImportError: If ``rclpy`` is not installed.
    """
    try:
        import rclpy  # noqa: F401  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(_ROS2_MISSING_MSG) from exc

    from apyrobo_skills_ros_nav.nav2_client import Nav2Client

    logger.info("nav2_recover: triggering Nav2 recovery behaviours")
    with Nav2Client() as client:
        success = client.recover()

    if success:
        logger.info("nav2_recover: recovery complete")
    else:
        logger.warning("nav2_recover: recovery did not complete successfully")
    return success

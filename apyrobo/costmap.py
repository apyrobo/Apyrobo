"""
Nav2 costmap pre-validation for APYROBO.

Subscribes to /global_costmap/costmap (nav_msgs/OccupancyGrid) and validates
navigation goals before they are sent to Nav2, rejecting goals in lethal
obstacle cells and warning on inscribed/unknown cells.

Cost thresholds (ROS 2 Nav2 convention):
    0–252   free / traversable   → allow
    253     inscribed             → warn + allow
    254     lethal obstacle       → reject (raise ValueError in navigate_to)
    255     unknown               → warn + allow

Usage (real Nav2)::

    checker = CostmapChecker(ros2_node)
    adapter = Nav2Adapter(config)
    adapter.set_costmap_checker(checker)
    await adapter.navigate_to(NavigationGoal(x=2.0, y=3.0))

Usage (tests / mock)::

    checker = MockCostmapChecker()
    checker.block(5.0, 5.0)          # mark one cell lethal
    adapter = MockNav2Adapter()
    adapter.set_costmap_checker(checker)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

logger = logging.getLogger(__name__)

# ROS 2 Nav2 occupancy cost thresholds
COST_FREE_MAX = 252
COST_INSCRIBED = 253
COST_LETHAL = 254
COST_UNKNOWN = 255


class CostmapChecker:
    """
    Subscribes to /global_costmap/costmap and validates navigation goals.

    When rclpy / nav_msgs are available (inside Docker / ROS 2 environment)
    the subscription is live.  Otherwise the checker silently operates in
    stub mode (``is_ready`` stays False, all goals are allowed through).
    """

    def __init__(self, node: Any) -> None:
        self._node = node
        self._data: list[int] | None = None
        self._resolution: float = 0.05
        self._origin_x: float = 0.0
        self._origin_y: float = 0.0
        self._width: int = 0
        self._height: int = 0
        self._ready = False

        try:
            from nav_msgs.msg import OccupancyGrid  # type: ignore[import]
            self._sub = node.create_subscription(
                OccupancyGrid,
                "/global_costmap/costmap",
                self._on_costmap,
                10,
            )
            logger.info("CostmapChecker: subscribed to /global_costmap/costmap")
        except Exception as exc:
            logger.warning("CostmapChecker: nav_msgs unavailable (%s) — stub mode", exc)
            self._sub = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_costmap(self, msg: Any) -> None:
        info = msg.info
        self._resolution = info.resolution
        self._origin_x = info.origin.position.x
        self._origin_y = info.origin.position.y
        self._width = info.width
        self._height = info.height
        self._data = list(msg.data)
        if not self._ready:
            logger.info(
                "CostmapChecker: first map received (%dx%d @ %.3f m/cell)",
                self._width, self._height, self._resolution,
            )
        self._ready = True

    def _world_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        """Convert world (x, y) to grid (col, row). Returns None if out of bounds."""
        col = int((x - self._origin_x) / self._resolution)
        row = int((y - self._origin_y) / self._resolution)
        if 0 <= col < self._width and 0 <= row < self._height:
            return col, row
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True once at least one costmap message has been received."""
        return self._ready

    def get_cost(self, x: float, y: float) -> int | None:
        """
        Return the occupancy cost at world position (x, y), or None if unknown.

        None is returned when no costmap has been received yet or the point
        is outside the costmap bounds.
        """
        if self._data is None:
            return None
        cell = self._world_to_cell(x, y)
        if cell is None:
            return None
        col, row = cell
        idx = row * self._width + col
        if 0 <= idx < len(self._data):
            return self._data[idx]
        return None

    def is_goal_valid(self, x: float, y: float) -> tuple[bool, str]:
        """
        Validate a navigation goal at world position (x, y).

        Returns:
            (True, reason)  if the goal may be sent to Nav2.
            (False, reason) if the goal must be rejected (lethal cell).

        The caller is responsible for raising an error on False.
        """
        cost = self.get_cost(x, y)

        if cost is None:
            return True, "no costmap data — proceeding"

        if cost <= COST_FREE_MAX:
            return True, f"free (cost={cost})"

        if cost == COST_INSCRIBED:
            warnings.warn(
                f"Goal ({x:.2f}, {y:.2f}) is in inscribed zone (cost=253) — "
                "robot footprint may clip an obstacle",
                stacklevel=4,
            )
            return True, "inscribed — footprint warning"

        if cost == COST_LETHAL:
            return False, f"Goal ({x:.2f}, {y:.2f}) is in a lethal obstacle cell (cost=254)"

        if cost == COST_UNKNOWN:
            warnings.warn(
                f"Goal ({x:.2f}, {y:.2f}) has unknown cost (255) — no sensor data at target",
                stacklevel=4,
            )
            return True, "unknown cost — proceeding with caution"

        return True, f"cost={cost}"


class MockCostmapChecker:
    """
    Always-valid costmap checker for tests and offline development.

    All cells are free by default.  Call ``block(x, y)`` to mark a specific
    world coordinate as a lethal obstacle — ``is_goal_valid`` will then
    reject any goal at exactly that position.

    Example::

        checker = MockCostmapChecker()
        checker.block(5.0, 5.0)
        ok, reason = checker.is_goal_valid(5.0, 5.0)
        assert not ok
    """

    def __init__(self) -> None:
        self._blocked: set[tuple[float, float]] = set()

    def block(self, x: float, y: float) -> None:
        """Mark (x, y) as a lethal obstacle."""
        self._blocked.add((x, y))

    def unblock(self, x: float, y: float) -> None:
        """Remove the lethal block at (x, y)."""
        self._blocked.discard((x, y))

    @property
    def is_ready(self) -> bool:
        return True

    def get_cost(self, x: float, y: float) -> int | None:
        """Return COST_LETHAL (254) if blocked, else 0 (free)."""
        return COST_LETHAL if (x, y) in self._blocked else 0

    def is_goal_valid(self, x: float, y: float) -> tuple[bool, str]:
        cost = self.get_cost(x, y)
        if cost == COST_LETHAL:
            return False, f"Goal ({x:.2f}, {y:.2f}) is in a lethal obstacle cell (cost=254)"
        return True, "free"

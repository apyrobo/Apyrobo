"""Tests for apyrobo.costmap — CostmapChecker and MockCostmapChecker."""
from __future__ import annotations

import asyncio
import warnings

import pytest

from apyrobo.costmap import (
    CostmapChecker,
    MockCostmapChecker,
    COST_FREE_MAX,
    COST_INSCRIBED,
    COST_LETHAL,
    COST_UNKNOWN,
)
from apyrobo.nav2 import MockNav2Adapter, NavigationGoal


# ---------------------------------------------------------------------------
# MockCostmapChecker
# ---------------------------------------------------------------------------

class TestMockCostmapChecker:
    def test_is_ready_always_true(self):
        checker = MockCostmapChecker()
        assert checker.is_ready is True

    def test_all_cells_free_by_default(self):
        checker = MockCostmapChecker()
        ok, reason = checker.is_goal_valid(1.0, 2.0)
        assert ok is True
        assert "free" in reason

    def test_get_cost_free_by_default(self):
        checker = MockCostmapChecker()
        assert checker.get_cost(0.0, 0.0) == 0

    def test_block_makes_cell_lethal(self):
        checker = MockCostmapChecker()
        checker.block(3.0, 4.0)
        ok, reason = checker.is_goal_valid(3.0, 4.0)
        assert ok is False
        assert "lethal" in reason

    def test_blocked_get_cost_returns_lethal(self):
        checker = MockCostmapChecker()
        checker.block(1.0, 1.0)
        assert checker.get_cost(1.0, 1.0) == COST_LETHAL

    def test_unblock_restores_free(self):
        checker = MockCostmapChecker()
        checker.block(2.0, 2.0)
        checker.unblock(2.0, 2.0)
        ok, _ = checker.is_goal_valid(2.0, 2.0)
        assert ok is True

    def test_unblock_nonexistent_is_noop(self):
        checker = MockCostmapChecker()
        checker.unblock(99.0, 99.0)  # should not raise

    def test_block_multiple_cells(self):
        checker = MockCostmapChecker()
        checker.block(1.0, 0.0)
        checker.block(2.0, 0.0)
        assert checker.is_goal_valid(1.0, 0.0)[0] is False
        assert checker.is_goal_valid(2.0, 0.0)[0] is False
        assert checker.is_goal_valid(3.0, 0.0)[0] is True

    def test_other_positions_unaffected_by_block(self):
        checker = MockCostmapChecker()
        checker.block(5.0, 5.0)
        ok, _ = checker.is_goal_valid(5.1, 5.0)
        assert ok is True


# ---------------------------------------------------------------------------
# CostmapChecker — before data arrives
# ---------------------------------------------------------------------------

class TestCostmapCheckerBeforeReady:
    def _make_checker(self) -> CostmapChecker:
        """Create a CostmapChecker without a real node (nav_msgs import will fail)."""

        class _FakeNode:
            def create_subscription(self, *args, **kwargs):
                return None

        return CostmapChecker(_FakeNode())

    def test_not_ready_before_first_message(self):
        checker = self._make_checker()
        assert checker.is_ready is False

    def test_get_cost_returns_none_before_ready(self):
        checker = self._make_checker()
        assert checker.get_cost(0.0, 0.0) is None

    def test_is_goal_valid_allows_when_no_data(self):
        checker = self._make_checker()
        ok, reason = checker.is_goal_valid(1.0, 1.0)
        assert ok is True
        assert "no costmap" in reason


# ---------------------------------------------------------------------------
# CostmapChecker — injecting costmap data via _on_costmap
# ---------------------------------------------------------------------------

class _FakeOccupancyGrid:
    """Minimal stand-in for nav_msgs/OccupancyGrid."""

    class _Info:
        def __init__(self, resolution, ox, oy, w, h):
            self.resolution = resolution
            self.origin = type("P", (), {"position": type("XY", (), {"x": ox, "y": oy})()})()
            self.width = w
            self.height = h

    def __init__(self, width, height, resolution=1.0, ox=0.0, oy=0.0, fill=0):
        self.info = self._Info(resolution, ox, oy, width, height)
        self.data = [fill] * (width * height)


def _make_ready_checker(width=10, height=10, resolution=1.0, fill=0) -> CostmapChecker:
    class _FakeNode:
        def create_subscription(self, *args, **kwargs):
            return None

    checker = CostmapChecker(_FakeNode())
    grid = _FakeOccupancyGrid(width, height, resolution, fill=fill)
    checker._on_costmap(grid)
    return checker


class TestCostmapCheckerWithData:
    def test_is_ready_after_first_message(self):
        checker = _make_ready_checker()
        assert checker.is_ready is True

    def test_free_cell_returns_valid(self):
        checker = _make_ready_checker(fill=0)
        ok, reason = checker.is_goal_valid(0.0, 0.0)
        assert ok is True
        assert "free" in reason

    def test_cost_252_is_valid(self):
        checker = _make_ready_checker(fill=COST_FREE_MAX)
        ok, _ = checker.is_goal_valid(0.0, 0.0)
        assert ok is True

    def test_lethal_cell_is_rejected(self):
        checker = _make_ready_checker(fill=COST_LETHAL)
        ok, reason = checker.is_goal_valid(0.0, 0.0)
        assert ok is False
        assert "lethal" in reason

    def test_inscribed_cell_warns_and_allows(self):
        checker = _make_ready_checker(fill=COST_INSCRIBED)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ok, reason = checker.is_goal_valid(0.0, 0.0)
        assert ok is True
        assert "inscribed" in reason
        assert any("inscribed" in str(w.message) for w in caught)

    def test_unknown_cell_warns_and_allows(self):
        checker = _make_ready_checker(fill=COST_UNKNOWN)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ok, reason = checker.is_goal_valid(0.0, 0.0)
        assert ok is True
        assert "unknown" in reason.lower()
        assert any("unknown" in str(w.message).lower() for w in caught)

    def test_out_of_bounds_returns_none(self):
        checker = _make_ready_checker(width=5, height=5)
        assert checker.get_cost(100.0, 100.0) is None

    def test_get_cost_specific_cell(self):
        checker = _make_ready_checker(width=5, height=5, resolution=1.0, fill=0)
        # Manually set one cell lethal
        checker._data[2 * 5 + 3] = COST_LETHAL  # row=2, col=3 → (x=3.0, y=2.0)
        assert checker.get_cost(3.0, 2.0) == COST_LETHAL

    def test_world_to_cell_with_offset_origin(self):
        class _FakeNode:
            def create_subscription(self, *args, **kwargs):
                return None

        checker = CostmapChecker(_FakeNode())
        grid = _FakeOccupancyGrid(10, 10, resolution=0.5, ox=-2.5, oy=-2.5)
        checker._on_costmap(grid)
        # World (0, 0) → col=(0-(-2.5))/0.5=5, row=5 → idx=5*10+5=55
        cost = checker.get_cost(0.0, 0.0)
        assert cost == 0


# ---------------------------------------------------------------------------
# Nav2Adapter integration
# ---------------------------------------------------------------------------

class TestNav2AdapterCostmapIntegration:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_navigate_succeeds_without_checker(self):
        adapter = MockNav2Adapter()
        self._run(adapter.connect())
        result = self._run(adapter.navigate_to(NavigationGoal(x=1.0, y=1.0)))
        assert result.success is True

    def test_navigate_succeeds_with_free_checker(self):
        adapter = MockNav2Adapter()
        self._run(adapter.connect())
        checker = MockCostmapChecker()
        adapter.set_costmap_checker(checker)
        result = self._run(adapter.navigate_to(NavigationGoal(x=1.0, y=1.0)))
        assert result.success is True

    def test_navigate_raises_on_lethal_goal(self):
        adapter = MockNav2Adapter()
        self._run(adapter.connect())
        checker = MockCostmapChecker()
        checker.block(3.0, 4.0)
        adapter.set_costmap_checker(checker)
        with pytest.raises(ValueError, match="lethal"):
            self._run(adapter.navigate_to(NavigationGoal(x=3.0, y=4.0)))

    def test_navigate_warns_on_inscribed_goal(self):
        adapter = MockNav2Adapter()
        self._run(adapter.connect())

        class _InscribedChecker(MockCostmapChecker):
            def is_goal_valid(self, x, y):
                warnings.warn("inscribed zone", stacklevel=3)
                return True, "inscribed — footprint warning"

        adapter.set_costmap_checker(_InscribedChecker())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self._run(adapter.navigate_to(NavigationGoal(x=1.0, y=1.0)))
        assert result.success is True
        assert any("inscribed" in str(w.message) for w in caught)

    def test_checker_skipped_when_not_ready(self):
        adapter = MockNav2Adapter()
        self._run(adapter.connect())

        class _NotReadyChecker:
            is_ready = False

            def is_goal_valid(self, x, y):
                raise AssertionError("should not be called when not ready")

        adapter.set_costmap_checker(_NotReadyChecker())
        result = self._run(adapter.navigate_to(NavigationGoal(x=1.0, y=1.0)))
        assert result.success is True

    def test_set_costmap_checker_stores_instance(self):
        adapter = MockNav2Adapter()
        checker = MockCostmapChecker()
        adapter.set_costmap_checker(checker)
        assert adapter._costmap_checker is checker

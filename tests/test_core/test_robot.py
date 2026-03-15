"""Tests for Robot discovery and the mock adapter."""

import pytest

from apyrobo.core.adapters import MockAdapter
from apyrobo.core.robot import Robot
from apyrobo.core.schemas import CapabilityType


class TestRobotDiscovery:
    """Robot.discover() with mock:// URIs."""

    def test_discover_mock(self) -> None:
        robot = Robot.discover("mock://test_bot")
        assert robot.robot_id == "test_bot"
        assert isinstance(robot._adapter, MockAdapter)

    def test_discover_bad_uri(self) -> None:
        with pytest.raises(ValueError, match="Invalid robot URI"):
            Robot.discover("noscheme")

    def test_discover_unknown_scheme(self) -> None:
        with pytest.raises(ValueError, match="No adapter registered"):
            Robot.discover("unknown://bot")

    def test_repr(self) -> None:
        robot = Robot.discover("mock://r1")
        assert "MockAdapter" in repr(robot)


class TestRobotCapabilities:
    """Robot.capabilities() via mock adapter."""

    def test_capabilities_returned(self) -> None:
        robot = Robot.discover("mock://tb4")
        caps = robot.capabilities()
        assert caps.robot_id == "tb4"
        assert len(caps.capabilities) > 0
        cap_types = {c.capability_type for c in caps.capabilities}
        assert CapabilityType.NAVIGATE in cap_types

    def test_capabilities_cached(self) -> None:
        robot = Robot.discover("mock://tb4")
        c1 = robot.capabilities()
        c2 = robot.capabilities()
        assert c1 is c2  # same object, cached

    def test_capabilities_refresh(self) -> None:
        robot = Robot.discover("mock://tb4")
        c1 = robot.capabilities()
        c2 = robot.capabilities(refresh=True)
        assert c1 is not c2  # new object


class TestRobotCommands:
    """Robot.move() and .stop() via mock adapter."""

    def test_move(self) -> None:
        robot = Robot.discover("mock://tb4")
        robot.move(x=2.0, y=3.0, speed=0.5)
        adapter: MockAdapter = robot._adapter  # type: ignore[assignment]
        assert adapter.position == (2.0, 3.0)
        assert adapter.is_moving is True
        assert len(adapter.move_history) == 1

    def test_stop(self) -> None:
        robot = Robot.discover("mock://tb4")
        robot.move(x=1.0, y=1.0)
        robot.stop()
        adapter: MockAdapter = robot._adapter  # type: ignore[assignment]
        assert adapter.is_moving is False

    def test_multiple_moves(self) -> None:
        robot = Robot.discover("mock://tb4")
        robot.move(x=1.0, y=0.0)
        robot.move(x=2.0, y=0.0)
        robot.move(x=3.0, y=0.0)
        adapter: MockAdapter = robot._adapter  # type: ignore[assignment]
        assert len(adapter.move_history) == 3
        assert adapter.position == (3.0, 0.0)

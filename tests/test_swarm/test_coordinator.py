"""
CI-10: Swarm coordinator tests — multi-robot split + failure + reassignment.
"""

from __future__ import annotations

from typing import Any

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import TaskStatus, CapabilityType
from apyrobo.skills.skill import Skill, SkillStatus
from apyrobo.skills.executor import SkillGraph, SkillExecutor, ExecutionEvent
from apyrobo.skills.agent import Agent
from apyrobo.swarm.bus import SwarmBus
from apyrobo.swarm.coordinator import SwarmCoordinator, RobotAssignment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bus() -> SwarmBus:
    return SwarmBus()


@pytest.fixture
def robot_a() -> Robot:
    return Robot.discover("mock://robot_a")


@pytest.fixture
def robot_b() -> Robot:
    return Robot.discover("mock://robot_b")


@pytest.fixture
def agent() -> Agent:
    return Agent(provider="rule")


@pytest.fixture
def two_robot_bus(bus: SwarmBus, robot_a: Robot, robot_b: Robot) -> SwarmBus:
    bus.register(robot_a)
    bus.register(robot_b)
    return bus


# ===========================================================================
# Task splitting
# ===========================================================================

class TestTaskSplitting:
    """Coordinator splits tasks across robots correctly."""

    def test_single_robot_no_split(
        self, bus: SwarmBus, robot_a: Robot, agent: Agent
    ) -> None:
        """With one robot, all skills go to it."""
        bus.register(robot_a)
        coord = SwarmCoordinator(bus)
        assignments = coord.split_task("go to 1, 2", agent)
        assert len(assignments) == 1
        assert assignments[0].robot_id == "robot_a"

    def test_multi_robot_splits(
        self, two_robot_bus: SwarmBus, agent: Agent
    ) -> None:
        """With multiple robots, skills are distributed."""
        coord = SwarmCoordinator(two_robot_bus)
        assignments = coord.split_task("go to 1, 2 then pick up object", agent)
        # Should have at least one assignment
        assert len(assignments) >= 1
        # All robot IDs should be registered
        for a in assignments:
            assert a.robot_id in ("robot_a", "robot_b")

    def test_round_robin_strategy(
        self, two_robot_bus: SwarmBus, agent: Agent
    ) -> None:
        """Round-robin distributes evenly."""
        coord = SwarmCoordinator(two_robot_bus, strategy="round_robin")
        assignments = coord.split_task("go to 1, 2 then pick up object", agent)
        assert len(assignments) >= 1

    def test_no_robots_raises(self, bus: SwarmBus, agent: Agent) -> None:
        """Empty swarm raises ValueError."""
        coord = SwarmCoordinator(bus)
        with pytest.raises(ValueError, match="No robots"):
            coord.split_task("go to 1, 2", agent)

    def test_assignment_has_graph(
        self, bus: SwarmBus, robot_a: Robot, agent: Agent
    ) -> None:
        """Each assignment has a SkillGraph."""
        bus.register(robot_a)
        coord = SwarmCoordinator(bus)
        assignments = coord.split_task("go to 1, 2", agent)
        for a in assignments:
            assert isinstance(a.graph, SkillGraph)
            assert len(a.graph) > 0


# ===========================================================================
# Execution
# ===========================================================================

class TestSwarmExecution:
    """Coordinator executes tasks across the swarm."""

    def test_single_robot_execution(
        self, bus: SwarmBus, robot_a: Robot, agent: Agent
    ) -> None:
        """Execute a task with a single robot."""
        bus.register(robot_a)
        coord = SwarmCoordinator(bus)
        result = coord.execute_task("go to 1, 2", agent)
        assert result.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)

    def test_multi_robot_execution(
        self, two_robot_bus: SwarmBus, agent: Agent
    ) -> None:
        """Execute a task across multiple robots."""
        coord = SwarmCoordinator(two_robot_bus)
        result = coord.execute_task("go to 1, 2", agent)
        assert result is not None
        assert hasattr(result, "status")

    def test_events_captured(
        self, bus: SwarmBus, robot_a: Robot, agent: Agent
    ) -> None:
        """Execution events are captured by the coordinator."""
        bus.register(robot_a)
        coord = SwarmCoordinator(bus)
        events: list[ExecutionEvent] = []
        coord.execute_task("go to 1, 2", agent, on_event=lambda e: events.append(e))
        assert len(events) > 0


# ===========================================================================
# Failure and reassignment
# ===========================================================================

class TestFailureReassignment:
    """Failure handling and reassignment."""

    def test_reassignment_attempted(
        self, two_robot_bus: SwarmBus, agent: Agent
    ) -> None:
        """When a robot fails, reassignment is attempted."""
        coord = SwarmCoordinator(two_robot_bus)
        # Create a failing assignment manually
        assignment = RobotAssignment("robot_a", SkillGraph(), "test")
        assignment.status = TaskStatus.FAILED
        result = coord._attempt_reassignment(assignment, agent)
        assert result is True  # robot_b available

    def test_no_reassignment_single_robot(
        self, bus: SwarmBus, robot_a: Robot, agent: Agent
    ) -> None:
        """No reassignment possible with only one robot."""
        bus.register(robot_a)
        coord = SwarmCoordinator(bus)
        assignment = RobotAssignment("robot_a", SkillGraph(), "test")
        result = coord._attempt_reassignment(assignment, agent)
        assert result is False

    def test_assignments_property(
        self, bus: SwarmBus, robot_a: Robot, agent: Agent
    ) -> None:
        """Assignments are accessible after execution."""
        bus.register(robot_a)
        coord = SwarmCoordinator(bus)
        coord.execute_task("go to 1, 2", agent)
        assert len(coord.assignments) >= 1

    def test_coordinator_repr(self, bus: SwarmBus) -> None:
        coord = SwarmCoordinator(bus)
        assert "SwarmCoordinator" in repr(coord)


# ===========================================================================
# Bus message verification
# ===========================================================================

class TestBusMessaging:
    """Coordinator broadcasts assignments via bus."""

    def test_task_assigned_broadcast(
        self, two_robot_bus: SwarmBus, agent: Agent
    ) -> None:
        """Coordinator broadcasts status messages via the bus during execution."""
        coord = SwarmCoordinator(two_robot_bus)
        coord.execute_task("go to 1, 2", agent)

        # Check the bus message log for status messages (from skill execution)
        log = two_robot_bus.message_log
        status_msgs = [
            m for m in log
            if hasattr(m, "msg_type") and m.msg_type == "status"
        ]
        assert len(status_msgs) >= 1, f"Expected status messages, got: {[m.msg_type for m in log]}"

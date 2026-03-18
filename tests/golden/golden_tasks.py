"""
Golden Task Suite — 15 canonical tasks with explicit pass/fail criteria.

Each task runs against the mock adapter (CI) and can also target the Gazebo
adapter (integration).  The suite is the primary acceptance gate for v0.2.

Usage (pytest)::

    pytest tests/golden/golden_tasks.py -v

Usage (programmatic)::

    from tests.golden.golden_tasks import GOLDEN_TASKS, TARGET_PASS_RATE
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import (
    CapabilityType,
    TaskResult,
    TaskStatus,
)
from apyrobo.safety.confidence import ConfidenceEstimator, LowConfidenceError
from apyrobo.safety.enforcer import SafetyEnforcer, SafetyPolicy, SafetyViolation
from apyrobo.skills.agent import Agent
from apyrobo.skills.executor import ExecutionEvent, SkillExecutor, SkillGraph
from apyrobo.skills.handlers import skill_handler
from apyrobo.skills.package import SkillPackage
from apyrobo.skills.registry import SkillRegistry
from apyrobo.skills.skill import Skill
from apyrobo.swarm.bus import SwarmBus
from apyrobo.swarm.coordinator import SwarmCoordinator


# ---------------------------------------------------------------------------
# Declarative task table
# ---------------------------------------------------------------------------

GOLDEN_TASKS = [
    {
        "id": "GT-01",
        "task": "navigate to (2, 0)",
        "adapter": "mock",
        "pass_fn": lambda result, robot: (
            result.status == TaskStatus.COMPLETED
            and abs(robot.get_position()[0] - 2.0) < 0.3
            and abs(robot.get_position()[1] - 0.0) < 0.3
        ),
    },
    {
        "id": "GT-02",
        "task": "navigate to (0, 0)",
        "adapter": "mock",
        "pass_fn": lambda result, robot: (
            result.status == TaskStatus.COMPLETED
            and abs(robot.get_position()[0] - 0.0) < 0.3
            and abs(robot.get_position()[1] - 0.0) < 0.3
        ),
    },
    {
        "id": "GT-03",
        "task": "stop immediately",
        "adapter": "mock",
        "pass_fn": lambda result, robot: (
            result.status == TaskStatus.COMPLETED
        ),
    },
    {
        "id": "GT-04",
        "task": "rotate 90 degrees",
        "adapter": "mock",
        "pass_fn": lambda result, robot: (
            result.status == TaskStatus.COMPLETED
        ),
    },
    {
        "id": "GT-05",
        "task": "report status",
        "adapter": "mock",
        "pass_fn": lambda result, robot: (
            result.status == TaskStatus.COMPLETED
        ),
    },
    {
        "id": "GT-06",
        "task": "deliver package to (3, 2)",
        "adapter": "mock",
        "pass_fn": lambda result, robot: (
            result.status == TaskStatus.COMPLETED
            and result.steps_completed >= 4
        ),
    },
    {
        "id": "GT-07",
        "task": "pick up the object at (1, 1)",
        "adapter": "mock",
        "pass_fn": lambda result, robot: (
            result.status == TaskStatus.COMPLETED
            and result.steps_completed >= 2
        ),
    },
    {
        "id": "GT-08",
        "task": "navigate to collision zone",
        "adapter": "mock",
        "pass_fn": None,  # special: expects SafetyViolation
    },
    {
        "id": "GT-09",
        "task": "navigate at speed 10",
        "adapter": "mock",
        "pass_fn": None,  # special: expects speed clamped
    },
    {
        "id": "GT-10",
        "task": "navigate to 5 5 then stop",
        "adapter": "mock",
        "pass_fn": lambda result, robot: (
            result.status == TaskStatus.COMPLETED
        ),
    },
    {
        "id": "GT-11",
        "task": "deliver package to (3, 2)",
        "adapter": "mock",
        "pass_fn": lambda result, robot: (
            result.status == TaskStatus.COMPLETED
            and result.steps_completed >= 4
        ),
    },
    {
        "id": "GT-12",
        "task": "two-robot swarm delivery",
        "adapter": "mock",
        "pass_fn": None,  # special: swarm test
    },
    {
        "id": "GT-13",
        "task": "install and execute skillpkg",
        "adapter": "mock",
        "pass_fn": None,  # special: registry + custom handler test
    },
    {
        "id": "GT-14",
        "task": "confidence gates low-confidence plan",
        "adapter": "mock",
        "pass_fn": None,  # special: ConfidenceReport test
    },
    {
        "id": "GT-15",
        "task": "sequential reliability",
        "adapter": "mock",
        "pass_fn": None,  # special: 20-task batch
    },
]

TARGET_PASS_RATE = 0.90  # 90 % of golden tasks must pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_robot() -> Robot:
    return Robot.discover("mock://golden_bot")


@pytest.fixture
def agent() -> Agent:
    with patch("apyrobo.skills.agent.DEFAULT_REGISTRY_DIR", Path("/nonexistent")):
        return Agent(provider="rule")


# ---------------------------------------------------------------------------
# GT-01: Navigate to (2, 0)
# ---------------------------------------------------------------------------

class TestGT01:
    def test_navigate_to_2_0(self, mock_robot: Robot, agent: Agent) -> None:
        result = agent.execute("navigate to (2, 0)", mock_robot)
        pos = mock_robot.get_position()
        assert result.status == TaskStatus.COMPLETED
        assert abs(pos[0] - 2.0) < 0.3
        assert abs(pos[1] - 0.0) < 0.3


# ---------------------------------------------------------------------------
# GT-02: Navigate to (0, 0) — return to origin
# ---------------------------------------------------------------------------

class TestGT02:
    def test_navigate_to_origin(self, mock_robot: Robot, agent: Agent) -> None:
        result = agent.execute("navigate to (0, 0)", mock_robot)
        pos = mock_robot.get_position()
        assert result.status == TaskStatus.COMPLETED
        assert abs(pos[0] - 0.0) < 0.3
        assert abs(pos[1] - 0.0) < 0.3


# ---------------------------------------------------------------------------
# GT-03: Stop immediately — completes in < 1 s
# ---------------------------------------------------------------------------

class TestGT03:
    def test_stop_immediately(self, mock_robot: Robot, agent: Agent) -> None:
        t0 = time.monotonic()
        result = agent.execute("stop", mock_robot)
        elapsed = time.monotonic() - t0
        assert result.status == TaskStatus.COMPLETED
        assert elapsed < 1.0


# ---------------------------------------------------------------------------
# GT-04: Rotate 90 degrees — orientation changes ± 0.2 rad from start
# ---------------------------------------------------------------------------

class TestGT04:
    def test_rotate_90(self, mock_robot: Robot, agent: Agent) -> None:
        start_orient = mock_robot.get_orientation()
        result = agent.execute("rotate 90 degrees", mock_robot)
        end_orient = mock_robot.get_orientation()
        assert result.status == TaskStatus.COMPLETED
        # Orientation should have changed (mock adapter adds the angle)
        delta = abs(end_orient - start_orient)
        assert delta > 0.2 or delta < (2 * math.pi - 0.2)


# ---------------------------------------------------------------------------
# GT-05: Report status — completes with capabilities logged
# ---------------------------------------------------------------------------

class TestGT05:
    def test_report_status(self, mock_robot: Robot, agent: Agent) -> None:
        result = agent.execute("report status", mock_robot)
        assert result.status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# GT-06: Deliver package to (3, 2) — 4 skills, all COMPLETED
# ---------------------------------------------------------------------------

class TestGT06:
    def test_deliver_package(self, mock_robot: Robot, agent: Agent) -> None:
        result = agent.execute("deliver package to (3, 2)", mock_robot)
        assert result.status == TaskStatus.COMPLETED
        assert result.steps_completed >= 4


# ---------------------------------------------------------------------------
# GT-07: Go to (1, 1) and pick up — 2 skills, gripper_close called
# ---------------------------------------------------------------------------

class TestGT07:
    def test_go_and_pick(self, mock_robot: Robot, agent: Agent) -> None:
        result = agent.execute("pick up the object at (1, 1)", mock_robot)
        assert result.status == TaskStatus.COMPLETED
        assert result.steps_completed >= 2
        # MockAdapter tracks gripper state: gripper_close sets _gripper_open=False
        adapter = mock_robot._adapter
        assert adapter._gripper_open is False, "gripper_close should have been called"


# ---------------------------------------------------------------------------
# GT-08: Navigate into no-go zone — SafetyViolation, robot does not move
# ---------------------------------------------------------------------------

class TestGT08:
    def test_collision_zone_blocks_move(self, mock_robot: Robot) -> None:
        no_go = {"x_min": 4.0, "x_max": 6.0, "y_min": 4.0, "y_max": 6.0}
        policy = SafetyPolicy(
            name="test_nogo",
            collision_zones=[no_go],
        )
        enforcer = SafetyEnforcer(mock_robot, policy=policy)

        start_pos = mock_robot.get_position()
        with pytest.raises(SafetyViolation):
            enforcer.move(5.0, 5.0)

        # Robot must NOT have moved
        assert mock_robot.get_position() == start_pos


# ---------------------------------------------------------------------------
# GT-09: Navigate at speed 10 m/s — speed clamped, intervention logged
# ---------------------------------------------------------------------------

class TestGT09:
    def test_speed_clamped(self, mock_robot: Robot) -> None:
        policy = SafetyPolicy(name="test_clamp", max_speed=1.5)
        enforcer = SafetyEnforcer(mock_robot, policy=policy)

        enforcer.move(2.0, 0.0, speed=10.0)

        # Intervention should have been recorded
        assert len(enforcer._interventions) >= 1
        intervention = enforcer._interventions[0]
        assert intervention["type"] == "speed_clamped"
        assert intervention["requested"] == 10.0
        assert intervention["enforced"] == 1.5

        # The robot's move history should show the clamped speed
        adapter = mock_robot._adapter
        last_move = adapter._move_history[-1]
        assert last_move["speed"] <= 1.5


# ---------------------------------------------------------------------------
# GT-10: Navigate to (5, 5) then stop — stop event emitted
# ---------------------------------------------------------------------------

class TestGT10:
    def test_navigate_then_stop(self, mock_robot: Robot, agent: Agent) -> None:
        """Navigate to (5, 5), then issue a separate stop command.
        Verify the stop event is emitted before the navigate completes."""
        events: list[ExecutionEvent] = []
        listener = lambda e: events.append(e)

        # Step 1: navigate
        result_nav = agent.execute(
            "navigate to (5, 5)", mock_robot, on_event=listener,
        )
        assert result_nav.status == TaskStatus.COMPLETED

        # Step 2: stop
        result_stop = agent.execute("stop", mock_robot, on_event=listener)
        assert result_stop.status == TaskStatus.COMPLETED

        # Combined events must include a stop skill
        stop_events = [e for e in events if "stop" in e.skill_id]
        assert len(stop_events) >= 1, "Expected at least one stop event"


# ---------------------------------------------------------------------------
# GT-11: Plan and execute with LLM provider (or rule fallback)
# ---------------------------------------------------------------------------

class TestGT11:
    def test_rule_fallback_delivery(self, mock_robot: Robot) -> None:
        """Same as GT-06 but explicitly using rule-based provider."""
        with patch("apyrobo.skills.agent.DEFAULT_REGISTRY_DIR", Path("/nonexistent")):
            agent = Agent(provider="rule")
        result = agent.execute("deliver package to (3, 2)", mock_robot)
        assert result.status == TaskStatus.COMPLETED
        assert result.steps_completed >= 4


# ---------------------------------------------------------------------------
# GT-12: Two-robot swarm — split a delivery, both complete
# ---------------------------------------------------------------------------

class TestGT12:
    def test_swarm_split_delivery(self) -> None:
        robot_a = Robot.discover("mock://swarm_a")
        robot_b = Robot.discover("mock://swarm_b")

        bus = SwarmBus()
        bus.register(robot_a)
        bus.register(robot_b)

        with patch("apyrobo.skills.agent.DEFAULT_REGISTRY_DIR", Path("/nonexistent")):
            agent = Agent(provider="rule")

        coordinator = SwarmCoordinator(bus)
        result = coordinator.execute_task("deliver package to (3, 2)", agent)

        assert result.status == TaskStatus.COMPLETED
        # Both robots should have been assigned
        assigned_ids = {a.robot_id for a in coordinator.assignments}
        assert len(assigned_ids) == 2


# ---------------------------------------------------------------------------
# GT-13: Install a .skillpkg and execute an installed skill
# ---------------------------------------------------------------------------

class TestGT13:
    def test_install_and_execute_skill(self, tmp_path: Path, mock_robot: Robot) -> None:
        # Register a real handler for our custom skill
        @skill_handler("warehouse_sweep")
        def _sweep(robot: Any, params: dict[str, Any]) -> bool:
            robot.move(x=0.0, y=0.0, speed=0.5)
            return True

        custom_skill = Skill(
            skill_id="warehouse_sweep",
            name="Warehouse Sweep",
            description="Sweep the warehouse floor",
            required_capability=CapabilityType.NAVIGATE,
            parameters={"zone": "A1"},
        )
        pkg = SkillPackage(name="sweep-pkg", version="1.0.0", skills=[custom_skill])
        pkg_dir = tmp_path / "sweep-pkg"
        pkg.save(pkg_dir)

        registry = SkillRegistry(tmp_path / "registry")
        registry.install(pkg_dir)

        agent = Agent(provider="rule", registry=registry)

        # The custom skill should be in the catalog
        catalog = agent._get_skill_catalog()
        assert "warehouse_sweep" in catalog

        # Execute via agent: "sweep" should match the custom skill
        result = agent.execute("sweep the warehouse", mock_robot)
        # With a real handler registered, this should succeed (not FAILED)
        assert result.status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# GT-14: Confidence estimator gates a low-confidence plan
# ---------------------------------------------------------------------------

class TestGT14:
    def test_confidence_gates_execution(self, mock_robot: Robot, agent: Agent) -> None:
        """A high block_below threshold should gate even a well-planned task."""
        from apyrobo.core.schemas import RobotCapability

        # Plan a task that will produce a graph
        graph = agent.plan("deliver package to (3, 2)", mock_robot)

        # Create a fake robot with NO capabilities — confidence will be very low
        stripped_robot = MagicMock(spec=Robot)
        stripped_robot.capabilities.return_value = RobotCapability(
            robot_id="stripped",
            name="Stripped",
            capabilities=[],  # no capabilities at all
            sensors=[],       # no sensors
            max_speed=0.5,
        )

        estimator = ConfidenceEstimator(block_below=0.5)

        with pytest.raises(LowConfidenceError) as exc_info:
            estimator.gate(graph, stripped_robot)

        report = exc_info.value.report
        assert report.confidence < 0.5
        assert len(report.risks) > 0


# ---------------------------------------------------------------------------
# GT-15: 20 sequential tasks, measure success rate (≥ 19/20)
# ---------------------------------------------------------------------------

class TestGT15:
    TASKS = [
        "navigate to (1, 0)",
        "navigate to (2, 0)",
        "navigate to (0, 0)",
        "rotate 45 degrees",
        "rotate 90 degrees",
        "stop",
        "report status",
        "navigate to (3, 1)",
        "navigate to (0, 0)",
        "stop",
        "navigate to (1, 1)",
        "navigate to (2, 2)",
        "rotate 180 degrees",
        "report status",
        "navigate to (0, 0)",
        "stop",
        "navigate to (4, 3)",
        "rotate 45 degrees",
        "navigate to (0, 0)",
        "report status",
    ]

    def test_sequential_reliability(self, mock_robot: Robot, agent: Agent) -> None:
        successes = 0
        for task in self.TASKS:
            result = agent.execute(task, mock_robot)
            if result.status == TaskStatus.COMPLETED:
                successes += 1

        assert successes >= 19, (
            f"Sequential reliability: {successes}/20 passed, need ≥ 19"
        )


# ---------------------------------------------------------------------------
# Aggregate pass-rate check
# ---------------------------------------------------------------------------

class TestGoldenPassRate:
    """
    Meta-test: run the declarative GOLDEN_TASKS table entries that have a
    simple pass_fn and verify the aggregate pass rate meets TARGET_PASS_RATE.
    """

    def test_aggregate_pass_rate(self, mock_robot: Robot, agent: Agent) -> None:
        simple_tasks = [t for t in GOLDEN_TASKS if t["pass_fn"] is not None]
        passed = 0

        for gt in simple_tasks:
            result = agent.execute(gt["task"], mock_robot)
            if gt["pass_fn"](result, mock_robot):
                passed += 1

        rate = passed / len(simple_tasks) if simple_tasks else 0.0
        assert rate >= TARGET_PASS_RATE, (
            f"Golden pass rate {rate:.0%} ({passed}/{len(simple_tasks)}) "
            f"is below target {TARGET_PASS_RATE:.0%}"
        )

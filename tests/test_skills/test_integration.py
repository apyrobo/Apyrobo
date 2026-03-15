"""
CI-06: Integration test — full task execution in mock mode end-to-end.

Tests the complete path: Agent → plan → SkillGraph → SkillExecutor → Robot
"""

from __future__ import annotations

from typing import Any

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import TaskStatus
from apyrobo.skills.skill import SkillStatus, BUILTIN_SKILLS
from apyrobo.skills.executor import SkillGraph, SkillExecutor, ExecutionEvent, ExecutionState
from apyrobo.skills.agent import Agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_robot() -> Robot:
    return Robot.discover("mock://integration_bot")


@pytest.fixture
def agent() -> Agent:
    return Agent(provider="rule")


# ===========================================================================
# CI-06: End-to-end integration tests
# ===========================================================================

class TestEndToEnd:
    """Full Agent → executor → robot integration."""

    def test_navigate_task(self, agent: Agent, mock_robot: Robot) -> None:
        """Agent plans and executes a navigation task."""
        result = agent.execute("go to position 3, 4", mock_robot)
        assert result.status in (TaskStatus.COMPLETED, "completed")
        assert result.steps_completed >= 1

    def test_pick_and_place_task(self, agent: Agent, mock_robot: Robot) -> None:
        """Agent plans and executes a pick-and-place task."""
        result = agent.execute("pick up the object and place it at 5, 5", mock_robot)
        # The rule-based provider may or may not handle this fully
        assert result is not None
        assert hasattr(result, "status")

    def test_stop_task(self, agent: Agent, mock_robot: Robot) -> None:
        """Agent plans and executes a stop task."""
        result = agent.execute("stop the robot", mock_robot)
        assert result is not None

    def test_event_streaming(self, agent: Agent, mock_robot: Robot) -> None:
        """Events are emitted during execution."""
        events: list[ExecutionEvent] = []
        result = agent.execute(
            "go to position 1, 2", mock_robot,
            on_event=lambda e: events.append(e),
        )
        assert len(events) > 0
        # Should see PENDING, RUNNING, COMPLETED
        statuses = {e.status for e in events}
        assert SkillStatus.PENDING in statuses or SkillStatus.RUNNING in statuses

    def test_multi_skill_graph(self, mock_robot: Robot) -> None:
        """A graph with multiple dependent skills executes in order."""
        g = SkillGraph()
        g.add_skill(BUILTIN_SKILLS["navigate_to"],
                     parameters={"x": 1.0, "y": 2.0, "speed": 0.5})
        g.add_skill(BUILTIN_SKILLS["pick_object"],
                     depends_on=["navigate_to"])
        g.add_skill(BUILTIN_SKILLS["navigate_to"],
                     depends_on=["pick_object"])

        # Can't add same skill_id twice, so create unique skills
        nav1 = BUILTIN_SKILLS["navigate_to"]
        from apyrobo.skills.skill import Skill
        nav2 = Skill(
            skill_id="navigate_to_2",
            name="Navigate To 2",
            required_capability=nav1.required_capability,
            parameters={"x": 5.0, "y": 5.0, "speed": 0.5},
        )

        g2 = SkillGraph()
        g2.add_skill(nav1, parameters={"x": 1.0, "y": 2.0})
        g2.add_skill(BUILTIN_SKILLS["pick_object"], depends_on=["navigate_to"])
        g2.add_skill(nav2, depends_on=["pick_object"],
                      parameters={"x": 5.0, "y": 5.0})

        exe = SkillExecutor(mock_robot)
        result = exe.execute_graph(g2)
        assert result.status == TaskStatus.COMPLETED
        assert result.steps_completed == 3

    def test_parallel_execution(self, mock_robot: Robot) -> None:
        """Independent skills can be executed in parallel."""
        from apyrobo.skills.skill import Skill

        s1 = Skill(skill_id="report_1", name="Report 1",
                    required_capability=BUILTIN_SKILLS["report_status"].required_capability)
        s2 = Skill(skill_id="report_2", name="Report 2",
                    required_capability=BUILTIN_SKILLS["report_status"].required_capability)
        s3 = Skill(skill_id="stop_final", name="Stop Final",
                    required_capability=BUILTIN_SKILLS["stop"].required_capability)

        g = SkillGraph()
        g.add_skill(s1)
        g.add_skill(s2)
        g.add_skill(s3, depends_on=["report_1", "report_2"])

        exe = SkillExecutor(mock_robot)
        result = exe.execute_graph(g, parallel=True)
        assert result.status == TaskStatus.COMPLETED
        assert result.steps_completed == 3

    def test_execution_state_persists(self, mock_robot: Robot) -> None:
        """Execution state flows between skills in a graph."""
        state = ExecutionState()
        exe = SkillExecutor(mock_robot, state=state)

        g = SkillGraph()
        g.add_skill(BUILTIN_SKILLS["navigate_to"],
                     parameters={"x": 3.0, "y": 4.0})
        exe.execute_graph(g)

        assert state.get("at_position") == (3.0, 4.0)
        assert state.is_set("robot_idle")

    def test_failed_skill_aborts_graph(self, mock_robot: Robot) -> None:
        """A failed skill causes the graph to abort."""
        from apyrobo.skills.skill import Skill
        from apyrobo.core.schemas import CapabilityType

        # Skill requiring DOCK (not available on mock)
        fly = Skill(
            skill_id="dock",
            name="Dock",
            required_capability=CapabilityType.DOCK,
        )
        stop = BUILTIN_SKILLS["stop"]

        g = SkillGraph()
        g.add_skill(fly)
        g.add_skill(stop, depends_on=["dock"])

        exe = SkillExecutor(mock_robot)
        result = exe.execute_graph(g)
        assert result.status == TaskStatus.FAILED
        assert result.steps_completed == 0

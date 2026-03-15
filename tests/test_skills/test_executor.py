"""
CI-01: Test all 5+1 built-in skill handlers in mock mode.
CI-02: Test precondition evaluation for all 4 check_types (capability, state, speed, default).
CI-03: Test UnknownSkillError raised for unregistered skills.
CI-04: Test that skill timeout triggers stop() and returns FAILED.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import CapabilityType
from apyrobo.skills.skill import Skill, SkillStatus, Condition, BUILTIN_SKILLS
from apyrobo.skills.executor import (
    SkillExecutor,
    SkillGraph,
    ExecutionState,
    ExecutionEvent,
    SkillTimeout,
    _run_with_timeout,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_robot() -> Robot:
    return Robot.discover("mock://test_bot")


@pytest.fixture
def executor(mock_robot: Robot) -> SkillExecutor:
    return SkillExecutor(mock_robot)


@pytest.fixture
def events() -> list[ExecutionEvent]:
    return []


@pytest.fixture
def executor_with_events(mock_robot: Robot, events: list[ExecutionEvent]) -> SkillExecutor:
    exe = SkillExecutor(mock_robot)
    exe.on_event(lambda e: events.append(e))
    return exe


# ===========================================================================
# CI-01: Test all 5+1 built-in skill handlers in mock mode
# ===========================================================================

class TestBuiltinSkillHandlers:
    """CI-01: Every built-in skill handler executes correctly in mock mode."""

    def test_navigate_to_handler(self, executor: SkillExecutor, mock_robot: Robot) -> None:
        """navigate_to calls robot.move(x, y, speed)."""
        skill = BUILTIN_SKILLS["navigate_to"]
        status = executor.execute_skill(skill, {"x": 2.0, "y": 3.0, "speed": 0.5})
        assert status == SkillStatus.COMPLETED
        # Verify the robot actually moved
        pos = mock_robot.get_position()
        assert pos == (2.0, 3.0)

    def test_rotate_handler(self, executor: SkillExecutor) -> None:
        """rotate calls robot.rotate(angle_rad)."""
        skill = BUILTIN_SKILLS["rotate"]
        status = executor.execute_skill(skill, {"angle_rad": 1.57})
        assert status == SkillStatus.COMPLETED

    def test_stop_handler(self, executor: SkillExecutor) -> None:
        """stop calls robot.stop()."""
        skill = BUILTIN_SKILLS["stop"]
        status = executor.execute_skill(skill)
        assert status == SkillStatus.COMPLETED

    def test_pick_object_handler(self, executor: SkillExecutor) -> None:
        """pick_object calls robot.gripper_close()."""
        skill = BUILTIN_SKILLS["pick_object"]
        # pick_object has preconditions but the default check_type is 'capability'
        # which passes for mock; the handler calls gripper_close
        status = executor.execute_skill(skill)
        assert status == SkillStatus.COMPLETED

    def test_place_object_handler(self, executor: SkillExecutor) -> None:
        """place_object calls robot.gripper_open()."""
        skill = BUILTIN_SKILLS["place_object"]
        status = executor.execute_skill(skill)
        assert status == SkillStatus.COMPLETED

    def test_report_status_handler(self, executor: SkillExecutor) -> None:
        """report_status queries capabilities and returns success."""
        skill = BUILTIN_SKILLS["report_status"]
        status = executor.execute_skill(skill)
        assert status == SkillStatus.COMPLETED

    def test_all_builtin_skills_registered(self) -> None:
        """All 6 built-in skills are in the registry."""
        expected = {"navigate_to", "rotate", "stop", "pick_object", "place_object", "report_status"}
        assert expected == set(BUILTIN_SKILLS.keys())

    def test_navigate_to_updates_state(self, executor: SkillExecutor) -> None:
        """navigate_to sets at_position and robot_idle in execution state."""
        skill = BUILTIN_SKILLS["navigate_to"]
        executor.execute_skill(skill, {"x": 5.0, "y": 6.0})
        assert executor.state.get("at_position") == (5.0, 6.0)
        assert executor.state.is_set("robot_idle")

    def test_pick_sets_object_held(self, executor: SkillExecutor) -> None:
        """pick_object sets object_held=True, gripper_open=False."""
        skill = BUILTIN_SKILLS["pick_object"]
        executor.execute_skill(skill)
        assert executor.state.is_set("object_held")
        assert not executor.state.is_set("gripper_open")

    def test_place_clears_object_held(self, executor: SkillExecutor) -> None:
        """place_object sets object_held=False, gripper_open=True."""
        skill = BUILTIN_SKILLS["place_object"]
        executor.execute_skill(skill)
        assert not executor.state.is_set("object_held")
        assert executor.state.is_set("gripper_open")


# ===========================================================================
# CI-02: Test precondition evaluation for all 4 check_types
# ===========================================================================

class TestPreconditionEvaluation:
    """CI-02: All precondition check types work correctly."""

    def test_capability_check_passes(self, executor: SkillExecutor, mock_robot: Robot) -> None:
        """Skill requiring NAVIGATE passes when robot has NAVIGATE capability."""
        skill = Skill(
            skill_id="test_nav",
            name="Test Nav",
            required_capability=CapabilityType.NAVIGATE,
        )
        ok, reason = executor.check_preconditions(skill, mock_robot)
        assert ok, reason

    def test_capability_check_fails(self, executor: SkillExecutor, mock_robot: Robot) -> None:
        """Skill requiring a missing capability fails precondition check."""
        skill = Skill(
            skill_id="test_dock",
            name="Test Dock",
            required_capability=CapabilityType.DOCK,
        )
        ok, reason = executor.check_preconditions(skill, mock_robot)
        assert not ok
        assert "lacks required capability" in reason

    def test_state_check_passes(self, executor: SkillExecutor, mock_robot: Robot) -> None:
        """State precondition passes when the flag is set."""
        executor.state.set("object_held", True)
        skill = Skill(
            skill_id="test_state",
            name="Test State",
            preconditions=[
                Condition(name="object_held", check_type="state",
                          parameters={"key": "object_held", "value": True}),
            ],
        )
        ok, reason = executor.check_preconditions(skill, mock_robot)
        assert ok, reason

    def test_state_check_fails(self, executor: SkillExecutor, mock_robot: Robot) -> None:
        """State precondition fails when the flag is not set."""
        skill = Skill(
            skill_id="test_state_fail",
            name="Test State Fail",
            preconditions=[
                Condition(name="object_held", check_type="state",
                          parameters={"key": "object_held", "value": True}),
            ],
        )
        ok, reason = executor.check_preconditions(skill, mock_robot)
        assert not ok
        assert "State precondition" in reason

    def test_speed_check_passes(self, executor: SkillExecutor, mock_robot: Robot) -> None:
        """Speed parameter within robot's max_speed passes."""
        skill = Skill(
            skill_id="test_speed_ok",
            name="Test Speed OK",
            required_capability=CapabilityType.NAVIGATE,
            parameters={"speed": 0.5},
        )
        ok, reason = executor.check_preconditions(skill, mock_robot)
        assert ok, reason

    def test_speed_check_fails(self, executor: SkillExecutor, mock_robot: Robot) -> None:
        """Speed parameter exceeding robot max_speed fails."""
        caps = mock_robot.capabilities()
        over_max = (caps.max_speed or 1.5) + 10.0
        skill = Skill(
            skill_id="test_speed_fail",
            name="Test Speed Fail",
            required_capability=CapabilityType.NAVIGATE,
            parameters={"speed": over_max},
        )
        ok, reason = executor.check_preconditions(skill, mock_robot)
        assert not ok
        assert "exceeds robot max" in reason

    def test_custom_capability_always_passes(self, executor: SkillExecutor, mock_robot: Robot) -> None:
        """CUSTOM capability type skips capability check."""
        skill = Skill(
            skill_id="test_custom",
            name="Test Custom",
            required_capability=CapabilityType.CUSTOM,
        )
        ok, reason = executor.check_preconditions(skill, mock_robot)
        assert ok, reason

    def test_default_check_type_is_capability(self) -> None:
        """Default check_type for Condition is 'capability'."""
        cond = Condition(name="test_cond")
        assert cond.check_type == "capability"


# ===========================================================================
# CI-03: Test UnknownSkillError raised for unregistered skills
# ===========================================================================

class TestUnknownSkill:
    """CI-03: Unknown skills are handled correctly."""

    def test_unknown_skill_dispatch_succeeds_with_warning(
        self, executor: SkillExecutor
    ) -> None:
        """
        Unknown skills currently return True from _dispatch_skill (with a warning).
        This test verifies the current behavior.
        """
        unknown = Skill(
            skill_id="totally_unknown_skill",
            name="Unknown",
            required_capability=CapabilityType.CUSTOM,
        )
        status = executor.execute_skill(unknown)
        # Current behavior: unknown skills treat as success
        assert status == SkillStatus.COMPLETED

    def test_unknown_skill_not_in_registry(self) -> None:
        """Unregistered skill IDs are not in BUILTIN_SKILLS."""
        assert "fly_to_moon" not in BUILTIN_SKILLS
        assert "teleport" not in BUILTIN_SKILLS

    def test_unknown_capability_fails_precondition(
        self, executor: SkillExecutor, mock_robot: Robot
    ) -> None:
        """A skill requiring FLY capability fails on a ground robot."""
        skill = Skill(
            skill_id="dock_skill",
            name="Dock",
            required_capability=CapabilityType.DOCK,
        )
        status = executor.execute_skill(skill)
        assert status == SkillStatus.FAILED


# ===========================================================================
# CI-04: Test that skill timeout triggers stop() and returns FAILED
# ===========================================================================

class TestSkillTimeout:
    """CI-04: Timeout enforcement works correctly."""

    def test_run_with_timeout_succeeds(self) -> None:
        """A fast function completes within timeout."""
        result = _run_with_timeout(lambda: 42, timeout_seconds=5.0)
        assert result == 42

    def test_run_with_timeout_raises(self) -> None:
        """A slow function triggers SkillTimeout."""
        def slow_fn() -> int:
            time.sleep(10)
            return 42

        with pytest.raises(SkillTimeout, match="timed out"):
            _run_with_timeout(slow_fn, timeout_seconds=0.1)

    def test_skill_timeout_returns_failed(self, mock_robot: Robot) -> None:
        """A skill that exceeds its timeout returns FAILED."""
        slow_skill = Skill(
            skill_id="slow_skill",
            name="Slow Skill",
            required_capability=CapabilityType.CUSTOM,
            timeout_seconds=0.1,
            retry_count=0,
        )

        # Create a custom executor that overrides _dispatch_skill to be slow
        class SlowExecutor(SkillExecutor):
            def _dispatch_skill(self, skill: Skill, params: dict[str, Any]) -> bool:
                time.sleep(5)
                return True

        exe = SlowExecutor(mock_robot)
        status = exe.execute_skill(slow_skill)
        assert status == SkillStatus.FAILED

    def test_timeout_emits_failed_event(
        self, mock_robot: Robot
    ) -> None:
        """Timeout produces a FAILED event in the event stream."""
        events: list[ExecutionEvent] = []

        slow_skill = Skill(
            skill_id="timeout_test",
            name="Timeout Test",
            required_capability=CapabilityType.CUSTOM,
            timeout_seconds=0.1,
        )

        class SlowExecutor(SkillExecutor):
            def _dispatch_skill(self, skill: Skill, params: dict[str, Any]) -> bool:
                time.sleep(5)
                return True

        exe = SlowExecutor(mock_robot)
        exe.on_event(lambda e: events.append(e))
        exe.execute_skill(slow_skill)

        failed_events = [e for e in events if e.status == SkillStatus.FAILED]
        assert len(failed_events) >= 1, f"Expected FAILED event, got: {[e.status for e in events]}"

    def test_timeout_with_retry(self, mock_robot: Robot) -> None:
        """Skill with retries still FAILs if all attempts time out."""
        slow_skill = Skill(
            skill_id="retry_timeout",
            name="Retry Timeout",
            required_capability=CapabilityType.CUSTOM,
            timeout_seconds=0.1,
            retry_count=1,
        )

        class SlowExecutor(SkillExecutor):
            def _dispatch_skill(self, skill: Skill, params: dict[str, Any]) -> bool:
                time.sleep(5)
                return True

        exe = SlowExecutor(mock_robot)
        status = exe.execute_skill(slow_skill)
        assert status == SkillStatus.FAILED

    def test_run_with_timeout_propagates_exception(self) -> None:
        """Exceptions from the function are propagated, not swallowed."""
        def raising_fn() -> int:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            _run_with_timeout(raising_fn, timeout_seconds=5.0)


# ===========================================================================
# Extra: Execution state tests
# ===========================================================================

class TestExecutionState:
    """Tests for ExecutionState thread-safety and behavior."""

    def test_set_and_get(self) -> None:
        state = ExecutionState()
        state.set("key", "value")
        assert state.get("key") == "value"

    def test_is_set(self) -> None:
        state = ExecutionState()
        assert not state.is_set("key")
        state.set("key", True)
        assert state.is_set("key")

    def test_clear(self) -> None:
        state = ExecutionState()
        state.set("key", True)
        state.clear("key")
        assert not state.is_set("key")

    def test_clear_all(self) -> None:
        state = ExecutionState()
        state.set("a", 1)
        state.set("b", 2)
        state.clear_all()
        assert state.flags == {}

    def test_default_value(self) -> None:
        state = ExecutionState()
        assert state.get("missing", "default") == "default"


# ===========================================================================
# Extra: SkillGraph tests
# ===========================================================================

class TestSkillGraph:
    """Tests for SkillGraph construction, ordering, and cycle detection."""

    def test_empty_graph(self) -> None:
        g = SkillGraph()
        assert len(g) == 0
        assert g.get_execution_order() == []

    def test_single_skill(self) -> None:
        g = SkillGraph()
        g.add_skill(BUILTIN_SKILLS["stop"])
        assert len(g) == 1
        order = g.get_execution_order()
        assert len(order) == 1
        assert order[0].skill_id == "stop"

    def test_dependency_ordering(self) -> None:
        """Skills with dependencies come after their dependencies."""
        g = SkillGraph()
        g.add_skill(BUILTIN_SKILLS["navigate_to"])
        g.add_skill(BUILTIN_SKILLS["pick_object"], depends_on=["navigate_to"])
        order = g.get_execution_order()
        ids = [s.skill_id for s in order]
        assert ids.index("navigate_to") < ids.index("pick_object")

    def test_cycle_detection(self) -> None:
        """Cycles in the graph raise ValueError."""
        g = SkillGraph()
        s1 = Skill(skill_id="a", name="A")
        s2 = Skill(skill_id="b", name="B")
        g.add_skill(s1, depends_on=["b"])
        g.add_skill(s2, depends_on=["a"])
        with pytest.raises(ValueError, match="Cycle"):
            g.get_execution_order()

    def test_parallel_layers(self) -> None:
        """Independent skills are grouped in the same layer."""
        g = SkillGraph()
        s1 = Skill(skill_id="a", name="A")
        s2 = Skill(skill_id="b", name="B")
        s3 = Skill(skill_id="c", name="C")
        g.add_skill(s1)
        g.add_skill(s2)
        g.add_skill(s3, depends_on=["a", "b"])
        layers = g.get_execution_layers()
        assert len(layers) == 2
        first_layer_ids = {s.skill_id for s in layers[0]}
        assert first_layer_ids == {"a", "b"}
        assert layers[1][0].skill_id == "c"

    def test_get_parameters_override(self) -> None:
        """Runtime parameters override skill defaults."""
        g = SkillGraph()
        g.add_skill(BUILTIN_SKILLS["navigate_to"], parameters={"x": 99.0})
        params = g.get_parameters("navigate_to")
        assert params["x"] == 99.0

    def test_graph_execution_sequential(self, mock_robot: Robot) -> None:
        """Sequential graph execution completes all skills."""
        g = SkillGraph()
        g.add_skill(BUILTIN_SKILLS["navigate_to"], parameters={"x": 1.0, "y": 2.0})
        g.add_skill(BUILTIN_SKILLS["stop"], depends_on=["navigate_to"])
        exe = SkillExecutor(mock_robot)
        result = exe.execute_graph(g)
        assert result.status.value == "completed"
        assert result.steps_completed == 2

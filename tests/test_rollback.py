"""
Tests for RollbackRegistry and RollbackExecutor.

Coverage:
    - RollbackRegistry — get_undo returns no-op for unknown skills
    - RollbackRegistry — get_undo for pick_object returns callable that opens gripper
    - RollbackRegistry — get_undo for navigate_to captures position before skill runs
    - RollbackRegistry — register() adds custom undo factory
    - RollbackExecutor — rollback stack empty before execution
    - RollbackExecutor — rollback stack grows as skills complete
    - RollbackExecutor — successful graph: rollback stack populated, no rollback triggered
    - RollbackExecutor — failed graph: undo called in reverse order
    - RollbackExecutor — undo error doesn't mask original failure
    - RollbackExecutor — last_rollback empty after successful graph
    - RollbackExecutor — last_rollback contains IDs in reverse order after failure
    - RollbackExecutor — emit_event called with correct fields on rollback
    - RollbackExecutor — result status is FAILED when graph fails
    - RollbackExecutor — result status is COMPLETED when graph succeeds
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import TaskStatus, RecoveryAction
from apyrobo.skills.agent import Agent
from apyrobo.skills.executor import SkillExecutor, SkillGraph, SkillStatus
from apyrobo.skills.skill import Skill, BUILTIN_SKILLS, CapabilityType
from apyrobo.skills.rollback import RollbackAction, RollbackRegistry, RollbackExecutor
from apyrobo.observability import clear_event_handlers, on_event, ObservabilityEvent


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_robot() -> Robot:
    return Robot.discover("mock://turtlebot4")


@pytest.fixture
def registry(mock_robot: Robot) -> RollbackRegistry:
    return RollbackRegistry(mock_robot)


@pytest.fixture
def executor(mock_robot: Robot) -> SkillExecutor:
    return SkillExecutor(mock_robot)


@pytest.fixture(autouse=True)
def clear_obs_handlers():
    """Remove any global observability handlers between tests."""
    yield
    clear_event_handlers()


def _make_failing_executor(mock_robot: Robot, fail_on: str) -> SkillExecutor:
    """Return a SkillExecutor whose execute_skill returns FAILED for *fail_on*."""

    class PatchedExecutor(SkillExecutor):
        def execute_skill(
            self, skill: Skill, parameters: dict[str, Any] | None = None
        ) -> SkillStatus:
            if skill.skill_id == fail_on:
                return SkillStatus.FAILED
            return super().execute_skill(skill, parameters)

    return PatchedExecutor(mock_robot)


def _make_always_failing_executor(mock_robot: Robot) -> SkillExecutor:
    """Return a SkillExecutor whose every execute_skill returns FAILED."""

    class AlwaysFailExecutor(SkillExecutor):
        def execute_skill(
            self, skill: Skill, parameters: dict[str, Any] | None = None
        ) -> SkillStatus:
            return SkillStatus.FAILED

    return AlwaysFailExecutor(mock_robot)


def _simple_graph(*skill_ids: str, **params_per_id: dict) -> SkillGraph:
    """Build a simple linear (sequential) SkillGraph from built-in skill IDs."""
    g = SkillGraph()
    prev: str | None = None
    for sid in skill_ids:
        skill = BUILTIN_SKILLS[sid]
        extra = params_per_id.get(sid, {})
        g.add_skill(skill, depends_on=[prev] if prev else None, parameters=extra)
        prev = sid
    return g


# ---------------------------------------------------------------------------
# RollbackRegistry tests
# ---------------------------------------------------------------------------

class TestRollbackRegistry:

    # 1. get_undo returns no-op for unknown skills
    def test_get_undo_unknown_skill_is_noop(self, registry: RollbackRegistry) -> None:
        undo = registry.get_undo("totally_unknown_skill", {})
        assert callable(undo)
        # Must not raise
        undo()

    # 2. get_undo for pick_object returns callable that opens gripper
    def test_get_undo_pick_object_opens_gripper(
        self, registry: RollbackRegistry, mock_robot: Robot
    ) -> None:
        # Close the gripper first so we can verify it opens
        mock_robot.gripper_close()
        undo = registry.get_undo("pick_object", {})
        assert callable(undo)
        undo()
        # After undo, the gripper should be open
        # (MockAdapter tracks gripper state via _gripper_open flag)
        adapter = mock_robot._adapter
        assert getattr(adapter, "_gripper_open", True) is True

    # 3. get_undo for navigate_to captures position before skill runs
    def test_get_undo_navigate_to_captures_pre_skill_position(
        self, registry: RollbackRegistry, mock_robot: Robot
    ) -> None:
        # Move robot to a known start position
        mock_robot.move(1.0, 2.0)
        assert mock_robot.get_position() == (1.0, 2.0)

        # Get the undo callable (captures position = (1.0, 2.0))
        undo = registry.get_undo("navigate_to", {"x": 9.0, "y": 9.0})

        # Simulate the skill running — robot moves elsewhere
        mock_robot.move(9.0, 9.0)
        assert mock_robot.get_position() == (9.0, 9.0)

        # Undo should restore robot to (1.0, 2.0)
        undo()
        assert mock_robot.get_position() == (1.0, 2.0)

    # 4. navigate_to undo does NOT capture position after skill runs
    def test_get_undo_navigate_to_position_is_snapshot_not_live(
        self, registry: RollbackRegistry, mock_robot: Robot
    ) -> None:
        mock_robot.move(3.0, 4.0)
        undo = registry.get_undo("navigate_to", {})
        # Move robot after capturing undo
        mock_robot.move(5.0, 6.0)
        undo()
        # Should go back to (3.0, 4.0), not (5.0, 6.0)
        assert mock_robot.get_position() == (3.0, 4.0)

    # 5. get_undo for place_object returns callable that closes gripper
    def test_get_undo_place_object_closes_gripper(
        self, registry: RollbackRegistry, mock_robot: Robot
    ) -> None:
        mock_robot.gripper_open()
        undo = registry.get_undo("place_object", {})
        undo()
        adapter = mock_robot._adapter
        assert getattr(adapter, "_gripper_open", False) is False

    # 6. get_undo for stop is a no-op (but callable)
    def test_get_undo_stop_is_noop(self, registry: RollbackRegistry) -> None:
        undo = registry.get_undo("stop", {})
        assert callable(undo)
        undo()  # Must not raise

    # 7. register() adds a custom undo factory
    def test_register_custom_factory(
        self, registry: RollbackRegistry
    ) -> None:
        called: list[str] = []

        def custom_factory(params: dict):
            return lambda: called.append("undone")

        registry.register("custom_skill", custom_factory)
        undo = registry.get_undo("custom_skill", {})
        undo()
        assert called == ["undone"]

    # 8. register() overrides built-in factory
    def test_register_overrides_builtin(
        self, registry: RollbackRegistry
    ) -> None:
        called: list[str] = []

        def my_factory(params: dict):
            return lambda: called.append("my_undo")

        registry.register("pick_object", my_factory)
        undo = registry.get_undo("pick_object", {})
        undo()
        assert called == ["my_undo"]

    # 9. suffixed skill IDs (e.g. navigate_to_0) are handled
    def test_get_undo_suffixed_skill_id(
        self, registry: RollbackRegistry, mock_robot: Robot
    ) -> None:
        mock_robot.move(7.0, 8.0)
        undo = registry.get_undo("navigate_to_0", {"x": 0.0, "y": 0.0})
        mock_robot.move(0.0, 0.0)
        undo()
        assert mock_robot.get_position() == (7.0, 8.0)

    # 10. factory exception falls back to no-op without raising
    def test_factory_exception_falls_back_to_noop(
        self, registry: RollbackRegistry
    ) -> None:
        def bad_factory(params: dict):
            raise RuntimeError("factory error")

        registry.register("bad_skill", bad_factory)
        undo = registry.get_undo("bad_skill", {})
        assert callable(undo)
        undo()  # Must not raise


# ---------------------------------------------------------------------------
# RollbackExecutor tests
# ---------------------------------------------------------------------------

class TestRollbackExecutor:

    # 11. rollback stack is empty before execution
    def test_rollback_stack_empty_before_execution(
        self, executor: SkillExecutor, registry: RollbackRegistry
    ) -> None:
        rb = RollbackExecutor(executor, registry)
        assert rb.rollback_stack == []

    # 12. rollback stack grows as skills complete
    def test_rollback_stack_grows_on_success(
        self, executor: SkillExecutor, registry: RollbackRegistry
    ) -> None:
        graph = _simple_graph("stop")
        rb = RollbackExecutor(executor, registry)
        rb.execute_graph(graph)
        assert len(rb.rollback_stack) == 1
        assert rb.rollback_stack[0].skill_id == "stop"

    # 13. successful graph: rollback stack populated, no rollback triggered
    def test_successful_graph_stack_populated_no_rollback(
        self,
        mock_robot: Robot,
        executor: SkillExecutor,
        registry: RollbackRegistry,
    ) -> None:
        graph = _simple_graph(
            "navigate_to", "stop",
            **{"navigate_to": {"x": 1.0, "y": 1.0}},
        )
        rb = RollbackExecutor(executor, registry)
        result = rb.execute_graph(graph)

        assert result.status == TaskStatus.COMPLETED
        assert len(rb.rollback_stack) == 2
        assert rb.last_rollback == []

    # 14. result status is COMPLETED when graph succeeds
    def test_result_status_completed_on_success(
        self, executor: SkillExecutor, registry: RollbackRegistry
    ) -> None:
        graph = _simple_graph("stop")
        rb = RollbackExecutor(executor, registry)
        result = rb.execute_graph(graph)
        assert result.status == TaskStatus.COMPLETED

    # 15. result status is FAILED when graph fails
    def test_result_status_failed_on_failure(
        self, mock_robot: Robot, registry: RollbackRegistry
    ) -> None:
        failing = _make_always_failing_executor(mock_robot)
        graph = _simple_graph("stop")
        rb = RollbackExecutor(failing, registry)
        result = rb.execute_graph(graph)
        assert result.status == TaskStatus.FAILED

    # 16. failed graph: undo called in reverse order
    def test_undo_called_in_reverse_order(
        self, mock_robot: Robot, registry: RollbackRegistry
    ) -> None:
        call_log: list[str] = []

        def nav_factory(params: dict):
            return lambda: call_log.append("undo_navigate_to")

        def stop_factory(params: dict):
            return lambda: call_log.append("undo_stop")

        registry.register("navigate_to", nav_factory)
        registry.register("stop", stop_factory)

        failing_exec = _make_failing_executor(mock_robot, fail_on="pick_object")

        # graph: navigate_to → stop → pick_object (fails)
        graph = SkillGraph()
        graph.add_skill(BUILTIN_SKILLS["navigate_to"], parameters={"x": 1.0, "y": 0.0})
        graph.add_skill(BUILTIN_SKILLS["stop"], depends_on=["navigate_to"])
        graph.add_skill(BUILTIN_SKILLS["pick_object"], depends_on=["stop"])

        rb = RollbackExecutor(failing_exec, registry)
        rb.execute_graph(graph)

        # navigate_to and stop committed; pick_object failed
        # Undo order: stop first, then navigate_to (LIFO)
        assert call_log == ["undo_stop", "undo_navigate_to"]

    # 17. undo error doesn't mask original failure
    def test_undo_error_does_not_mask_failure(
        self, mock_robot: Robot, registry: RollbackRegistry
    ) -> None:
        def exploding_factory(params: dict):
            def _undo():
                raise RuntimeError("undo exploded")
            return _undo

        registry.register("stop", exploding_factory)

        failing_exec = _make_failing_executor(mock_robot, fail_on="pick_object")
        graph = SkillGraph()
        graph.add_skill(BUILTIN_SKILLS["stop"])
        graph.add_skill(BUILTIN_SKILLS["pick_object"], depends_on=["stop"])

        rb = RollbackExecutor(failing_exec, registry)
        result = rb.execute_graph(graph)

        # Despite undo error, result is still FAILED (original failure preserved)
        assert result.status == TaskStatus.FAILED

    # 18. last_rollback is empty after successful execution
    def test_last_rollback_empty_after_success(
        self, executor: SkillExecutor, registry: RollbackRegistry
    ) -> None:
        graph = _simple_graph("stop")
        rb = RollbackExecutor(executor, registry)
        rb.execute_graph(graph)
        assert rb.last_rollback == []

    # 19. last_rollback contains IDs in reverse order after failure
    def test_last_rollback_reverse_order_after_failure(
        self, mock_robot: Robot, registry: RollbackRegistry
    ) -> None:
        failing_exec = _make_failing_executor(mock_robot, fail_on="pick_object")

        graph = SkillGraph()
        graph.add_skill(BUILTIN_SKILLS["navigate_to"], parameters={"x": 1.0, "y": 0.0})
        graph.add_skill(BUILTIN_SKILLS["stop"], depends_on=["navigate_to"])
        graph.add_skill(BUILTIN_SKILLS["pick_object"], depends_on=["stop"])

        rb = RollbackExecutor(failing_exec, registry)
        rb.execute_graph(graph)

        # navigate_to and stop committed; pick_object failed
        # Reverse order: stop first, then navigate_to
        assert rb.last_rollback == ["stop", "navigate_to"]

    # 20. emit_event called with correct fields on rollback
    def test_emit_event_on_rollback(
        self, mock_robot: Robot, registry: RollbackRegistry
    ) -> None:
        captured: list[ObservabilityEvent] = []
        on_event(captured.append)

        failing_exec = _make_failing_executor(mock_robot, fail_on="pick_object")

        graph = SkillGraph()
        graph.add_skill(BUILTIN_SKILLS["stop"])
        graph.add_skill(BUILTIN_SKILLS["pick_object"], depends_on=["stop"])

        rb = RollbackExecutor(failing_exec, registry)
        rb.execute_graph(graph)

        rollback_events = [e for e in captured if e.event_type == "plan.rolled_back"]
        assert len(rollback_events) == 1
        evt = rollback_events[0]
        assert "rolled_back_skills" in evt.data
        assert "reason" in evt.data
        assert isinstance(evt.data["rolled_back_skills"], list)

    # 21. emit_event not called on successful execution
    def test_no_emit_event_on_success(
        self, executor: SkillExecutor, registry: RollbackRegistry
    ) -> None:
        captured: list[ObservabilityEvent] = []
        on_event(captured.append)

        graph = _simple_graph("stop")
        rb = RollbackExecutor(executor, registry)
        rb.execute_graph(graph)

        rollback_events = [e for e in captured if e.event_type == "plan.rolled_back"]
        assert rollback_events == []

    # 22. result metadata contains rolled_back=True on failure
    def test_result_metadata_rolled_back_true_on_failure(
        self, mock_robot: Robot, registry: RollbackRegistry
    ) -> None:
        failing_exec = _make_always_failing_executor(mock_robot)
        graph = _simple_graph("stop")
        rb = RollbackExecutor(failing_exec, registry)
        result = rb.execute_graph(graph)
        assert result.metadata.get("rolled_back") is True

    # 23. result metadata contains rolled_back=False on success
    def test_result_metadata_rolled_back_false_on_success(
        self, executor: SkillExecutor, registry: RollbackRegistry
    ) -> None:
        graph = _simple_graph("stop")
        rb = RollbackExecutor(executor, registry)
        result = rb.execute_graph(graph)
        assert result.metadata.get("rolled_back") is False

    # 24. RollbackExecutor works without a registry (no-ops for all undos)
    def test_works_without_registry(
        self, mock_robot: Robot
    ) -> None:
        failing_exec = _make_failing_executor(mock_robot, fail_on="pick_object")

        graph = SkillGraph()
        graph.add_skill(BUILTIN_SKILLS["stop"])
        graph.add_skill(BUILTIN_SKILLS["pick_object"], depends_on=["stop"])

        rb = RollbackExecutor(failing_exec, registry=None)
        result = rb.execute_graph(graph)

        assert result.status == TaskStatus.FAILED
        # No registry means no undo actions — rollback still tracks IDs
        assert "stop" in rb.last_rollback

    # 25. rollback_stack is reset between execute_graph calls
    def test_rollback_stack_reset_between_calls(
        self, executor: SkillExecutor, registry: RollbackRegistry
    ) -> None:
        graph = _simple_graph("stop")
        rb = RollbackExecutor(executor, registry)

        # First call — stack grows to 1
        rb.execute_graph(graph)
        assert len(rb.rollback_stack) == 1

        # Second call — stack starts fresh
        rb.execute_graph(graph)
        assert len(rb.rollback_stack) == 1

    # 26. rolled_back_skills in event data matches last_rollback
    def test_event_rolled_back_skills_matches_last_rollback(
        self, mock_robot: Robot, registry: RollbackRegistry
    ) -> None:
        captured: list[ObservabilityEvent] = []
        on_event(captured.append)

        failing_exec = _make_failing_executor(mock_robot, fail_on="pick_object")

        graph = SkillGraph()
        graph.add_skill(BUILTIN_SKILLS["navigate_to"], parameters={"x": 1.0, "y": 0.0})
        graph.add_skill(BUILTIN_SKILLS["stop"], depends_on=["navigate_to"])
        graph.add_skill(BUILTIN_SKILLS["pick_object"], depends_on=["stop"])

        rb = RollbackExecutor(failing_exec, registry)
        rb.execute_graph(graph)

        rollback_events = [e for e in captured if e.event_type == "plan.rolled_back"]
        assert len(rollback_events) == 1
        assert rollback_events[0].data["rolled_back_skills"] == rb.last_rollback

    # 27. empty graph succeeds with no rollback
    def test_empty_graph_succeeds(
        self, executor: SkillExecutor, registry: RollbackRegistry
    ) -> None:
        graph = SkillGraph()
        rb = RollbackExecutor(executor, registry)
        result = rb.execute_graph(graph)
        assert result.status == TaskStatus.COMPLETED
        assert rb.rollback_stack == []
        assert rb.last_rollback == []

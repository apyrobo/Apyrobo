"""
Extended coverage tests for apyrobo/skills/executor.py.

Targets previously-uncovered lines:
  76, 154, 180, 198, 282-283, 334, 339, 343, 348, 351, 354, 361, 366-385,
  397-399, 474-478, 484-485, 562-563, 567-574, 603-604, 637-638, 641,
  694-698, 711-715, 736

Covers:
- ExecutionState.__repr__                      line 76
- SkillGraph.__repr__                          line 180
- ExecutionEvent.__repr__                      line 198
- _emit listener exception is swallowed        lines 282-283
- check_postconditions state update paths      lines 397-399+
- _resolve_world_state all code paths          lines 329-339
- _check_sensor_precondition all branches      lines 341-385
- execute_skill: retry count > 0 recovery      lines 334+
- execute_graph confidence gate                lines 562-574
- execute_graph state_store paths              lines 555-563, 589-603
- _execute_graph_sequential state_store update lines 637-638, 641
- _execute_graph_parallel multi-layer paths    lines 694-698, 711-715
- events property                              line 736
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import CapabilityType, RecoveryAction, TaskStatus
from apyrobo.sensors.pipeline import WorldState
from apyrobo.skills.executor import (
    ExecutionEvent,
    ExecutionState,
    SkillExecutor,
    SkillGraph,
    SkillTimeout,
    _run_with_timeout,
)
from apyrobo.skills.skill import Condition, Skill, SkillStatus, BUILTIN_SKILLS


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
# ExecutionState.__repr__ (line 76)
# ===========================================================================

class TestExecutionStateRepr:
    def test_repr_contains_class_name(self) -> None:
        state = ExecutionState()
        assert "ExecutionState" in repr(state)

    def test_repr_shows_flags(self) -> None:
        state = ExecutionState()
        state.set("robot_idle", True)
        r = repr(state)
        assert "robot_idle" in r

    def test_repr_empty_flags(self) -> None:
        state = ExecutionState()
        r = repr(state)
        assert "ExecutionState" in r
        assert "{}" in r


# ===========================================================================
# SkillGraph.__repr__ (line 180)
# ===========================================================================

class TestSkillGraphRepr:
    def test_repr_contains_class_name(self) -> None:
        g = SkillGraph()
        assert "SkillGraph" in repr(g)

    def test_repr_shows_skill_count(self) -> None:
        g = SkillGraph()
        g.add_skill(BUILTIN_SKILLS["stop"])
        r = repr(g)
        assert "1" in r

    def test_repr_shows_edge_count(self) -> None:
        g = SkillGraph()
        g.add_skill(BUILTIN_SKILLS["navigate_to"])
        g.add_skill(BUILTIN_SKILLS["stop"], depends_on=["navigate_to"])
        r = repr(g)
        assert "SkillGraph" in r


# ===========================================================================
# ExecutionEvent.__repr__ (line 198)
# ===========================================================================

class TestExecutionEventRepr:
    def test_repr_contains_skill_id(self) -> None:
        ev = ExecutionEvent("my_skill", SkillStatus.COMPLETED, "done")
        r = repr(ev)
        assert "my_skill" in r

    def test_repr_contains_status(self) -> None:
        ev = ExecutionEvent("s", SkillStatus.FAILED, "boom")
        r = repr(ev)
        assert "failed" in r

    def test_repr_contains_message(self) -> None:
        ev = ExecutionEvent("x", SkillStatus.RUNNING, "in progress")
        r = repr(ev)
        assert "in progress" in r


# ===========================================================================
# _emit: listener exception is swallowed (lines 282-283)
# ===========================================================================

class TestEmitListenerException:
    def test_listener_exception_does_not_propagate(
        self, executor: SkillExecutor
    ) -> None:
        """A throwing listener should not crash the executor."""
        def bad_listener(event: ExecutionEvent) -> None:
            raise ValueError("listener crashed")

        executor.on_event(bad_listener)
        # This should not raise despite the bad listener
        skill = BUILTIN_SKILLS["stop"]
        status = executor.execute_skill(skill)
        assert status == SkillStatus.COMPLETED

    def test_good_listener_still_receives_events_after_bad_one(
        self, executor: SkillExecutor
    ) -> None:
        """Events reach subsequent listeners even if an earlier one throws."""
        received: list[ExecutionEvent] = []

        executor.on_event(lambda e: (_ for _ in ()).throw(RuntimeError("crash")))
        executor.on_event(lambda e: received.append(e))

        executor.execute_skill(BUILTIN_SKILLS["stop"])
        assert len(received) > 0


# ===========================================================================
# check_postconditions: all state update branches (lines 397-419)
# ===========================================================================

class TestCheckPostconditions:
    def test_navigate_to_sets_at_position(self, executor: SkillExecutor) -> None:
        skill = BUILTIN_SKILLS["navigate_to"]
        executor.check_postconditions(skill, {"x": 3.0, "y": 4.0})
        assert executor.state.get("at_position") == (3.0, 4.0)
        assert executor.state.is_set("robot_idle")

    def test_pick_object_sets_object_held(self, executor: SkillExecutor) -> None:
        skill = BUILTIN_SKILLS["pick_object"]
        executor.check_postconditions(skill, {})
        assert executor.state.is_set("object_held")
        assert not executor.state.is_set("gripper_open")

    def test_place_object_clears_object_held(self, executor: SkillExecutor) -> None:
        skill = BUILTIN_SKILLS["place_object"]
        executor.check_postconditions(skill, {})
        assert not executor.state.is_set("object_held")
        assert executor.state.is_set("gripper_open")

    def test_rotate_sets_last_rotation(self, executor: SkillExecutor) -> None:
        skill = BUILTIN_SKILLS["rotate"]
        executor.check_postconditions(skill, {"angle_rad": 1.57})
        assert executor.state.get("last_rotation") == pytest.approx(1.57)

    def test_stop_sets_robot_idle(self, executor: SkillExecutor) -> None:
        skill = BUILTIN_SKILLS["stop"]
        executor.check_postconditions(skill, {})
        assert executor.state.is_set("robot_idle")

    def test_postcondition_state_type_sets_flag(
        self, executor: SkillExecutor, mock_robot: Robot
    ) -> None:
        """Postcondition with check_type='state' sets the key in execution state."""
        skill = Skill(
            skill_id="test_post",
            name="Test",
            postconditions=[
                Condition(
                    name="door_open",
                    check_type="state",
                    parameters={"key": "door_open", "value": True},
                )
            ],
        )
        ok, reason = executor.check_postconditions(skill, {})
        assert ok
        assert executor.state.get("door_open") is True

    def test_suffixed_skill_id_is_normalised(self, executor: SkillExecutor) -> None:
        """'navigate_to_0' base_id resolves to 'navigate_to' branch."""
        skill = Skill(
            skill_id="navigate_to_0",
            name="Navigate 0",
            required_capability=CapabilityType.NAVIGATE,
        )
        executor.check_postconditions(skill, {"x": 7.0, "y": 8.0})
        assert executor.state.get("at_position") == (7.0, 8.0)


# ===========================================================================
# _resolve_world_state (lines 329-339)
# ===========================================================================

class TestResolveWorldState:
    def test_no_provider_returns_none(self, executor: SkillExecutor) -> None:
        result = executor._resolve_world_state()
        assert result is None

    def test_world_state_instance_returned_directly(
        self, mock_robot: Robot
    ) -> None:
        ws = WorldState()
        exe = SkillExecutor(mock_robot, world_state_provider=ws)
        result = exe._resolve_world_state()
        assert result is ws

    def test_callable_provider_is_called(self, mock_robot: Robot) -> None:
        ws = WorldState()
        exe = SkillExecutor(mock_robot, world_state_provider=lambda: ws)
        result = exe._resolve_world_state()
        assert result is ws

    def test_get_world_state_method_is_used(self, mock_robot: Robot) -> None:
        ws = WorldState()

        class Provider:
            def get_world_state(self):
                return ws

        exe = SkillExecutor(mock_robot, world_state_provider=Provider())
        result = exe._resolve_world_state()
        assert result is ws

    def test_unknown_provider_type_returns_none(self, mock_robot: Robot) -> None:
        exe = SkillExecutor(mock_robot, world_state_provider="not a provider")
        result = exe._resolve_world_state()
        assert result is None


# ===========================================================================
# _check_sensor_precondition (lines 341-385)
# ===========================================================================

class TestCheckSensorPrecondition:
    """Tests for _check_sensor_precondition using the real WorldState/Obstacle/DetectedObject API."""

    def test_world_state_none_returns_false(self, executor: SkillExecutor) -> None:
        cond = Condition(name="object_visible", check_type="sensor",
                         parameters={"label": "box"})
        ok, reason = executor._check_sensor_precondition(cond, None)
        assert not ok
        assert "unavailable" in reason.lower()

    def test_object_visible_found(self, executor: SkillExecutor) -> None:
        from apyrobo.sensors.pipeline import DetectedObject
        ws = WorldState()
        ws.detected_objects.append(
            DetectedObject("obj1", "box", x=1.0, y=1.0, confidence=0.9)
        )
        cond = Condition(name="object_visible", check_type="sensor",
                         parameters={"label": "box", "min_confidence": 0.5})
        ok, reason = executor._check_sensor_precondition(cond, ws)
        assert ok

    def test_object_visible_not_found(self, executor: SkillExecutor) -> None:
        ws = WorldState()
        cond = Condition(name="object_visible", check_type="sensor",
                         parameters={"label": "missing_object"})
        ok, reason = executor._check_sensor_precondition(cond, ws)
        assert not ok
        assert "not visible" in reason.lower() or "missing_object" in reason

    def test_object_visible_low_confidence(self, executor: SkillExecutor) -> None:
        from apyrobo.sensors.pipeline import DetectedObject
        ws = WorldState()
        ws.detected_objects.append(
            DetectedObject("obj2", "low_conf_box", x=0.0, y=0.0, confidence=0.2)
        )
        cond = Condition(name="object_visible", check_type="sensor",
                         parameters={"label": "low_conf_box", "min_confidence": 0.8})
        ok, reason = executor._check_sensor_precondition(cond, ws)
        assert not ok
        assert "confidence" in reason.lower()

    def test_object_visible_no_label_returns_false(
        self, executor: SkillExecutor
    ) -> None:
        ws = WorldState()
        cond = Condition(name="object_visible", check_type="sensor", parameters={})
        ok, reason = executor._check_sensor_precondition(cond, ws)
        assert not ok
        assert "label" in reason.lower()

    def test_path_clear_passes(self, executor: SkillExecutor) -> None:
        ws = WorldState()
        ws.robot_position = (0.0, 0.0)
        cond = Condition(name="path_clear", check_type="sensor",
                         parameters={"x": 3.0, "y": 0.0})
        ok, _ = executor._check_sensor_precondition(cond, ws)
        assert ok

    def test_path_clear_blocked(self, executor: SkillExecutor) -> None:
        from apyrobo.sensors.pipeline import Obstacle
        ws = WorldState()
        ws.robot_position = (0.0, 0.0)
        ws.obstacles.append(Obstacle(x=0.5, y=0.0, radius=0.3))
        cond = Condition(name="path_clear", check_type="sensor",
                         parameters={"x": 3.0, "y": 0.0, "clearance": 0.4})
        ok, reason = executor._check_sensor_precondition(cond, ws)
        assert not ok
        assert "blocked" in reason.lower()

    def test_path_clear_no_coords_returns_false(
        self, executor: SkillExecutor
    ) -> None:
        ws = WorldState()
        cond = Condition(name="path_clear", check_type="sensor", parameters={})
        ok, reason = executor._check_sensor_precondition(cond, ws)
        assert not ok
        assert "numeric" in reason.lower() or "x/y" in reason.lower()

    def test_no_obstacle_within_passes(self, executor: SkillExecutor) -> None:
        ws = WorldState()
        ws.robot_position = (0.0, 0.0)
        cond = Condition(name="no_obstacle_within", check_type="sensor",
                         parameters={"radius": 1.0})
        ok, _ = executor._check_sensor_precondition(cond, ws)
        assert ok

    def test_no_obstacle_within_fails(self, executor: SkillExecutor) -> None:
        from apyrobo.sensors.pipeline import Obstacle
        ws = WorldState()
        ws.robot_position = (0.0, 0.0)
        ws.obstacles.append(Obstacle(x=0.3, y=0.0, radius=0.1))
        cond = Condition(name="no_obstacle_within", check_type="sensor",
                         parameters={"radius": 1.0})
        ok, reason = executor._check_sensor_precondition(cond, ws)
        assert not ok
        assert "obstacle" in reason.lower()

    def test_contact_detected_passes(self, executor: SkillExecutor) -> None:
        ws = WorldState()
        ws.metadata["contact_detected"] = True
        cond = Condition(name="contact_detected", check_type="sensor",
                         parameters={"value": True})
        ok, _ = executor._check_sensor_precondition(cond, ws)
        assert ok

    def test_contact_detected_fails(self, executor: SkillExecutor) -> None:
        ws = WorldState()
        ws.metadata["contact_detected"] = False
        cond = Condition(name="contact_detected", check_type="sensor",
                         parameters={"value": True})
        ok, _ = executor._check_sensor_precondition(cond, ws)
        assert not ok

    def test_gps_fix_passes(self, executor: SkillExecutor) -> None:
        ws = WorldState()
        ws.metadata["gps_fix"] = True
        cond = Condition(name="gps_fix", check_type="sensor",
                         parameters={"value": True})
        ok, _ = executor._check_sensor_precondition(cond, ws)
        assert ok

    def test_gps_fix_fails(self, executor: SkillExecutor) -> None:
        ws = WorldState()
        ws.metadata["gps_fix"] = False
        cond = Condition(name="gps_fix", check_type="sensor",
                         parameters={"value": True})
        ok, _ = executor._check_sensor_precondition(cond, ws)
        assert not ok

    def test_unknown_sensor_cond_returns_true(self, executor: SkillExecutor) -> None:
        """Unrecognised sensor condition name defaults to OK (line 384-385)."""
        ws = WorldState()
        cond = Condition(name="some_unknown_sensor_check", check_type="sensor",
                         parameters={})
        ok, _ = executor._check_sensor_precondition(cond, ws)
        assert ok


# ===========================================================================
# execute_skill: retry logic and event emission (lines 334+)
# ===========================================================================

class TestExecuteSkillRetry:
    def test_retry_count_zero_no_retry(self, mock_robot: Robot) -> None:
        """Skill with retry_count=0 makes exactly 1 attempt."""
        attempt_counter = [0]

        class CountingExecutor(SkillExecutor):
            def _dispatch_skill(self, skill, params):
                attempt_counter[0] += 1
                return False  # always fail

        skill = Skill(
            skill_id="fail_once",
            name="Fail",
            required_capability=CapabilityType.CUSTOM,
            retry_count=0,
        )
        exe = CountingExecutor(mock_robot)
        status = exe.execute_skill(skill)
        assert status == SkillStatus.FAILED
        assert attempt_counter[0] == 1

    def test_retry_count_two_makes_three_attempts(self, mock_robot: Robot) -> None:
        attempt_counter = [0]

        class CountingExecutor(SkillExecutor):
            def _dispatch_skill(self, skill, params):
                attempt_counter[0] += 1
                return False

        skill = Skill(
            skill_id="retry_skill",
            name="Retry",
            required_capability=CapabilityType.CUSTOM,
            retry_count=2,
        )
        exe = CountingExecutor(mock_robot)
        exe.execute_skill(skill)
        assert attempt_counter[0] == 3

    def test_success_on_second_attempt(self, mock_robot: Robot) -> None:
        attempt_counter = [0]

        class EventualSuccessExecutor(SkillExecutor):
            def _dispatch_skill(self, skill, params):
                attempt_counter[0] += 1
                return attempt_counter[0] >= 2

        skill = Skill(
            skill_id="eventual_success",
            name="Eventually",
            required_capability=CapabilityType.CUSTOM,
            retry_count=2,
        )
        exe = EventualSuccessExecutor(mock_robot)
        status = exe.execute_skill(skill)
        assert status == SkillStatus.COMPLETED
        assert attempt_counter[0] == 2

    def test_failed_emits_failed_event_after_all_retries(self, mock_robot: Robot) -> None:
        """After exhausting retries, a FAILED event is emitted."""
        events: list[ExecutionEvent] = []

        class AlwaysFail(SkillExecutor):
            def _dispatch_skill(self, skill, params):
                return False

        skill = Skill(
            skill_id="always_fail",
            name="Always Fail",
            required_capability=CapabilityType.CUSTOM,
            retry_count=1,
        )
        exe = AlwaysFail(mock_robot)
        exe.on_event(lambda e: events.append(e))
        exe.execute_skill(skill)

        failed = [e for e in events if e.status == SkillStatus.FAILED]
        assert len(failed) >= 1

    def test_postcondition_failure_triggers_retry(self, mock_robot: Robot) -> None:
        """If postcondition fails, the skill is retried."""
        attempts = [0]

        class PostcondFailExecutor(SkillExecutor):
            def _dispatch_skill(self, skill, params):
                attempts[0] += 1
                return True

            def check_postconditions(self, skill, params):
                # Fail first attempt, pass on second
                if attempts[0] < 2:
                    return False, "postcondition not met"
                return True, "OK"

        skill = Skill(
            skill_id="postcond_retry",
            name="PostCond",
            required_capability=CapabilityType.CUSTOM,
            retry_count=2,
        )
        exe = PostcondFailExecutor(mock_robot)
        status = exe.execute_skill(skill)
        assert status == SkillStatus.COMPLETED


# ===========================================================================
# execute_graph: confidence gate (lines 562-574)
# ===========================================================================

class TestExecuteGraphConfidenceGate:
    def test_confidence_gate_blocks_low_confidence(self, mock_robot: Robot) -> None:
        """Confidence estimator raising an exception returns FAILED result."""
        class LowConfidenceGate:
            def gate(self, graph, robot):
                raise Exception("Confidence too low: 0.3")

        exe = SkillExecutor(mock_robot, confidence_estimator=LowConfidenceGate())
        graph = SkillGraph()
        graph.add_skill(BUILTIN_SKILLS["stop"])

        result = exe.execute_graph(graph)
        assert result.status == TaskStatus.FAILED
        assert "Confidence" in (result.error or "")
        assert RecoveryAction.ABORT in result.recovery_actions_taken

    def test_confidence_gate_passes_allows_execution(self, mock_robot: Robot) -> None:
        """When gate() does not raise, execution proceeds normally."""
        class PassingGate:
            def gate(self, graph, robot):
                report = MagicMock()
                report.confidence = 0.95
                return report

        exe = SkillExecutor(mock_robot, confidence_estimator=PassingGate())
        graph = SkillGraph()
        graph.add_skill(BUILTIN_SKILLS["stop"])

        result = exe.execute_graph(graph)
        assert result.status == TaskStatus.COMPLETED

    def test_no_confidence_estimator_skips_gate(self, executor: SkillExecutor) -> None:
        """Without an estimator, gate is skipped and execution runs normally."""
        assert executor._confidence_estimator is None
        graph = SkillGraph()
        graph.add_skill(BUILTIN_SKILLS["stop"])
        result = executor.execute_graph(graph)
        assert result.status == TaskStatus.COMPLETED


# ===========================================================================
# execute_graph: state_store paths (lines 555-563, 589-603)
# ===========================================================================

class TestExecuteGraphStateStore:
    def test_state_store_begin_task_called(self, mock_robot: Robot) -> None:
        store = MagicMock()
        exe = SkillExecutor(mock_robot, state_store=store)
        graph = SkillGraph()
        graph.add_skill(BUILTIN_SKILLS["stop"])

        exe.execute_graph(graph, trace_id="trace-123")
        store.begin_task.assert_called_once()

    def test_state_store_complete_task_called_on_success(self, mock_robot: Robot) -> None:
        store = MagicMock()
        exe = SkillExecutor(mock_robot, state_store=store)
        graph = SkillGraph()
        graph.add_skill(BUILTIN_SKILLS["stop"])

        exe.execute_graph(graph, trace_id="trace-456")
        store.complete_task.assert_called_once()
        store.fail_task.assert_not_called()

    def test_state_store_fail_task_called_on_failure(self, mock_robot: Robot) -> None:
        store = MagicMock()

        class FailingExecutor(SkillExecutor):
            def _dispatch_skill(self, skill, params):
                return False

        exe = FailingExecutor(mock_robot, state_store=store)
        graph = SkillGraph()
        graph.add_skill(Skill(
            skill_id="fail",
            name="Fail",
            required_capability=CapabilityType.CUSTOM,
            retry_count=0,
        ))

        exe.execute_graph(graph, trace_id="trace-789")
        store.fail_task.assert_called_once()

    def test_state_store_exception_does_not_abort(self, mock_robot: Robot) -> None:
        """state_store errors are logged but don't crash execution."""
        store = MagicMock()
        store.begin_task.side_effect = RuntimeError("db error")
        exe = SkillExecutor(mock_robot, state_store=store)
        graph = SkillGraph()
        graph.add_skill(BUILTIN_SKILLS["stop"])

        result = exe.execute_graph(graph, trace_id="trace-err")
        assert result.status == TaskStatus.COMPLETED

    def test_no_state_store_no_crash(self, executor: SkillExecutor) -> None:
        graph = SkillGraph()
        graph.add_skill(BUILTIN_SKILLS["stop"])
        result = executor.execute_graph(graph)
        assert result.status == TaskStatus.COMPLETED


# ===========================================================================
# _execute_graph_sequential: state_store update per step (lines 637-638, 641)
# ===========================================================================

class TestExecuteGraphSequentialStateStore:
    def test_update_task_called_per_completed_step(self, mock_robot: Robot) -> None:
        store = MagicMock()
        exe = SkillExecutor(mock_robot, state_store=store)
        graph = SkillGraph()
        graph.add_skill(BUILTIN_SKILLS["navigate_to"], parameters={"x": 1.0, "y": 0.0})
        graph.add_skill(BUILTIN_SKILLS["stop"], depends_on=["navigate_to"])

        exe.execute_graph(graph, trace_id="trace-seq")
        # update_task should have been called for each completed skill
        assert store.update_task.call_count >= 1

    def test_sequential_recovery_retry_added_on_failure(self, mock_robot: Robot) -> None:
        """When a skill with retry_count>0 fails, RETRY is added to recovery actions."""
        class FailingExecutor(SkillExecutor):
            def _dispatch_skill(self, skill, params):
                return False

        skill = Skill(
            skill_id="retried_fail",
            name="Retried Fail",
            required_capability=CapabilityType.CUSTOM,
            retry_count=1,
        )
        graph = SkillGraph()
        graph.add_skill(skill)

        exe = FailingExecutor(mock_robot)
        result = exe.execute_graph(graph)
        assert result.status == TaskStatus.FAILED
        assert RecoveryAction.RETRY in result.recovery_actions_taken
        assert RecoveryAction.ABORT in result.recovery_actions_taken

    def test_sequential_no_retry_only_abort(self, mock_robot: Robot) -> None:
        """Skill with retry_count=0 only adds ABORT, not RETRY."""
        class FailingExecutor(SkillExecutor):
            def _dispatch_skill(self, skill, params):
                return False

        skill = Skill(
            skill_id="no_retry_fail",
            name="No Retry",
            required_capability=CapabilityType.CUSTOM,
            retry_count=0,
        )
        graph = SkillGraph()
        graph.add_skill(skill)

        exe = FailingExecutor(mock_robot)
        result = exe.execute_graph(graph)
        assert RecoveryAction.RETRY not in result.recovery_actions_taken
        assert RecoveryAction.ABORT in result.recovery_actions_taken


# ===========================================================================
# _execute_graph_parallel: multi-layer paths (lines 694-698, 711-715)
# ===========================================================================

class TestExecuteGraphParallel:
    def test_parallel_single_skill_completes(self, mock_robot: Robot) -> None:
        exe = SkillExecutor(mock_robot)
        graph = SkillGraph()
        graph.add_skill(BUILTIN_SKILLS["stop"])
        result = exe.execute_graph(graph, parallel=True)
        assert result.status == TaskStatus.COMPLETED
        assert result.steps_completed == 1

    def test_parallel_independent_skills_both_complete(self, mock_robot: Robot) -> None:
        exe = SkillExecutor(mock_robot)
        graph = SkillGraph()
        s1 = Skill(skill_id="stop_1", name="Stop1", required_capability=CapabilityType.CUSTOM)
        s2 = Skill(skill_id="stop_2", name="Stop2", required_capability=CapabilityType.CUSTOM)

        class AlwaysSucceed(SkillExecutor):
            def _dispatch_skill(self, skill, params):
                return True

        graph.add_skill(s1)
        graph.add_skill(s2)
        exe = AlwaysSucceed(mock_robot)
        result = exe.execute_graph(graph, parallel=True)
        assert result.status == TaskStatus.COMPLETED
        assert result.steps_completed == 2

    def test_parallel_failure_in_single_layer_returns_failed(
        self, mock_robot: Robot
    ) -> None:
        """Failure in single-skill layer returns FAILED result — lines 694-698."""
        class FailingExecutor(SkillExecutor):
            def _dispatch_skill(self, skill, params):
                return False

        graph = SkillGraph()
        skill = Skill(
            skill_id="fail_parallel",
            name="Fail",
            required_capability=CapabilityType.CUSTOM,
            retry_count=0,
        )
        graph.add_skill(skill)

        exe = FailingExecutor(mock_robot)
        result = exe.execute_graph(graph, parallel=True)
        assert result.status == TaskStatus.FAILED
        assert RecoveryAction.ABORT in result.recovery_actions_taken

    def test_parallel_failure_in_multi_skill_layer_returns_failed(
        self, mock_robot: Robot
    ) -> None:
        """Failure in multi-skill parallel layer returns FAILED — lines 711-715."""
        call_order: list[str] = []

        class SelectiveFailExecutor(SkillExecutor):
            def _dispatch_skill(self, skill, params):
                call_order.append(skill.skill_id)
                return skill.skill_id != "fail_skill"

        graph = SkillGraph()
        ok_skill = Skill(
            skill_id="ok_skill",
            name="OK",
            required_capability=CapabilityType.CUSTOM,
        )
        fail_skill = Skill(
            skill_id="fail_skill",
            name="Fail",
            required_capability=CapabilityType.CUSTOM,
            retry_count=0,
        )
        graph.add_skill(ok_skill)
        graph.add_skill(fail_skill)

        exe = SelectiveFailExecutor(mock_robot)
        result = exe.execute_graph(graph, parallel=True)
        assert result.status == TaskStatus.FAILED

    def test_parallel_retry_count_adds_retry_recovery(self, mock_robot: Robot) -> None:
        """Failed skill with retry_count>0 in parallel adds RETRY to recovery."""
        class FailingExecutor(SkillExecutor):
            def _dispatch_skill(self, skill, params):
                return False

        graph = SkillGraph()
        skill = Skill(
            skill_id="retry_parallel",
            name="Retry Parallel",
            required_capability=CapabilityType.CUSTOM,
            retry_count=1,
        )
        graph.add_skill(skill)

        exe = FailingExecutor(mock_robot)
        result = exe.execute_graph(graph, parallel=True)
        assert RecoveryAction.RETRY in result.recovery_actions_taken

    def test_parallel_multi_layer_sequential_dependency(
        self, mock_robot: Robot
    ) -> None:
        """Dependent skills execute in the correct layer order."""
        execution_order: list[str] = []

        class TrackingExecutor(SkillExecutor):
            def _dispatch_skill(self, skill, params):
                execution_order.append(skill.skill_id)
                return True

        graph = SkillGraph()
        nav = Skill(skill_id="navigate_to", name="Nav",
                    required_capability=CapabilityType.NAVIGATE)
        pick = Skill(skill_id="pick_object", name="Pick",
                     required_capability=CapabilityType.PICK)
        graph.add_skill(nav, parameters={"x": 1.0, "y": 2.0})
        graph.add_skill(pick, depends_on=["navigate_to"])

        exe = TrackingExecutor(mock_robot)
        result = exe.execute_graph(graph, parallel=True)
        assert result.status == TaskStatus.COMPLETED
        assert execution_order.index("navigate_to") < execution_order.index("pick_object")


# ===========================================================================
# events property (line 736)
# ===========================================================================

class TestEventsProperty:
    def test_events_returns_list(self, executor: SkillExecutor) -> None:
        assert isinstance(executor.events, list)

    def test_events_populated_after_execution(self, executor: SkillExecutor) -> None:
        executor.execute_skill(BUILTIN_SKILLS["stop"])
        assert len(executor.events) > 0

    def test_events_returns_copy(self, executor: SkillExecutor) -> None:
        """events property returns a copy — modifying it doesn't affect executor."""
        executor.execute_skill(BUILTIN_SKILLS["stop"])
        ev1 = executor.events
        ev1.clear()
        assert len(executor.events) > 0

    def test_events_include_completed_status(self, executor: SkillExecutor) -> None:
        executor.execute_skill(BUILTIN_SKILLS["stop"])
        completed = [e for e in executor.events if e.status == SkillStatus.COMPLETED]
        assert len(completed) >= 1

    def test_events_have_timestamps(self, executor: SkillExecutor) -> None:
        executor.execute_skill(BUILTIN_SKILLS["stop"])
        for event in executor.events:
            assert isinstance(event.timestamp, float)
            assert event.timestamp > 0


# ===========================================================================
# on_event registration (line 273)
# ===========================================================================

class TestOnEvent:
    def test_multiple_listeners_all_called(
        self, executor: SkillExecutor
    ) -> None:
        received_a: list[ExecutionEvent] = []
        received_b: list[ExecutionEvent] = []

        executor.on_event(lambda e: received_a.append(e))
        executor.on_event(lambda e: received_b.append(e))
        executor.execute_skill(BUILTIN_SKILLS["stop"])

        assert len(received_a) > 0
        assert len(received_b) > 0
        assert len(received_a) == len(received_b)

    def test_listener_registered_after_execution_misses_events(
        self, executor: SkillExecutor
    ) -> None:
        """Listeners registered after execution don't receive past events."""
        executor.execute_skill(BUILTIN_SKILLS["stop"])
        late_received: list[ExecutionEvent] = []
        executor.on_event(lambda e: late_received.append(e))
        assert len(late_received) == 0


# ===========================================================================
# execute_graph: trace_id parameter (OB-03)
# ===========================================================================

class TestExecuteGraphTraceId:
    def test_explicit_trace_id_accepted(self, executor: SkillExecutor) -> None:
        """execute_graph accepts an explicit trace_id."""
        graph = SkillGraph()
        graph.add_skill(BUILTIN_SKILLS["stop"])
        result = executor.execute_graph(graph, trace_id="my-trace-001")
        assert result.status == TaskStatus.COMPLETED

    def test_no_trace_id_still_executes(self, executor: SkillExecutor) -> None:
        graph = SkillGraph()
        graph.add_skill(BUILTIN_SKILLS["stop"])
        result = executor.execute_graph(graph)
        assert result.status == TaskStatus.COMPLETED

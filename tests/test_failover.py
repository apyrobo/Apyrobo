"""
Tests for apyrobo/skills/failover.py — v4.0.0 Production Hardening.

Covers:
- FailoverPolicy.default() correct defaults
- action_for() exact match, prefix match, fallback
- FailoverExecutor safe-state called on failure
- FailoverExecutor no safe-state on success
- Each SafeStateAction dispatches correctly
- Safe-state errors don't mask original failure
- HOME/DOCK fall back to stop when fn not provided
- last_safe_state_action tracking
- emit_event called with correct fields
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import TaskStatus
from apyrobo.skills.executor import SkillExecutor, SkillGraph
from apyrobo.skills.failover import FailoverExecutor, FailoverPolicy, SafeStateAction
from apyrobo.skills.skill import Skill, SkillStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(*skill_ids: str) -> SkillGraph:
    """Build a linear SkillGraph from a list of skill IDs."""
    graph = SkillGraph()
    prev: str | None = None
    for sid in skill_ids:
        skill = Skill.simple(sid)
        graph.add_skill(skill, depends_on=[prev] if prev else None)
        prev = sid
    return graph


def _mock_executor(robot: MagicMock, outcomes: dict[str, SkillStatus]) -> SkillExecutor:
    """Return a SkillExecutor whose execute_skill is mocked with given per-skill outcomes."""
    executor = SkillExecutor.__new__(SkillExecutor)
    executor._robot = robot
    executor._listeners = []
    executor._events = []

    def _execute_skill(skill: Skill, parameters=None) -> SkillStatus:
        return outcomes.get(skill.skill_id, SkillStatus.COMPLETED)

    executor.execute_skill = _execute_skill  # type: ignore[method-assign]

    def _get_params(skill_id: str) -> dict:
        return {}

    # We need execute_skill to be callable by FailoverExecutor
    return executor  # type: ignore[return-value]


def _make_robot() -> MagicMock:
    robot = MagicMock()
    robot.stop = MagicMock()
    robot.gripper_open = MagicMock()
    robot.get_position = MagicMock(return_value=(0.0, 0.0))
    return robot


# ---------------------------------------------------------------------------
# FailoverPolicy.default() — correct defaults
# ---------------------------------------------------------------------------

class TestFailoverPolicyDefaults:
    def test_pick_object_defaults_to_open_gripper(self) -> None:
        policy = FailoverPolicy.default()
        assert policy.action_for("pick_object") == SafeStateAction.OPEN_GRIPPER

    def test_place_object_defaults_to_open_gripper(self) -> None:
        policy = FailoverPolicy.default()
        assert policy.action_for("place_object") == SafeStateAction.OPEN_GRIPPER

    def test_grasp_defaults_to_open_gripper(self) -> None:
        policy = FailoverPolicy.default()
        assert policy.action_for("grasp") == SafeStateAction.OPEN_GRIPPER

    def test_report_status_defaults_to_none(self) -> None:
        policy = FailoverPolicy.default()
        assert policy.action_for("report_status") == SafeStateAction.NONE

    def test_get_pose_defaults_to_none(self) -> None:
        policy = FailoverPolicy.default()
        assert policy.action_for("get_pose") == SafeStateAction.NONE

    def test_capture_image_defaults_to_none(self) -> None:
        policy = FailoverPolicy.default()
        assert policy.action_for("capture_image") == SafeStateAction.NONE

    def test_unknown_skill_defaults_to_stop(self) -> None:
        policy = FailoverPolicy.default()
        assert policy.action_for("navigate_to") == SafeStateAction.STOP

    def test_default_action_is_stop(self) -> None:
        policy = FailoverPolicy.default()
        assert policy.default_action == SafeStateAction.STOP


# ---------------------------------------------------------------------------
# FailoverPolicy.action_for() — matching logic
# ---------------------------------------------------------------------------

class TestActionFor:
    def test_exact_match(self) -> None:
        policy = FailoverPolicy(
            default_action=SafeStateAction.STOP,
            per_skill={"my_skill": SafeStateAction.NONE},
        )
        assert policy.action_for("my_skill") == SafeStateAction.NONE

    def test_prefix_match_suffixed_id(self) -> None:
        """pick_object_0 should match 'pick_object' prefix."""
        policy = FailoverPolicy.default()
        assert policy.action_for("pick_object_0") == SafeStateAction.OPEN_GRIPPER

    def test_prefix_match_numeric_suffix(self) -> None:
        """grasp_42 should match 'grasp' prefix."""
        policy = FailoverPolicy.default()
        assert policy.action_for("grasp_42") == SafeStateAction.OPEN_GRIPPER

    def test_prefix_match_place_object_suffixed(self) -> None:
        policy = FailoverPolicy.default()
        assert policy.action_for("place_object_1") == SafeStateAction.OPEN_GRIPPER

    def test_fallback_to_default_when_no_match(self) -> None:
        policy = FailoverPolicy(
            default_action=SafeStateAction.HOME,
            per_skill={"pick_object": SafeStateAction.OPEN_GRIPPER},
        )
        assert policy.action_for("completely_unknown_skill") == SafeStateAction.HOME

    def test_exact_match_takes_priority_over_prefix(self) -> None:
        """An exact match wins even if a prefix would also match."""
        policy = FailoverPolicy(
            default_action=SafeStateAction.STOP,
            per_skill={
                "pick_object": SafeStateAction.OPEN_GRIPPER,
                "pick_object_0": SafeStateAction.NONE,  # exact match
            },
        )
        assert policy.action_for("pick_object_0") == SafeStateAction.NONE

    def test_longest_prefix_wins(self) -> None:
        """When multiple prefixes match, the longest one is chosen."""
        policy = FailoverPolicy(
            default_action=SafeStateAction.STOP,
            per_skill={
                "pick": SafeStateAction.NONE,
                "pick_object": SafeStateAction.OPEN_GRIPPER,
            },
        )
        assert policy.action_for("pick_object_special") == SafeStateAction.OPEN_GRIPPER


# ---------------------------------------------------------------------------
# FailoverExecutor — safe-state on failure
# ---------------------------------------------------------------------------

class TestFailoverExecutorSafeState:
    def _build(
        self,
        skill_outcomes: dict[str, SkillStatus],
        policy: FailoverPolicy | None = None,
        home_fn=None,
        dock_fn=None,
    ) -> tuple[FailoverExecutor, MagicMock]:
        robot = _make_robot()
        executor = _mock_executor(robot, skill_outcomes)
        failover = FailoverExecutor(executor, policy, home_fn=home_fn, dock_fn=dock_fn)
        return failover, robot

    # -- safe-state triggered on failure --

    def test_stop_action_calls_robot_stop(self) -> None:
        policy = FailoverPolicy(
            default_action=SafeStateAction.STOP,
            per_skill={"my_task": SafeStateAction.STOP},
        )
        failover, robot = self._build({"my_task": SkillStatus.FAILED}, policy)
        graph = _make_graph("my_task")
        result = failover.execute_graph(graph)
        assert result.status == TaskStatus.FAILED
        robot.stop.assert_called_once()
        robot.gripper_open.assert_not_called()

    def test_open_gripper_action_calls_gripper_open(self) -> None:
        policy = FailoverPolicy.default()
        failover, robot = self._build({"pick_object": SkillStatus.FAILED}, policy)
        graph = _make_graph("pick_object")
        result = failover.execute_graph(graph)
        assert result.status == TaskStatus.FAILED
        robot.gripper_open.assert_called_once()
        robot.stop.assert_not_called()

    def test_none_action_calls_nothing(self) -> None:
        policy = FailoverPolicy.default()
        failover, robot = self._build({"report_status": SkillStatus.FAILED}, policy)
        graph = _make_graph("report_status")
        result = failover.execute_graph(graph)
        assert result.status == TaskStatus.FAILED
        robot.stop.assert_not_called()
        robot.gripper_open.assert_not_called()

    def test_home_action_calls_home_fn(self) -> None:
        policy = FailoverPolicy(
            default_action=SafeStateAction.HOME,
            per_skill={},
        )
        home_fn = MagicMock()
        failover, robot = self._build({"nav_task": SkillStatus.FAILED}, policy, home_fn=home_fn)
        graph = _make_graph("nav_task")
        failover.execute_graph(graph)
        home_fn.assert_called_once()
        robot.stop.assert_not_called()

    def test_home_falls_back_to_stop_when_no_home_fn(self) -> None:
        policy = FailoverPolicy(
            default_action=SafeStateAction.HOME,
            per_skill={},
        )
        failover, robot = self._build({"nav_task": SkillStatus.FAILED}, policy)
        graph = _make_graph("nav_task")
        failover.execute_graph(graph)
        robot.stop.assert_called_once()

    def test_dock_action_calls_dock_fn(self) -> None:
        policy = FailoverPolicy(
            default_action=SafeStateAction.DOCK,
            per_skill={},
        )
        dock_fn = MagicMock()
        failover, robot = self._build({"nav_task": SkillStatus.FAILED}, policy, dock_fn=dock_fn)
        graph = _make_graph("nav_task")
        failover.execute_graph(graph)
        dock_fn.assert_called_once()
        robot.stop.assert_not_called()

    def test_dock_falls_back_to_stop_when_no_dock_fn(self) -> None:
        policy = FailoverPolicy(
            default_action=SafeStateAction.DOCK,
            per_skill={},
        )
        failover, robot = self._build({"nav_task": SkillStatus.FAILED}, policy)
        graph = _make_graph("nav_task")
        failover.execute_graph(graph)
        robot.stop.assert_called_once()

    # -- no safe-state on success --

    def test_no_safe_state_on_successful_execution(self) -> None:
        policy = FailoverPolicy.default()
        failover, robot = self._build({"pick_object": SkillStatus.COMPLETED}, policy)
        graph = _make_graph("pick_object")
        result = failover.execute_graph(graph)
        assert result.status == TaskStatus.COMPLETED
        robot.stop.assert_not_called()
        robot.gripper_open.assert_not_called()

    def test_multi_skill_success_no_safe_state(self) -> None:
        policy = FailoverPolicy.default()
        failover, robot = self._build(
            {"navigate_to": SkillStatus.COMPLETED, "pick_object": SkillStatus.COMPLETED},
            policy,
        )
        graph = _make_graph("navigate_to", "pick_object")
        result = failover.execute_graph(graph)
        assert result.status == TaskStatus.COMPLETED
        robot.stop.assert_not_called()

    # -- safe-state error doesn't mask failure --

    def test_safe_state_error_does_not_mask_failure(self) -> None:
        """If robot.stop() raises, the FAILED result is still returned."""
        policy = FailoverPolicy(default_action=SafeStateAction.STOP, per_skill={})
        robot = _make_robot()
        robot.stop.side_effect = RuntimeError("hardware fault")
        executor = _mock_executor(robot, {"bad_skill": SkillStatus.FAILED})
        failover = FailoverExecutor(executor, policy)
        graph = _make_graph("bad_skill")
        result = failover.execute_graph(graph)
        assert result.status == TaskStatus.FAILED  # still FAILED, not exception

    def test_gripper_open_error_does_not_mask_failure(self) -> None:
        policy = FailoverPolicy.default()
        robot = _make_robot()
        robot.gripper_open.side_effect = RuntimeError("gripper jammed")
        executor = _mock_executor(robot, {"pick_object": SkillStatus.FAILED})
        failover = FailoverExecutor(executor, policy)
        graph = _make_graph("pick_object")
        result = failover.execute_graph(graph)
        assert result.status == TaskStatus.FAILED


# ---------------------------------------------------------------------------
# FailoverExecutor — last_safe_state_action
# ---------------------------------------------------------------------------

class TestLastSafeStateAction:
    def test_none_before_any_execution(self) -> None:
        robot = _make_robot()
        executor = _mock_executor(robot, {})
        failover = FailoverExecutor(executor, FailoverPolicy.default())
        assert failover.last_safe_state_action is None

    def test_set_after_failure(self) -> None:
        robot = _make_robot()
        executor = _mock_executor(robot, {"pick_object": SkillStatus.FAILED})
        failover = FailoverExecutor(executor, FailoverPolicy.default())
        graph = _make_graph("pick_object")
        failover.execute_graph(graph)
        assert failover.last_safe_state_action == SafeStateAction.OPEN_GRIPPER

    def test_remains_none_after_success(self) -> None:
        robot = _make_robot()
        executor = _mock_executor(robot, {"report_status": SkillStatus.COMPLETED})
        failover = FailoverExecutor(executor, FailoverPolicy.default())
        graph = _make_graph("report_status")
        failover.execute_graph(graph)
        assert failover.last_safe_state_action is None

    def test_updated_on_each_failure(self) -> None:
        """second graph execution updates last_safe_state_action."""
        robot = _make_robot()
        executor = _mock_executor(robot, {
            "pick_object": SkillStatus.FAILED,
            "report_status": SkillStatus.FAILED,
        })
        failover = FailoverExecutor(executor, FailoverPolicy.default())

        graph1 = _make_graph("pick_object")
        failover.execute_graph(graph1)
        assert failover.last_safe_state_action == SafeStateAction.OPEN_GRIPPER

        graph2 = _make_graph("report_status")
        failover.execute_graph(graph2)
        assert failover.last_safe_state_action == SafeStateAction.NONE


# ---------------------------------------------------------------------------
# FailoverExecutor — emit_event on failover
# ---------------------------------------------------------------------------

class TestEmitEventOnFailover:
    def test_emit_event_called_with_correct_fields(self) -> None:
        robot = _make_robot()
        executor = _mock_executor(robot, {"pick_object": SkillStatus.FAILED})
        failover = FailoverExecutor(executor, FailoverPolicy.default())
        graph = _make_graph("pick_object")

        with patch("apyrobo.skills.failover.emit_event") as mock_emit:
            failover.execute_graph(graph)
            mock_emit.assert_called_once_with(
                "skill.failover",
                skill_id="pick_object",
                action=SafeStateAction.OPEN_GRIPPER.value,
                reason="skill_failed",
            )

    def test_emit_event_not_called_on_success(self) -> None:
        robot = _make_robot()
        executor = _mock_executor(robot, {"report_status": SkillStatus.COMPLETED})
        failover = FailoverExecutor(executor, FailoverPolicy.default())
        graph = _make_graph("report_status")

        with patch("apyrobo.skills.failover.emit_event") as mock_emit:
            failover.execute_graph(graph)
            mock_emit.assert_not_called()

    def test_emit_event_action_value_is_string(self) -> None:
        """action field in event must be the string value, not the enum."""
        robot = _make_robot()
        executor = _mock_executor(robot, {"navigate_to": SkillStatus.FAILED})
        policy = FailoverPolicy(default_action=SafeStateAction.STOP, per_skill={})
        failover = FailoverExecutor(executor, policy)
        graph = _make_graph("navigate_to")

        with patch("apyrobo.skills.failover.emit_event") as mock_emit:
            failover.execute_graph(graph)
            _, kwargs = mock_emit.call_args
            assert kwargs["action"] == "stop"
            assert isinstance(kwargs["action"], str)

    def test_emit_event_stop_action_for_unknown_skill(self) -> None:
        robot = _make_robot()
        executor = _mock_executor(robot, {"some_unknown": SkillStatus.FAILED})
        failover = FailoverExecutor(executor, FailoverPolicy.default())
        graph = _make_graph("some_unknown")

        with patch("apyrobo.skills.failover.emit_event") as mock_emit:
            failover.execute_graph(graph)
            _, kwargs = mock_emit.call_args
            assert kwargs["skill_id"] == "some_unknown"
            assert kwargs["action"] == "stop"
            assert kwargs["reason"] == "skill_failed"


# ---------------------------------------------------------------------------
# FailoverExecutor — TaskResult fields
# ---------------------------------------------------------------------------

class TestTaskResultFields:
    def test_failed_result_has_correct_steps(self) -> None:
        robot = _make_robot()
        # First skill succeeds, second fails
        call_count = {"n": 0}

        def execute_skill(skill, parameters=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return SkillStatus.COMPLETED
            return SkillStatus.FAILED

        executor = SkillExecutor.__new__(SkillExecutor)
        executor._robot = robot
        executor.execute_skill = execute_skill  # type: ignore[method-assign]

        failover = FailoverExecutor(executor, FailoverPolicy.default())
        graph = _make_graph("navigate_to", "pick_object")
        result = failover.execute_graph(graph)

        assert result.status == TaskStatus.FAILED
        assert result.steps_completed == 1
        assert result.steps_total == 2

    def test_successful_result_has_correct_steps(self) -> None:
        robot = _make_robot()
        executor = _mock_executor(robot, {
            "navigate_to": SkillStatus.COMPLETED,
            "pick_object": SkillStatus.COMPLETED,
        })
        failover = FailoverExecutor(executor, FailoverPolicy.default())
        graph = _make_graph("navigate_to", "pick_object")
        result = failover.execute_graph(graph)

        assert result.status == TaskStatus.COMPLETED
        assert result.steps_completed == 2
        assert result.steps_total == 2
        assert result.confidence == 1.0

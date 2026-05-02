"""Tests for FleetManager.handoff_task and execute_with_handoff."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from apyrobo.fleet.manager import FleetManager, RobotInfo
from apyrobo.core.schemas import TaskResult, TaskStatus, RecoveryAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fleet(*robot_ids: str) -> FleetManager:
    """Build a FleetManager with the given robots, all idle."""
    fm = FleetManager()
    for rid in robot_ids:
        fm.register(RobotInfo(robot_id=rid, capabilities=["move"]))
    return fm


def _task_result(status: TaskStatus, steps_completed: int = 0, steps_total: int = 1,
                 task_name: str = "test_task", error: str | None = None) -> TaskResult:
    return TaskResult(
        task_name=task_name,
        status=status,
        steps_completed=steps_completed,
        steps_total=steps_total,
        error=error,
        recovery_actions_taken=[],
    )


def _mock_robot(robot_id: str = "mock") -> MagicMock:
    robot = MagicMock()
    robot.robot_id = robot_id
    return robot


# ---------------------------------------------------------------------------
# handoff_task
# ---------------------------------------------------------------------------

class TestHandoffTask:
    def test_returns_new_robot_on_success(self):
        fm = _make_fleet("r1", "r2")
        fm._robots["r1"].status = "busy"
        result = _task_result(TaskStatus.FAILED, task_name="patrol")
        chosen = fm.handoff_task("r1", result)
        assert chosen == "r2"

    def test_failed_robot_released_back_to_idle(self):
        fm = _make_fleet("r1", "r2")
        fm._robots["r1"].status = "busy"
        fm._robots["r1"].current_task = "patrol"
        fm.handoff_task("r1", _task_result(TaskStatus.FAILED))
        assert fm._robots["r1"].status == "idle"
        assert fm._robots["r1"].current_task is None

    def test_chosen_robot_marked_busy(self):
        fm = _make_fleet("r1", "r2")
        fm._robots["r1"].status = "busy"
        fm.handoff_task("r1", _task_result(TaskStatus.FAILED, task_name="deliver"))
        assert fm._robots["r2"].status == "busy"
        assert fm._robots["r2"].current_task == "deliver"

    def test_returns_none_when_no_candidates(self):
        fm = _make_fleet("r1")
        fm._robots["r1"].status = "busy"
        result = fm.handoff_task("r1", _task_result(TaskStatus.FAILED))
        assert result is None

    def test_excludes_specified_robots(self):
        fm = _make_fleet("r1", "r2", "r3")
        fm._robots["r1"].status = "busy"
        result = fm.handoff_task("r1", _task_result(TaskStatus.FAILED), exclude_robots=["r2"])
        assert result == "r3"

    def test_excludes_failed_robot_implicitly(self):
        fm = _make_fleet("r1", "r2")
        fm._robots["r1"].status = "busy"
        fm._robots["r2"].status = "busy"
        result = fm.handoff_task("r1", _task_result(TaskStatus.FAILED))
        assert result is None  # r2 is busy, r1 is excluded

    def test_picks_least_recently_active(self):
        fm = _make_fleet("r1", "r2", "r3")
        fm._robots["r1"].status = "busy"
        # r3 has the oldest heartbeat → should be chosen first
        fm._robots["r2"].last_heartbeat = 1000.0
        fm._robots["r3"].last_heartbeat = 500.0
        chosen = fm.handoff_task("r1", _task_result(TaskStatus.FAILED))
        assert chosen == "r3"

    def test_handoff_with_missing_failed_robot_id(self):
        fm = _make_fleet("r1", "r2")
        result = fm.handoff_task("nonexistent", _task_result(TaskStatus.FAILED))
        # r1 and r2 are still idle — one should be chosen
        assert result in ("r1", "r2")

    def test_emits_observability_event(self, monkeypatch):
        events = []

        def fake_emit(event_type, **kwargs):
            events.append((event_type, kwargs))

        import apyrobo.fleet.manager as mgr_mod
        monkeypatch.setattr(
            "apyrobo.observability.emit_event", fake_emit, raising=False
        )

        fm = _make_fleet("r1", "r2")
        fm._robots["r1"].status = "busy"
        fm.handoff_task("r1", _task_result(TaskStatus.FAILED, task_name="sweep", error="timeout"))

        assert any(et == "task.handoff" for et, _ in events)
        evt = next(kw for et, kw in events if et == "task.handoff")
        assert evt["from_robot"] == "r1"
        assert evt["to_robot"] == "r2"
        assert evt["error"] == "timeout"


# ---------------------------------------------------------------------------
# execute_with_handoff
# ---------------------------------------------------------------------------

class TestExecuteWithHandoff:
    def _make_agent(self, side_effects: list) -> MagicMock:
        """
        Build a mock agent whose .execute() returns side_effects in sequence.
        side_effects: list of TaskResult or Exception.
        """
        agent = MagicMock()
        results = iter(side_effects)

        def _execute(task, robot, **kwargs):
            val = next(results)
            if isinstance(val, Exception):
                raise val
            return val

        agent.execute.side_effect = _execute
        return agent

    def _make_library(self) -> MagicMock:
        return MagicMock()

    def test_success_on_first_robot(self):
        fm = _make_fleet("r1", "r2")
        robots = {"r1": _mock_robot("r1"), "r2": _mock_robot("r2")}
        agent = self._make_agent([_task_result(TaskStatus.COMPLETED, 1, 1)])
        result, tried = fm.execute_with_handoff("patrol", self._make_library(), agent, robots)
        assert result.status == TaskStatus.COMPLETED
        assert tried == ["r1"]

    def test_handoff_on_failure(self):
        fm = _make_fleet("r1", "r2")
        robots = {"r1": _mock_robot("r1"), "r2": _mock_robot("r2")}
        agent = self._make_agent([
            _task_result(TaskStatus.FAILED, 0, 1, error="motor error"),
            _task_result(TaskStatus.COMPLETED, 1, 1),
        ])
        result, tried = fm.execute_with_handoff("patrol", self._make_library(), agent, robots)
        assert result.status == TaskStatus.COMPLETED
        assert tried == ["r1", "r2"]

    def test_returns_failure_when_all_robots_fail(self):
        fm = _make_fleet("r1", "r2", "r3")
        robots = {
            "r1": _mock_robot("r1"),
            "r2": _mock_robot("r2"),
            "r3": _mock_robot("r3"),
        }
        agent = self._make_agent([
            _task_result(TaskStatus.FAILED, 0, 1),
            _task_result(TaskStatus.FAILED, 0, 1),
            _task_result(TaskStatus.FAILED, 0, 1),
        ])
        result, tried = fm.execute_with_handoff(
            "patrol", self._make_library(), agent, robots, max_handoffs=2
        )
        assert result.status == TaskStatus.FAILED
        assert len(tried) == 3

    def test_max_handoffs_limits_attempts(self):
        fm = _make_fleet("r1", "r2", "r3")
        robots = {
            "r1": _mock_robot("r1"),
            "r2": _mock_robot("r2"),
            "r3": _mock_robot("r3"),
        }
        call_count = [0]

        def _execute(task, robot, **kwargs):
            call_count[0] += 1
            return _task_result(TaskStatus.FAILED)

        agent = MagicMock()
        agent.execute.side_effect = _execute

        fm.execute_with_handoff(
            "patrol", self._make_library(), agent, robots, max_handoffs=1
        )
        assert call_count[0] <= 2  # 1 original + 1 handoff

    def test_returns_failure_when_no_fleet_robot_in_robots_dict(self):
        fm = _make_fleet("r1")
        # robots dict is empty — r1 is registered but not in robots
        result, tried = fm.execute_with_handoff(
            "patrol", self._make_library(), MagicMock(), robots={}
        )
        assert result.status == TaskStatus.FAILED
        assert tried == []

    def test_returns_ordered_robot_ids(self):
        fm = _make_fleet("r1", "r2", "r3")
        robots = {
            "r1": _mock_robot("r1"),
            "r2": _mock_robot("r2"),
            "r3": _mock_robot("r3"),
        }
        # Give r1 oldest heartbeat so it's picked first
        fm._robots["r1"].last_heartbeat = 100.0
        fm._robots["r2"].last_heartbeat = 200.0
        fm._robots["r3"].last_heartbeat = 300.0

        agent = self._make_agent([
            _task_result(TaskStatus.FAILED),
            _task_result(TaskStatus.COMPLETED, 1, 1),
        ])
        _, tried = fm.execute_with_handoff("patrol", self._make_library(), agent, robots)
        assert tried[0] == "r1"
        assert len(tried) == 2

    def test_robot_released_after_success(self):
        fm = _make_fleet("r1")
        robots = {"r1": _mock_robot("r1")}
        agent = self._make_agent([_task_result(TaskStatus.COMPLETED, 1, 1)])
        fm.execute_with_handoff("patrol", self._make_library(), agent, robots)
        assert fm._robots["r1"].status == "idle"

    def test_robots_not_in_robots_dict_are_skipped_for_handoff(self):
        fm = _make_fleet("r1", "r2")
        # Only r1 in robots dict, r2 registered but not available for execution
        robots = {"r1": _mock_robot("r1")}
        agent = self._make_agent([
            _task_result(TaskStatus.FAILED),
            _task_result(TaskStatus.COMPLETED, 1, 1),  # won't be called
        ])
        result, tried = fm.execute_with_handoff("patrol", self._make_library(), agent, robots)
        # r2 is found by handoff but not in robots dict → no second attempt
        assert tried == ["r1"]
        assert result.status == TaskStatus.FAILED

    def test_default_max_handoffs_is_two(self):
        fm = _make_fleet("r1", "r2", "r3")
        robots = {rid: _mock_robot(rid) for rid in ("r1", "r2", "r3")}
        call_count = [0]

        def _execute(task, robot, **kwargs):
            call_count[0] += 1
            return _task_result(TaskStatus.FAILED)

        agent = MagicMock()
        agent.execute.side_effect = _execute

        fm.execute_with_handoff("patrol", self._make_library(), agent, robots)
        assert call_count[0] <= 3  # default max_handoffs=2 → ≤3 total attempts

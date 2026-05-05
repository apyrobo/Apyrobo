"""Tests for sim-to-real transfer (ST-01)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import apyrobo.skills.builtins  # noqa: F401
from apyrobo import Robot
from apyrobo.core.schemas import TaskResult, TaskStatus
from apyrobo.skills.executor import SkillExecutor
from apyrobo.skills.simtoreal import MockSimToRealTransfer, SimToRealTransfer, SimulationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _success_result(steps=1):
    return TaskResult(
        task_name="t", status=TaskStatus.COMPLETED,
        steps_completed=steps, steps_total=steps,
    )


def _failed_result(steps=0, total=1, error="Skill 'stop_0' failed"):
    return TaskResult(
        task_name="t", status=TaskStatus.FAILED,
        steps_completed=steps, steps_total=total, error=error,
    )


def _mock_graph(n=1):
    """Return a SkillGraph with n stop skills."""
    from apyrobo.skills.executor import SkillGraph
    from apyrobo.skills.skill import Skill
    g = SkillGraph()
    prev = None
    for i in range(n):
        s = Skill(skill_id=f"stop_{i}", name="stop", description="")
        g.add_skill(s, depends_on=[prev] if prev else None)
        prev = f"stop_{i}"
    return g


# ---------------------------------------------------------------------------
# SimulationResult
# ---------------------------------------------------------------------------

class TestSimulationResult:
    def test_fields_stored(self):
        sr = SimulationResult(
            success=True,
            steps_completed=3,
            steps_total=3,
            failures=[],
            duration_s=1.5,
            robot_final_position={"x": 1.0, "y": 2.0},
        )
        assert sr.success is True
        assert sr.steps_completed == 3
        assert sr.duration_s == 1.5
        assert sr.robot_final_position == {"x": 1.0, "y": 2.0}

    def test_defaults(self):
        sr = SimulationResult(success=False, steps_completed=0, steps_total=1)
        assert sr.failures == []
        assert sr.duration_s == 0.0
        assert sr.robot_final_position == {}

    def test_failures_list(self):
        sr = SimulationResult(success=False, steps_completed=0, steps_total=2,
                              failures=["step1 failed", "step2 failed"])
        assert len(sr.failures) == 2


# ---------------------------------------------------------------------------
# MockSimToRealTransfer
# ---------------------------------------------------------------------------

class TestMockSimToRealTransferDefaults:
    def test_default_sim_result_success(self):
        m = MockSimToRealTransfer()
        r = m.validate("plan")
        assert r.success is True

    def test_validate_records_call(self):
        m = MockSimToRealTransfer()
        m.validate("plan")
        assert len(m.validate_calls) == 1

    def test_deploy_records_call(self):
        m = MockSimToRealTransfer()
        m.deploy("plan")
        assert len(m.deploy_calls) == 1

    def test_deploy_success_by_default(self):
        m = MockSimToRealTransfer()
        assert m.deploy("plan") is True

    def test_deploy_can_be_set_false(self):
        m = MockSimToRealTransfer(deploy_success=False)
        assert m.deploy("plan") is False

    def test_set_sim_result_overrides(self):
        m = MockSimToRealTransfer()
        m.set_sim_result(SimulationResult(success=False, steps_completed=0, steps_total=1))
        r = m.validate("plan")
        assert r.success is False


class TestMockSimToRealTransferRun:
    def test_run_returns_tuple(self):
        m = MockSimToRealTransfer()
        result, deployed = m.run("plan")
        assert isinstance(result, SimulationResult)
        assert isinstance(deployed, bool)

    def test_run_no_auto_deploy(self):
        m = MockSimToRealTransfer()
        _, deployed = m.run("plan", auto_deploy=False)
        assert deployed is False
        assert len(m.deploy_calls) == 0

    def test_run_auto_deploy_on_success(self):
        m = MockSimToRealTransfer()
        _, deployed = m.run("plan", auto_deploy=True)
        assert deployed is True
        assert len(m.deploy_calls) == 1

    def test_run_auto_deploy_skipped_on_failure(self):
        m = MockSimToRealTransfer(
            sim_result=SimulationResult(success=False, steps_completed=0, steps_total=1)
        )
        _, deployed = m.run("plan", auto_deploy=True)
        assert deployed is False
        assert len(m.deploy_calls) == 0

    def test_run_records_validate_call(self):
        m = MockSimToRealTransfer()
        m.run("plan")
        assert len(m.validate_calls) == 1

    def test_run_passes_agent(self):
        m = MockSimToRealTransfer()
        agent = MagicMock()
        m.run("plan", agent=agent)
        assert m.validate_calls[0]["agent"] is agent


# ---------------------------------------------------------------------------
# SimToRealTransfer — validate() with real mock robot
# ---------------------------------------------------------------------------

class TestSimToRealTransferValidate:
    def test_validate_success(self):
        tr = SimToRealTransfer(sim_adapter_uri="mock://turtlebot4")
        graph = _mock_graph()
        with patch.object(SkillExecutor, "execute_graph", return_value=_success_result()):
            result = tr.validate(graph)
        assert result.success is True
        assert result.steps_completed == 1

    def test_validate_failure(self):
        tr = SimToRealTransfer(sim_adapter_uri="mock://turtlebot4")
        graph = _mock_graph()
        with patch.object(SkillExecutor, "execute_graph", return_value=_failed_result()):
            result = tr.validate(graph)
        assert result.success is False

    def test_validate_failure_records_error(self):
        tr = SimToRealTransfer(sim_adapter_uri="mock://turtlebot4")
        graph = _mock_graph()
        with patch.object(SkillExecutor, "execute_graph",
                          return_value=_failed_result(error="boom")):
            result = tr.validate(graph)
        assert "boom" in result.failures

    def test_validate_exception_returns_failed_result(self):
        tr = SimToRealTransfer(sim_adapter_uri="mock://turtlebot4")
        graph = _mock_graph()
        with patch.object(SkillExecutor, "execute_graph", side_effect=RuntimeError("crash")):
            result = tr.validate(graph)
        assert result.success is False
        assert len(result.failures) == 1


# ---------------------------------------------------------------------------
# SimToRealTransfer — deploy()
# ---------------------------------------------------------------------------

class TestSimToRealTransferDeploy:
    def test_deploy_no_real_uri_returns_false(self):
        tr = SimToRealTransfer(sim_adapter_uri="mock://turtlebot4", real_adapter_uri=None)
        assert tr.deploy(_mock_graph()) is False

    def test_deploy_success(self):
        tr = SimToRealTransfer(
            sim_adapter_uri="mock://turtlebot4",
            real_adapter_uri="mock://turtlebot4",
        )
        graph = _mock_graph()
        with patch.object(SkillExecutor, "execute_graph", return_value=_success_result()):
            deployed = tr.deploy(graph)
        assert deployed is True

    def test_deploy_failure_returns_false(self):
        tr = SimToRealTransfer(
            sim_adapter_uri="mock://turtlebot4",
            real_adapter_uri="mock://turtlebot4",
        )
        graph = _mock_graph()
        with patch.object(SkillExecutor, "execute_graph", return_value=_failed_result()):
            deployed = tr.deploy(graph)
        assert deployed is False

    def test_deploy_exception_returns_false(self):
        tr = SimToRealTransfer(
            sim_adapter_uri="mock://turtlebot4",
            real_adapter_uri="mock://turtlebot4",
        )
        with patch.object(SkillExecutor, "execute_graph", side_effect=RuntimeError("crash")):
            deployed = tr.deploy(_mock_graph())
        assert deployed is False


# ---------------------------------------------------------------------------
# SimToRealTransfer — run()
# ---------------------------------------------------------------------------

class TestSimToRealTransferRun:
    def test_run_returns_sim_result_and_bool(self):
        tr = SimToRealTransfer(sim_adapter_uri="mock://turtlebot4")
        with patch.object(SkillExecutor, "execute_graph", return_value=_success_result()):
            sim_result, deployed = tr.run(_mock_graph())
        assert isinstance(sim_result, SimulationResult)
        assert isinstance(deployed, bool)

    def test_run_no_auto_deploy(self):
        tr = SimToRealTransfer(
            sim_adapter_uri="mock://turtlebot4",
            real_adapter_uri="mock://turtlebot4",
        )
        with patch.object(SkillExecutor, "execute_graph", return_value=_success_result()):
            _, deployed = tr.run(_mock_graph(), auto_deploy=False)
        assert deployed is False

    def test_run_auto_deploy_on_success(self):
        tr = SimToRealTransfer(
            sim_adapter_uri="mock://turtlebot4",
            real_adapter_uri="mock://turtlebot4",
        )
        with patch.object(SkillExecutor, "execute_graph", return_value=_success_result()):
            _, deployed = tr.run(_mock_graph(), auto_deploy=True)
        assert deployed is True

    def test_run_auto_deploy_skipped_on_sim_failure(self):
        tr = SimToRealTransfer(
            sim_adapter_uri="mock://turtlebot4",
            real_adapter_uri="mock://turtlebot4",
        )
        with patch.object(SkillExecutor, "execute_graph", return_value=_failed_result()):
            _, deployed = tr.run(_mock_graph(), auto_deploy=True)
        assert deployed is False

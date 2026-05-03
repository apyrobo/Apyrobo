"""Tests for the LLM replanning loop (RP-01)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

import apyrobo.skills.builtins  # noqa: F401 — register builtin handlers

from apyrobo import Robot
from apyrobo.core.schemas import TaskResult, TaskStatus
from apyrobo.skills.agent import Agent, LLMProvider
from apyrobo.skills.executor import SkillExecutor
from apyrobo.skills.replanner import MockReplanner, ReplanContext, Replanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _robot():
    return Robot.discover("mock://turtlebot4")


class FakeLLMProvider(LLMProvider):
    """LLMProvider subclass that returns deterministic plans without litellm."""

    def __init__(self, plans: list[list[dict]]) -> None:
        self._plans = plans
        self._call_count = 0

    def plan(self, task, available_skills, capabilities, **kwargs):
        plan = self._plans[min(self._call_count, len(self._plans) - 1)]
        self._call_count += 1
        return plan


def _failed_result(**kwargs) -> TaskResult:
    defaults = dict(
        task_name="test_task",
        status=TaskStatus.FAILED,
        steps_completed=0,
        steps_total=1,
        error="Skill 'navigate_to_0' failed",
    )
    defaults.update(kwargs)
    return TaskResult(**defaults)


def _success_result(**kwargs) -> TaskResult:
    defaults = dict(
        task_name="test_task",
        status=TaskStatus.COMPLETED,
        steps_completed=1,
        steps_total=1,
    )
    defaults.update(kwargs)
    return TaskResult(**defaults)


# ---------------------------------------------------------------------------
# ReplanContext
# ---------------------------------------------------------------------------

class TestReplanContext:
    def test_fields_accessible(self):
        ctx = ReplanContext(
            task="go to lab",
            failed_skill_id="navigate_to",
            completed_steps=[],
            replan_attempt=1,
            available_skills=[{"skill_id": "navigate_to"}],
            capabilities=["navigation"],
            error="timed out",
        )
        assert ctx.task == "go to lab"
        assert ctx.failed_skill_id == "navigate_to"
        assert ctx.replan_attempt == 1
        assert ctx.error == "timed out"

    def test_error_defaults_to_empty_string(self):
        ctx = ReplanContext(
            task="t",
            failed_skill_id="s",
            completed_steps=[],
            replan_attempt=1,
            available_skills=[],
            capabilities=[],
        )
        assert ctx.error == ""

    def test_completed_steps_stored(self):
        steps = [{"skill_id": "stop", "parameters": {}}]
        ctx = ReplanContext(
            task="t", failed_skill_id="s", completed_steps=steps,
            replan_attempt=1, available_skills=[], capabilities=[],
        )
        assert ctx.completed_steps == steps


# ---------------------------------------------------------------------------
# MockReplanner
# ---------------------------------------------------------------------------

class TestMockReplanner:
    def test_returns_fixed_plan(self):
        plan = [{"skill_id": "stop", "parameters": {}}]
        mr = MockReplanner(plan)
        ctx = ReplanContext(
            task="t", failed_skill_id="s", completed_steps=[],
            replan_attempt=1, available_skills=[], capabilities=[],
        )
        assert mr.replan(ctx) == plan

    def test_records_calls(self):
        mr = MockReplanner([])
        ctx = ReplanContext(
            task="task_a", failed_skill_id="s", completed_steps=[],
            replan_attempt=1, available_skills=[], capabilities=[],
        )
        mr.replan(ctx)
        mr.replan(ctx)
        assert len(mr.calls) == 2

    def test_returns_copy_not_reference(self):
        plan = [{"skill_id": "stop", "parameters": {}}]
        mr = MockReplanner(plan)
        ctx = ReplanContext(
            task="t", failed_skill_id="s", completed_steps=[],
            replan_attempt=1, available_skills=[], capabilities=[],
        )
        result = mr.replan(ctx)
        result.append({"skill_id": "extra", "parameters": {}})
        assert len(mr.replan(ctx)) == 1  # original plan unchanged


# ---------------------------------------------------------------------------
# Replanner
# ---------------------------------------------------------------------------

class TestReplanner:
    def test_calls_provider_plan(self):
        mock_provider = MagicMock()
        mock_provider.plan.return_value = [{"skill_id": "stop", "parameters": {}}]
        replanner = Replanner(mock_provider)
        ctx = ReplanContext(
            task="go to room A",
            failed_skill_id="navigate_to",
            completed_steps=[],
            replan_attempt=1,
            available_skills=[{"skill_id": "stop"}],
            capabilities=["navigation"],
            error="timeout",
        )
        result = replanner.replan(ctx)
        mock_provider.plan.assert_called_once()
        assert result == [{"skill_id": "stop", "parameters": {}}]

    def test_augments_task_with_failure_context(self):
        mock_provider = MagicMock()
        mock_provider.plan.return_value = []
        replanner = Replanner(mock_provider)
        ctx = ReplanContext(
            task="deliver package",
            failed_skill_id="pick_object",
            completed_steps=[{"skill_id": "navigate_to_0", "parameters": {}}],
            replan_attempt=2,
            available_skills=[],
            capabilities=[],
            error="gripper not available",
        )
        replanner.replan(ctx)
        augmented = mock_provider.plan.call_args[0][0]
        assert "deliver package" in augmented
        assert "pick_object" in augmented
        assert "gripper not available" in augmented
        assert "Replanning attempt 2" in augmented

    def test_completed_steps_in_augmented_task(self):
        mock_provider = MagicMock()
        mock_provider.plan.return_value = []
        replanner = Replanner(mock_provider)
        ctx = ReplanContext(
            task="t",
            failed_skill_id="s",
            completed_steps=[{"skill_id": "navigate_to_0", "parameters": {}}],
            replan_attempt=1,
            available_skills=[],
            capabilities=[],
        )
        replanner.replan(ctx)
        augmented = mock_provider.plan.call_args[0][0]
        assert "navigate_to_0" in augmented

    def test_passes_available_skills_and_capabilities(self):
        mock_provider = MagicMock()
        mock_provider.plan.return_value = []
        replanner = Replanner(mock_provider)
        skills = [{"skill_id": "stop"}]
        caps = ["navigation", "manipulation"]
        ctx = ReplanContext(
            task="t", failed_skill_id="s", completed_steps=[],
            replan_attempt=1, available_skills=skills, capabilities=caps,
        )
        replanner.replan(ctx)
        _, available, capabilities = mock_provider.plan.call_args[0]
        assert available == skills
        assert capabilities == caps

    def test_no_completed_steps_shows_none_in_task(self):
        mock_provider = MagicMock()
        mock_provider.plan.return_value = []
        replanner = Replanner(mock_provider)
        ctx = ReplanContext(
            task="t", failed_skill_id="s", completed_steps=[],
            replan_attempt=1, available_skills=[], capabilities=[],
        )
        replanner.replan(ctx)
        augmented = mock_provider.plan.call_args[0][0]
        assert "none" in augmented.lower()


# ---------------------------------------------------------------------------
# Agent.execute() replanning integration
# ---------------------------------------------------------------------------

class TestAgentReplanning:
    """Test that Agent.execute() triggers replanning correctly."""

    def _make_agent_with_llm(self, plans):
        """Return an Agent whose provider is a FakeLLMProvider."""
        agent = Agent(provider="rule")  # start rule-based
        agent._provider = FakeLLMProvider(plans)
        return agent

    def test_replans_on_failure(self):
        """Agent replans and succeeds on second attempt."""
        robot = _robot()
        # First plan: skill that always fails; second plan: stop (always succeeds)
        agent = self._make_agent_with_llm([
            [{"skill_id": "navigate_to", "parameters": {"x": 1.0, "y": 1.0}}],
            [{"skill_id": "stop", "parameters": {}}],
        ])
        with patch.object(
            SkillExecutor, "execute_graph",
            side_effect=[_failed_result(), _success_result()],
        ):
            result = agent.execute("test task", robot, replanning=True, max_replans=2)
        assert result.status == TaskStatus.COMPLETED

    def test_no_replan_when_disabled(self):
        """replanning=False never triggers a second plan call."""
        robot = _robot()
        agent = self._make_agent_with_llm([
            [{"skill_id": "stop", "parameters": {}}],
        ])
        with patch.object(
            SkillExecutor, "execute_graph",
            return_value=_failed_result(),
        ):
            result = agent.execute("test task", robot, replanning=False)
        assert result.status == TaskStatus.FAILED
        # Provider.plan called exactly once (initial plan)
        assert agent._provider._call_count == 1

    def test_rule_based_agent_does_not_replan(self):
        """Rule-based provider is not an LLMProvider — no replanning."""
        robot = _robot()
        agent = Agent(provider="rule")  # RuleBasedProvider
        with patch.object(
            SkillExecutor, "execute_graph",
            return_value=_failed_result(),
        ):
            result = agent.execute("stop the robot", robot, replanning=True)
        assert result.status == TaskStatus.FAILED

    def test_max_replans_respected(self):
        """Agent gives up after max_replans attempts."""
        robot = _robot()
        agent = self._make_agent_with_llm([
            [{"skill_id": "stop", "parameters": {}}],  # initial (and all replans)
        ])
        with patch.object(
            SkillExecutor, "execute_graph",
            return_value=_failed_result(),
        ) as mock_exec:
            result = agent.execute(
                "test task", robot, replanning=True, max_replans=2
            )
        # Initial + 2 replans = 3 execute_graph calls
        assert mock_exec.call_count == 3
        assert result.status == TaskStatus.FAILED

    def test_no_replan_when_success(self):
        """Successful execution does not trigger replanning."""
        robot = _robot()
        agent = self._make_agent_with_llm([
            [{"skill_id": "stop", "parameters": {}}],
        ])
        with patch.object(
            SkillExecutor, "execute_graph",
            return_value=_success_result(),
        ) as mock_exec:
            result = agent.execute("test task", robot, replanning=True)
        assert mock_exec.call_count == 1
        assert result.status == TaskStatus.COMPLETED

    def test_task_replanned_event_emitted(self):
        """task.replanned observability event fires on replan."""
        robot = _robot()
        agent = self._make_agent_with_llm([
            [{"skill_id": "stop", "parameters": {}}],
            [{"skill_id": "stop", "parameters": {}}],
        ])
        emitted = []

        def _capture(*args, **kwargs):
            emitted.append(args[0])

        with patch("apyrobo.skills.agent.emit_event", side_effect=_capture):
            with patch.object(
                SkillExecutor, "execute_graph",
                side_effect=[_failed_result(), _success_result()],
            ):
                agent.execute("test task", robot, replanning=True)
        # Find the task.replanned event among all emitted events
        replanned = [e for e in emitted if e == "task.replanned"]
        assert len(replanned) == 1

    def test_task_replanned_event_not_emitted_on_success(self):
        """No task.replanned event when plan succeeds first time."""
        robot = _robot()
        agent = self._make_agent_with_llm([
            [{"skill_id": "stop", "parameters": {}}],
        ])
        emitted = []

        def _capture(*args, **kwargs):
            emitted.append(args[0])

        with patch("apyrobo.skills.agent.emit_event", side_effect=_capture):
            with patch.object(
                SkillExecutor, "execute_graph",
                return_value=_success_result(),
            ):
                agent.execute("test task", robot, replanning=True)
        assert "task.replanned" not in emitted

    def test_replan_context_has_correct_failed_skill(self):
        """ReplanContext.failed_skill_id strips the _N suffix."""
        robot = _robot()
        agent = self._make_agent_with_llm([
            [{"skill_id": "navigate_to", "parameters": {"x": 0.0, "y": 0.0}}],
            [{"skill_id": "stop", "parameters": {}}],
        ])
        captured_contexts = []

        original_replan = Replanner.replan

        def spy_replan(self_inner, context):
            captured_contexts.append(context)
            return [{"skill_id": "stop", "parameters": {}}]

        with patch.object(Replanner, "replan", spy_replan):
            with patch.object(
                SkillExecutor, "execute_graph",
                side_effect=[
                    _failed_result(error="Skill 'navigate_to_0' failed"),
                    _success_result(),
                ],
            ):
                agent.execute("test task", robot, replanning=True)

        assert len(captured_contexts) == 1
        assert captured_contexts[0].failed_skill_id == "navigate_to"

    def test_replan_context_has_correct_attempt_number(self):
        robot = _robot()
        agent = self._make_agent_with_llm([
            [{"skill_id": "stop", "parameters": {}}],
            [{"skill_id": "stop", "parameters": {}}],
        ])
        captured_contexts = []

        def spy_replan(self_inner, context):
            captured_contexts.append(context)
            return [{"skill_id": "stop", "parameters": {}}]

        with patch.object(Replanner, "replan", spy_replan):
            with patch.object(
                SkillExecutor, "execute_graph",
                side_effect=[_failed_result(), _success_result()],
            ):
                agent.execute("test task", robot, replanning=True, max_replans=2)

        assert captured_contexts[0].replan_attempt == 1

    def test_replanner_exception_does_not_crash_agent(self):
        """If Replanner.replan() raises, agent returns the failed result."""
        robot = _robot()
        agent = self._make_agent_with_llm([
            [{"skill_id": "stop", "parameters": {}}],
        ])
        with patch.object(Replanner, "replan", side_effect=RuntimeError("LLM down")):
            with patch.object(
                SkillExecutor, "execute_graph",
                return_value=_failed_result(),
            ):
                result = agent.execute("test task", robot, replanning=True)
        assert result.status == TaskStatus.FAILED

    def test_empty_replan_stops_loop(self):
        """If replanner returns an empty plan, agent stops and returns failed."""
        robot = _robot()
        agent = self._make_agent_with_llm([
            [{"skill_id": "stop", "parameters": {}}],
        ])
        with patch.object(Replanner, "replan", return_value=[]):
            with patch.object(
                SkillExecutor, "execute_graph",
                return_value=_failed_result(),
            ):
                result = agent.execute("test task", robot, replanning=True)
        assert result.status == TaskStatus.FAILED

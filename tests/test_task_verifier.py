"""Tests for VLM-based task verification (VF-01)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import apyrobo.skills.builtins  # noqa: F401

from apyrobo import Robot
from apyrobo.core.schemas import TaskStatus
from apyrobo.inference.vlm import MockVLMAdapter
from apyrobo.skills.agent import Agent, LLMProvider
from apyrobo.skills.executor import SkillExecutor
from apyrobo.skills.verifier import (
    MockTaskVerifier,
    TaskVerifier,
    VerificationResult,
    _strip_index_suffix,
)

SAMPLE_IMAGE = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # fake JPEG header


def _robot():
    return Robot.discover("mock://turtlebot4")


def _success_result(**kw):
    from apyrobo.core.schemas import TaskResult
    defaults = dict(task_name="t", status=TaskStatus.COMPLETED, steps_completed=1, steps_total=1)
    defaults.update(kw)
    return TaskResult(**defaults)


def _failed_result(**kw):
    from apyrobo.core.schemas import TaskResult
    defaults = dict(
        task_name="t", status=TaskStatus.FAILED, steps_completed=0, steps_total=1,
        error="Skill 'stop_0' failed",
    )
    defaults.update(kw)
    return TaskResult(**defaults)


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------

class TestVerificationResult:
    def test_fields_stored(self):
        vr = VerificationResult(
            skill_id="navigate_to",
            verified=True,
            confidence=0.95,
            observation="robot at goal",
        )
        assert vr.skill_id == "navigate_to"
        assert vr.verified is True
        assert vr.confidence == 0.95
        assert vr.observation == "robot at goal"

    def test_suggested_action_defaults_none(self):
        vr = VerificationResult("s", True, 0.9, "ok")
        assert vr.suggested_action is None

    def test_suggested_action_set(self):
        vr = VerificationResult("s", False, 0.8, "nope", "try again")
        assert vr.suggested_action == "try again"


# ---------------------------------------------------------------------------
# _strip_index_suffix helper
# ---------------------------------------------------------------------------

class TestStripIndexSuffix:
    def test_strips_digit(self):
        assert _strip_index_suffix("navigate_to_0") == "navigate_to"

    def test_strips_multi_digit(self):
        assert _strip_index_suffix("navigate_to_12") == "navigate_to"

    def test_no_suffix_unchanged(self):
        assert _strip_index_suffix("navigate_to") == "navigate_to"

    def test_word_suffix_unchanged(self):
        assert _strip_index_suffix("pick_object") == "pick_object"


# ---------------------------------------------------------------------------
# TaskVerifier — default outcomes
# ---------------------------------------------------------------------------

class TestDefaultOutcomes:
    def _v(self):
        return TaskVerifier(MockVLMAdapter())

    def test_navigate_outcome(self):
        out = self._v()._default_expected_outcome("navigate_to")
        assert "target position" in out

    def test_pick_outcome(self):
        out = self._v()._default_expected_outcome("pick_object")
        assert "holding" in out

    def test_place_outcome(self):
        out = self._v()._default_expected_outcome("place_object")
        assert "placed" in out.lower()

    def test_inspect_outcome(self):
        out = self._v()._default_expected_outcome("inspect_room")
        assert "view" in out.lower()

    def test_unknown_skill_fallback(self):
        out = self._v()._default_expected_outcome("unknown_skill_xyz")
        assert "unknown_skill_xyz" in out

    def test_strips_index_suffix_for_lookup(self):
        out = self._v()._default_expected_outcome("navigate_to_3")
        assert "target position" in out


# ---------------------------------------------------------------------------
# TaskVerifier — prompt generation
# ---------------------------------------------------------------------------

class TestGenerateVerificationPrompt:
    def _v(self):
        return TaskVerifier(MockVLMAdapter())

    def test_includes_skill_id(self):
        prompt = self._v().generate_verification_prompt("navigate_to", "robot at goal")
        assert "navigate_to" in prompt

    def test_strips_index_suffix_in_prompt(self):
        prompt = self._v().generate_verification_prompt("navigate_to_0", "robot at goal")
        assert "navigate_to_0" not in prompt
        assert "navigate_to" in prompt

    def test_includes_expected_outcome(self):
        prompt = self._v().generate_verification_prompt("stop", "robot has stopped")
        assert "robot has stopped" in prompt

    def test_asks_for_json(self):
        prompt = self._v().generate_verification_prompt("stop", "robot has stopped")
        assert "json" in prompt.lower() or "{" in prompt


# ---------------------------------------------------------------------------
# TaskVerifier — verify_skill
# ---------------------------------------------------------------------------

class TestVerifySkill:
    def test_calls_vlm_answer_question(self):
        mock_vlm = MockVLMAdapter(answers={})
        v = TaskVerifier(mock_vlm)
        v.verify_skill("stop", image=SAMPLE_IMAGE)
        assert len(mock_vlm.question_calls) == 1

    def test_parses_json_verified_true(self):
        response = json.dumps({
            "verified": True, "confidence": 0.92,
            "observation": "robot stopped", "suggested_action": None,
        })
        mock_vlm = MagicMock()
        mock_vlm.answer_question.return_value = response
        vr = TaskVerifier(mock_vlm).verify_skill("stop", image=SAMPLE_IMAGE)
        assert vr.verified is True
        assert abs(vr.confidence - 0.92) < 0.001
        assert vr.observation == "robot stopped"

    def test_parses_json_verified_false(self):
        response = json.dumps({
            "verified": False, "confidence": 0.75,
            "observation": "robot not at goal", "suggested_action": "retry navigate",
        })
        mock_vlm = MagicMock()
        mock_vlm.answer_question.return_value = response
        vr = TaskVerifier(mock_vlm).verify_skill("navigate_to", image=SAMPLE_IMAGE)
        assert vr.verified is False
        assert vr.suggested_action == "retry navigate"

    def test_fallback_keyword_yes(self):
        mock_vlm = MagicMock()
        mock_vlm.answer_question.return_value = "Yes, the robot has reached the goal."
        vr = TaskVerifier(mock_vlm).verify_skill("navigate_to", image=SAMPLE_IMAGE)
        assert vr.verified is True

    def test_fallback_keyword_no(self):
        mock_vlm = MagicMock()
        mock_vlm.answer_question.return_value = "No, the object is not visible."
        vr = TaskVerifier(mock_vlm).verify_skill("pick_object", image=SAMPLE_IMAGE)
        assert vr.verified is False

    def test_no_image_returns_unverified_zero_confidence(self):
        mock_vlm = MagicMock()
        vr = TaskVerifier(mock_vlm).verify_skill("stop", image=None, robot=None)
        assert vr.verified is False
        assert vr.confidence == 0.0
        assert "No image" in vr.observation

    def test_no_image_does_not_call_vlm(self):
        mock_vlm = MagicMock()
        TaskVerifier(mock_vlm).verify_skill("stop", image=None, robot=None)
        mock_vlm.answer_question.assert_not_called()

    def test_uses_capture_image_from_robot(self):
        fake_robot = MagicMock()
        fake_robot.capture_image.return_value = SAMPLE_IMAGE
        response = json.dumps({"verified": True, "confidence": 0.9, "observation": "ok"})
        mock_vlm = MagicMock()
        mock_vlm.answer_question.return_value = response
        vr = TaskVerifier(mock_vlm).verify_skill("stop", image=None, robot=fake_robot)
        fake_robot.capture_image.assert_called_once()
        assert vr.verified is True

    def test_robot_without_capture_image_graceful(self):
        fake_robot = MagicMock(spec=[])  # no capture_image
        vr = TaskVerifier(MagicMock()).verify_skill("stop", image=None, robot=fake_robot)
        assert vr.confidence == 0.0

    def test_custom_expected_outcome_used_in_prompt(self):
        captured = []
        mock_vlm = MagicMock()
        mock_vlm.answer_question.side_effect = lambda img, q: (
            captured.append(q) or '{"verified": true, "confidence": 0.9, "observation": "ok"}'
        )
        TaskVerifier(mock_vlm).verify_skill(
            "stop", expected_outcome="my custom outcome", image=SAMPLE_IMAGE
        )
        assert len(captured) == 1
        assert "my custom outcome" in captured[0]

    def test_json_in_prose_extracted(self):
        response = (
            'I analyzed the image. Here is my assessment: '
            '{"verified": true, "confidence": 0.88, "observation": "cup held", '
            '"suggested_action": null}'
            ' Overall the skill succeeded.'
        )
        mock_vlm = MagicMock()
        mock_vlm.answer_question.return_value = response
        vr = TaskVerifier(mock_vlm).verify_skill("pick_object", image=SAMPLE_IMAGE)
        assert vr.verified is True
        assert abs(vr.confidence - 0.88) < 0.001


# ---------------------------------------------------------------------------
# MockTaskVerifier
# ---------------------------------------------------------------------------

class TestMockTaskVerifier:
    def test_default_verified_true(self):
        mv = MockTaskVerifier()
        vr = mv.verify_skill("navigate_to", image=SAMPLE_IMAGE)
        assert vr.verified is True
        assert vr.confidence == 0.9

    def test_set_verified_false(self):
        mv = MockTaskVerifier()
        mv.set_verified("navigate_to", False, confidence=0.85)
        vr = mv.verify_skill("navigate_to", image=SAMPLE_IMAGE)
        assert vr.verified is False
        assert vr.confidence == 0.85

    def test_records_calls(self):
        mv = MockTaskVerifier()
        mv.verify_skill("stop")
        mv.verify_skill("navigate_to", image=SAMPLE_IMAGE)
        assert len(mv.calls) == 2

    def test_strips_index_suffix_for_lookup(self):
        mv = MockTaskVerifier()
        mv.set_verified("navigate_to", False, confidence=0.8)
        vr = mv.verify_skill("navigate_to_0")
        assert vr.verified is False

    def test_custom_observation_and_suggestion(self):
        mv = MockTaskVerifier()
        mv.set_verified("pick_object", False, observation="hand empty", suggested_action="reopen gripper")
        vr = mv.verify_skill("pick_object", image=SAMPLE_IMAGE)
        assert vr.observation == "hand empty"
        assert vr.suggested_action == "reopen gripper"


# ---------------------------------------------------------------------------
# Agent.execute() with verification
# ---------------------------------------------------------------------------

class _FakeLLMProvider(LLMProvider):
    def __init__(self, plan):
        self._plan = plan
    def plan(self, task, available_skills, capabilities, **kw):
        return self._plan


class TestAgentVerification:
    def _agent_with_llm(self, plan):
        a = Agent(provider="rule")
        a._provider = _FakeLLMProvider(plan)
        return a

    def test_verified_skill_continues(self):
        robot = _robot()
        agent = Agent(provider="rule")
        verifier = MockTaskVerifier()
        with patch.object(SkillExecutor, "execute_graph", return_value=_success_result()):
            result = agent.execute("stop the robot", robot, verification=True, verifier=verifier)
        assert result.status == TaskStatus.COMPLETED

    def test_no_verifier_calls_when_disabled(self):
        robot = _robot()
        agent = Agent(provider="rule")
        verifier = MockTaskVerifier()
        with patch.object(SkillExecutor, "execute_graph", return_value=_success_result()):
            agent.execute("stop the robot", robot, verification=False, verifier=verifier)
        assert len(verifier.calls) == 0

    def test_no_verifier_calls_when_verifier_none(self):
        robot = _robot()
        agent = Agent(provider="rule")
        with patch.object(SkillExecutor, "execute_graph", return_value=_success_result()):
            # Should not raise even though verifier=None
            result = agent.execute("stop the robot", robot, verification=True, verifier=None)
        assert result.status == TaskStatus.COMPLETED

    def test_unverified_high_confidence_emits_event(self):
        robot = _robot()
        agent = Agent(provider="rule")
        verifier = MockTaskVerifier()
        verifier.set_verified("stop", False, confidence=0.85)

        emitted = []

        def _capture(*args, **kwargs):
            emitted.append(args[0])

        with patch("apyrobo.skills.agent.emit_event", side_effect=_capture):
            agent.execute("stop the robot", robot, verification=True, verifier=verifier)

        assert "skill.verification_failed" in emitted

    def test_unverified_low_confidence_warns_and_continues(self):
        """Confidence <= 0.5 → warn but return COMPLETED (skill itself passed)."""
        robot = _robot()
        agent = Agent(provider="rule")
        verifier = MockTaskVerifier()
        verifier.set_verified("stop", False, confidence=0.3)

        with patch("apyrobo.skills.agent.logger") as mock_log:
            result = agent.execute("stop the robot", robot, verification=True, verifier=verifier)

        assert result.status == TaskStatus.COMPLETED
        # warning should have been logged
        mock_log.warning.assert_called()

    def test_verification_failed_event_has_skill_id(self):
        robot = _robot()
        agent = Agent(provider="rule")
        verifier = MockTaskVerifier()
        verifier.set_verified("stop", False, confidence=0.9)

        event_kwargs = {}

        def _capture(*args, **kwargs):
            if args[0] == "skill.verification_failed":
                event_kwargs.update(kwargs)

        with patch("apyrobo.skills.agent.emit_event", side_effect=_capture):
            agent.execute("stop the robot", robot, verification=True, verifier=verifier)

        assert "skill_id" in event_kwargs
        assert "confidence" in event_kwargs
        assert "observation" in event_kwargs

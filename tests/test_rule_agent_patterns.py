"""Tests for custom regex patterns in Agent(provider='rule', patterns=[...])."""

from __future__ import annotations

import pytest

from apyrobo.core.robot import Robot
from apyrobo.skills.agent import Agent, RuleBasedProvider


@pytest.fixture
def robot():
    return Robot.discover("mock://test")


# ---------------------------------------------------------------------------
# RuleBasedProvider directly
# ---------------------------------------------------------------------------

class TestRuleBasedProviderPatterns:
    def test_custom_pattern_matches(self):
        provider = RuleBasedProvider(patterns=[(r"inspect.*shelf", "inspect_shelf")])
        plan = provider.plan("inspect the shelf now", [], [])
        assert len(plan) == 1
        assert plan[0]["skill_id"] == "inspect_shelf"

    def test_custom_pattern_takes_priority_over_builtins(self):
        # "deliver" would normally match the built-in delivery pattern;
        # our custom pattern should win first.
        provider = RuleBasedProvider(patterns=[(r"deliver.*urgent", "urgent_deliver")])
        plan = provider.plan("deliver this package urgently", [], [])
        assert plan[0]["skill_id"] == "urgent_deliver"

    def test_no_match_falls_through_to_builtins(self):
        provider = RuleBasedProvider(patterns=[(r"very_specific_xyz", "some_skill")])
        plan = provider.plan("stop", [], [])
        # Built-in stop pattern should fire
        assert any(s["skill_id"] == "stop" for s in plan)

    def test_multiple_patterns_first_wins(self):
        provider = RuleBasedProvider(patterns=[
            (r"go left", "turn_left"),
            (r"go (left|right)", "turn_right"),
        ])
        plan = provider.plan("go left now", [], [])
        assert plan[0]["skill_id"] == "turn_left"

    def test_pattern_case_insensitive_via_lowercase(self):
        # plan() lowercases the task before matching
        provider = RuleBasedProvider(patterns=[(r"charge battery", "recharge")])
        plan = provider.plan("CHARGE BATTERY NOW", [], [])
        assert plan[0]["skill_id"] == "recharge"

    def test_no_patterns_behaves_like_default(self):
        provider = RuleBasedProvider()
        plan = provider.plan("navigate to room 3", [], [])
        assert any(s["skill_id"] == "navigate_to" for s in plan)

    def test_empty_patterns_list(self):
        provider = RuleBasedProvider(patterns=[])
        plan = provider.plan("stop", [], [])
        assert any(s["skill_id"] == "stop" for s in plan)


# ---------------------------------------------------------------------------
# Agent-level patterns kwarg
# ---------------------------------------------------------------------------

class TestAgentPatterns:
    def test_agent_patterns_kwarg_accepted(self, robot):
        agent = Agent(
            provider="rule",
            patterns=[(r"inspect.*shelf", "inspect_shelf")],
        )
        # Should not raise
        assert agent is not None

    def test_agent_custom_pattern_routes_correctly(self, robot):
        from apyrobo.skills.skill import Skill
        from apyrobo.skills.library import SkillLibrary

        inspect_skill = Skill.simple("inspect_shelf", "Inspect the shelf", capability="scan")
        lib = SkillLibrary()
        lib.register(inspect_skill)

        agent = Agent(
            provider="rule",
            library=lib,
            patterns=[(r"inspect.*shelf", "inspect_shelf")],
        )
        graph = agent.plan("inspect the shelf in aisle 3", robot)
        skill_ids = [s.skill_id.rsplit("_", 1)[0] for s in graph.get_execution_order()]
        assert "inspect_shelf" in skill_ids

    def test_agent_patterns_take_priority_over_builtins(self, robot):
        agent = Agent(
            provider="rule",
            patterns=[(r".*", "report_status")],  # match-all overrides everything
        )
        graph = agent.plan("deliver package to room 1", robot)
        skill_ids = [s.skill_id.rsplit("_", 1)[0] for s in graph.get_execution_order()]
        # Our catch-all pattern should fire before the built-in delivery plan
        assert skill_ids == ["report_status"]

    def test_agent_without_patterns_uses_builtins(self, robot):
        agent = Agent(provider="rule")
        graph = agent.plan("stop", robot)
        assert len(graph) > 0

    def test_agent_unknown_task_falls_through_to_fallback(self, robot):
        agent = Agent(
            provider="rule",
            patterns=[(r"only_matches_special", "special_skill")],
        )
        # This task doesn't match the custom pattern OR any built-in
        graph = agent.plan("zzz_no_match_at_all_xyzzy", robot)
        # Falls back to report_status
        skill_ids = [s.skill_id.rsplit("_", 1)[0] for s in graph.get_execution_order()]
        assert "report_status" in skill_ids

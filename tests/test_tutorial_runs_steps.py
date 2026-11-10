"""apyrobo tutorial runs every code step it shows, and step 4 runs on a SafetyEnforcer."""

from __future__ import annotations

import argparse

from apyrobo.cli import _TUTORIAL_STEPS, cmd_tutorial
from apyrobo.core.robot import Robot
from apyrobo.safety.enforcer import SafetyEnforcer
from apyrobo.skills.agent import Agent


def _run_steps() -> dict:
    ns: dict = {"Robot": Robot, "Agent": Agent}
    for step in _TUTORIAL_STEPS:
        if step.get("code"):
            exec(step["code"], ns)  # noqa: S102
    return ns


def test_every_step_shows_its_output(capsys) -> None:
    cmd_tutorial(argparse.Namespace(non_interactive=True))
    out = capsys.readouterr().out
    assert "Mock-turtlebot4" in out          # step 1
    assert "navigate_to: Move to a 2D position" in out  # step 2
    assert "navigate_to_0: Navigate To" in out  # step 3
    assert "Result: completed" in out         # step 4
    assert "wave 3/3" in out                  # step 5
    assert "this step failed" not in out


def test_step_four_executes_on_a_safety_enforcer() -> None:
    ns = _run_steps()
    assert isinstance(ns["safe_robot"], SafetyEnforcer)
    assert ns["executor"]._robot is ns["safe_robot"]
    assert ns["result"].status.value == "completed"


def test_step_four_text_matches_the_default_policy() -> None:
    step = next(s for s in _TUTORIAL_STEPS if s["title"].startswith("Step 4"))
    assert "enforcing safety policies at every step" not in step["description"]
    assert "a raw robot moves unchecked" in step["description"]
    assert "1.5 m/s" in step["description"]
    enforcer = SafetyEnforcer(Robot.discover("mock://tutorial_policy"))
    assert enforcer.policy.max_speed == 1.5


def test_a_failing_step_is_reported_not_hidden(capsys, monkeypatch) -> None:
    broken = [{"title": "Step X", "description": "d", "code": "undefined_name"}]
    monkeypatch.setattr("apyrobo.cli._TUTORIAL_STEPS", broken)
    cmd_tutorial(argparse.Namespace(non_interactive=True))
    out = capsys.readouterr().out
    assert "this step failed: NameError" in out

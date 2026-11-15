"""`apyrobo listen` refuses what the rule planner would drop, like `apyrobo plan`.

`listen` hears one command and prints the plan for it. With the rule planner,
"go to (3, 3) then go to (-2, 1)" printed a one-step plan to (-2, 1) and
"go 2 meters forward" printed a status report, both exit 0, while
`apyrobo plan` and `apyrobo mcp` refuse these tasks with a reason.
"""

from __future__ import annotations

import sys

import pytest

import apyrobo.voice as voice
from apyrobo import cli


def _listen(monkeypatch, heard: str) -> None:
    real = voice.MockVoiceAdapter

    def heard_once(*args, **kwargs):
        return real(listen_responses=[heard])

    monkeypatch.setattr(voice, "MockVoiceAdapter", heard_once)
    monkeypatch.setattr(
        sys, "argv",
        ["apyrobo", "listen", "--robot", "mock://turtlebot4", "--provider", "rule",
         "--adapter", "mock"],
    )
    cli.main()


REFUSED = [
    ("go to (3, 3) then go to (-2, 1)", "more than one place to drive"),
    ("patrol between (1, 1) and (4, 4)", "more than one place to drive"),
    ("go 2 meters forward", "did not recognise this task"),
    ("navigate to dock", "No target"),
]


@pytest.mark.parametrize("heard,why", REFUSED)
def test_listen_refuses_a_task_the_rule_planner_would_drop(monkeypatch, capsys, heard, why):
    with pytest.raises(SystemExit) as exc:
        _listen(monkeypatch, heard)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert f"Heard: {heard!r}" in out
    assert why in out
    assert "Plan:" not in out


def test_listen_refusal_names_the_cli_flag_not_mcp_tools(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _listen(monkeypatch, "go 2 meters forward")
    out = capsys.readouterr().out
    assert "use --provider llm" in out
    assert "move_to and rotate" not in out


def test_listen_still_prints_a_plan_it_can_read(monkeypatch, capsys):
    _listen(monkeypatch, "navigate to (1.0, 2.0)")
    out = capsys.readouterr().out
    assert "Plan: 1 skill(s)" in out
    assert "x: 1.0" in out
    assert "y: 2.0" in out


def test_default_mock_command_still_plans(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv",
        ["apyrobo", "listen", "--robot", "mock://turtlebot4", "--provider", "rule",
         "--adapter", "mock"],
    )
    cli.main()
    assert "Plan: 1 skill(s)" in capsys.readouterr().out

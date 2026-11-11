"""`apyrobo plan` and `apyrobo execute` refuse what the rule planner would drop.

The rule planner drives to one place per step and falls back to a status
report on a task it does not recognise. `apyrobo mcp` refuses such tasks, but
the CLI planned and ran them: "go to (3, 3) then go to (-2, 1)" drove straight
to (-2, 1) at 100% confidence, and "go 2 meters forward" ended in
"Result: completed" with nothing moved.
"""

from __future__ import annotations

import sys

import pytest

from apyrobo import cli
from apyrobo.core.robot import Robot


def _run(monkeypatch, command: str, task: str, *extra: str) -> list[tuple]:
    moves: list[tuple] = []
    original = Robot.move

    def spy(self, x, y, *args, **kwargs):
        moves.append((round(float(x), 6), round(float(y), 6)))
        return original(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Robot, "move", spy)
    monkeypatch.setattr(
        sys, "argv",
        ["apyrobo", command, task, "--robot", "mock://turtlebot4", "--provider", "rule", *extra],
    )
    cli.main()
    return moves


REFUSED = [
    ("go to (3, 3) then go to (-2, 1)", "more than one place to drive"),
    ("patrol between (1, 1) and (4, 4)", "more than one place to drive"),
    ("go 2 meters forward", "did not recognise this task"),
    ("navigate to dock", "No target"),
]


@pytest.mark.parametrize("task,why", REFUSED)
def test_execute_refuses_and_does_not_move(monkeypatch, capsys, task, why):
    moves: list[tuple] = []
    with pytest.raises(SystemExit) as exc:
        moves = _run(monkeypatch, "execute", task)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert why in out
    assert "Result: completed" not in out
    assert moves == []


def test_force_does_not_override_the_refusal(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, "execute", "go to (3, 3) then go to (-2, 1)", "--force")
    assert exc.value.code == 1
    assert "more than one place to drive" in capsys.readouterr().out


@pytest.mark.parametrize("task,why", REFUSED)
def test_plan_refuses_with_the_reason(monkeypatch, capsys, task, why):
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, "plan", task)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert why in out
    assert "Proceed: yes" not in out


def test_unrecognised_task_names_a_cli_option_not_mcp_tools(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _run(monkeypatch, "plan", "go 2 meters forward")
    out = capsys.readouterr().out
    assert "--provider llm" in out
    assert "move_to" not in out and "start the server" not in out


@pytest.mark.parametrize(
    "task,target",
    [("go to (3, 3)", (3.0, 3.0)), ("go to x=3 y=-1", (3.0, -1.0))],
)
def test_one_target_still_runs(monkeypatch, capsys, task, target):
    moves = _run(monkeypatch, "execute", task)
    assert "Result: completed" in capsys.readouterr().out
    assert moves and moves[-1] == target


def test_delivery_still_plans_both_stops(monkeypatch, capsys):
    _run(monkeypatch, "plan", "deliver from (1, 1) to (5, 1)")
    out = capsys.readouterr().out
    assert "Plan: 4 skills" in out
    assert "Proceed: yes" in out

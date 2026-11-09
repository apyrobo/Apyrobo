"""`apyrobo execute --max-speed` caps the speed the robot is actually sent.

The command built a SafetyEnforcer but executed the task on the raw robot,
so `--max-speed 0.2` still drove a "go to (3, 4) at 2 m/s" task at 2 m/s,
and nan, inf, 0 and negative values were accepted silently.
"""

from __future__ import annotations

import sys

import pytest

from apyrobo import cli
from apyrobo.core.robot import Robot


def _run(monkeypatch, task: str, *extra: str) -> list[dict]:
    calls: list[dict] = []
    original = Robot.move

    def spy(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Robot, "move", spy)
    monkeypatch.setattr(
        sys, "argv",
        ["apyrobo", "execute", task, "--robot", "mock://turtlebot4", *extra],
    )
    cli.main()
    return calls


def test_task_speed_above_max_speed_is_clamped(monkeypatch, capsys):
    calls = _run(monkeypatch, "go to (3, 4) at 2 m/s", "--max-speed", "0.2")
    assert calls, "the robot was never sent a move"
    assert all(c["speed"] is not None and c["speed"] <= 0.2 for c in calls), calls
    assert "Result: completed" in capsys.readouterr().out


def test_default_speed_above_max_speed_is_clamped(monkeypatch):
    calls = _run(monkeypatch, "go to (3, 4)", "--max-speed", "0.2")
    assert calls
    assert all(c["speed"] is not None and c["speed"] <= 0.2 for c in calls), calls


def test_speed_under_max_speed_is_kept(monkeypatch):
    calls = _run(monkeypatch, "go to (3, 4) at 0.3 m/s", "--max-speed", "1.0")
    assert [c["speed"] for c in calls] == [pytest.approx(0.3)]


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "0", "-1"])
def test_bad_max_speed_is_refused_before_moving(monkeypatch, capsys, bad):
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, "go to (3, 4)", f"--max-speed={bad}")
    assert exc.value.code == 2
    assert "--max-speed must be a positive number" in capsys.readouterr().err

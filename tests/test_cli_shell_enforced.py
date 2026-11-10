"""apyrobo shell puts a SafetyEnforcer in the REPL as ``robot``.

The raw robot is still there as ``raw_robot``, named in the banner, and
``--max-speed`` refuses the same values as ``serve`` and ``execute``.
"""
from __future__ import annotations

import argparse
import math
from typing import Any
from unittest.mock import patch

import pytest

from apyrobo.cli import cmd_shell
from apyrobo.core.robot import Robot
from apyrobo.safety.enforcer import SafetyEnforcer


def _run(**kw: Any) -> tuple[str, dict[str, Any]]:
    seen: dict[str, Any] = {}

    def interact(banner: str = "", local: dict | None = None, exitmsg: str = "") -> None:
        seen["banner"] = banner
        seen["ns"] = dict(local or {})

    args = argparse.Namespace(robot="mock://turtlebot4", provider="rule", **kw)
    with patch("code.interact", side_effect=interact):
        cmd_shell(args)
    return seen["banner"], seen["ns"]


def test_robot_is_an_enforcer_and_raw_robot_is_not() -> None:
    _, ns = _run(max_speed=1.5)
    assert isinstance(ns["robot"], SafetyEnforcer)
    assert isinstance(ns["raw_robot"], Robot)
    assert not isinstance(ns["raw_robot"], SafetyEnforcer)


def test_max_speed_sets_the_policy_and_caps_moves() -> None:
    _, ns = _run(max_speed=0.4)
    robot = ns["robot"]
    robot.move(x=1.0, y=0.0, speed=5.0)
    assert robot.last_speed is not None and robot.last_speed <= 0.4 + 1e-9


def test_banner_names_both_robots_and_the_limit() -> None:
    banner, _ = _run(max_speed=0.8)
    assert "SafetyEnforcer" in banner
    assert "0.8 m/s" in banner
    assert "raw_robot" in banner
    assert "no safety checks" in banner


def test_default_max_speed_without_the_flag() -> None:
    # Callers that build args without max_speed (older code, tests) get 1.5.
    banner, ns = _run()
    assert "1.5 m/s" in banner
    assert isinstance(ns["robot"], SafetyEnforcer)


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0.0, -1.0])
def test_bad_max_speed_is_refused(bad: float, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        _run(max_speed=bad)
    assert exc.value.code == 2
    assert "--max-speed must be a positive number" in capsys.readouterr().err


def test_max_speed_flag_parses() -> None:
    from apyrobo import cli

    with patch("code.interact") as interact, \
            patch("sys.argv", ["apyrobo", "shell", "--max-speed", "0.7"]):
        cli.main()
    ns = interact.call_args.kwargs["local"]
    assert isinstance(ns["robot"], SafetyEnforcer)
    assert "0.7 m/s" in interact.call_args.kwargs["banner"]

"""Every command that drives a robot goes through the SafetyEnforcer.

docs/comparison.md says "Every command passes through the SafetyEnforcer
before reaching the robot". The same bypass was found and fixed one command at
a time (serve --execute, voice, plan --simulate, skill compose, test-skill), so
this file checks the claim for all of them at once, from the command line where
it can: each entry point drives mock:// with an over-limit speed (it must reach
the robot clamped to the policy's 1.5 m/s) and with a target inside a no-go
zone (nothing must reach the robot).

The no-go zone is added to every SafetyEnforcer the command builds, so a
command that skips the enforcer moves into it and fails here.

Out of scope, with the reason:
- ``apyrobo shell`` is a Python REPL: the user types library calls on the
  ``robot`` object, the same as the library API.
- Library APIs that take a caller's robot (Agent.execute, the operations
  scheduler, fleet/manager.py): the caller chooses to pass a SafetyEnforcer.
- ``apyrobo listen`` and ``apyrobo plan`` without ``--simulate`` only plan;
  ``apyrobo dashboard`` only reads state. A test below checks plan-only moves
  nothing.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from typing import Any

import pytest

from apyrobo import cli
from apyrobo.core.robot import Robot
from apyrobo.safety.enforcer import SafetyEnforcer

LIMIT = 1.5  # the default policy's max speed, which every command below uses
ZONE = {"x_min": 2.0, "x_max": 4.0, "y_min": 2.0, "y_max": 5.0}
OVER = (1.0, 1.0, 9.0)  # x, y, speed: outside the zone, speed over the limit
INTO = (3.0, 4.0, 0.5)  # x, y, speed: inside the zone, speed within the limit


ENFORCERS: list[SafetyEnforcer] = []


@pytest.fixture
def moves(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every move that reaches a robot, and zone every enforcer."""
    calls: list[dict[str, Any]] = []
    ENFORCERS.clear()
    original_move = Robot.move

    def spy(self: Robot, *args: Any, **kwargs: Any) -> Any:
        x, y = (list(args) + [kwargs.get("x"), kwargs.get("y")])[:2]
        speed = kwargs.get("speed", args[2] if len(args) > 2 else None)
        calls.append({"x": x, "y": y, "speed": speed})
        return original_move(self, *args, **kwargs)

    monkeypatch.setattr(Robot, "move", spy)

    original_init = SafetyEnforcer.__init__

    def zoned_init(self: SafetyEnforcer, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self.add_collision_zone(dict(ZONE))
        ENFORCERS.append(self)

    monkeypatch.setattr(SafetyEnforcer, "__init__", zoned_init)
    return calls


def _main(monkeypatch: pytest.MonkeyPatch, *argv: str, stdin: str = "") -> None:
    monkeypatch.setattr(sys, "argv", ["apyrobo", *argv])
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    # A refused step exits non-zero; the moves are what is checked.
    with contextlib.suppress(SystemExit):
        cli.main()


def _task(x: float, y: float, speed: float) -> str:
    return f"go to ({x:g}, {y:g}) at speed {speed:g}"


def _execute(mp: pytest.MonkeyPatch, x: float, y: float, speed: float) -> None:
    _main(mp, "execute", _task(x, y, speed), "--force")


def _serve_execute(mp: pytest.MonkeyPatch, x: float, y: float, speed: float) -> None:
    _main(mp, "serve", "--execute", stdin=json.dumps({"task": _task(x, y, speed)}) + "\n")


def _voice(mp: pytest.MonkeyPatch, x: float, y: float, speed: float) -> None:
    import apyrobo.voice as voice

    real = voice.MockVoiceAdapter
    mp.setattr(voice, "MockVoiceAdapter", lambda *_a, **_k: real([_task(x, y, speed), "stop"]))
    _main(mp, "voice", "--listen", "--max-turns", "1")


def _plan_simulate_deploy(mp: pytest.MonkeyPatch, x: float, y: float, speed: float) -> None:
    _main(mp, "plan", _task(x, y, speed), "--simulate", "--auto-deploy",
          "--real-robot", "mock://real")


def _skill_compose(mp: pytest.MonkeyPatch, x: float, y: float, speed: float) -> None:
    lines = iter([f"test navigate_to x={x:g} y={y:g} speed={speed:g}", "quit"])

    def fake_input(_prompt: str = "") -> str:
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    mp.setattr("builtins.input", fake_input)
    _main(mp, "skill", "compose")


def _test_skill(mp: pytest.MonkeyPatch, x: float, y: float, speed: float) -> None:
    _main(mp, "test-skill", "navigate_to", "--params",
          json.dumps({"x": x, "y": y, "speed": speed}))


def _mcp(mp: pytest.MonkeyPatch, x: float, y: float, speed: float) -> None:
    # The server `apyrobo mcp` builds, called in-process: its stdio loop runs
    # threads that this file does not need to test.
    from apyrobo.mcp_server import build_server

    server = build_server("mock://turtlebot4")
    server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"}}})
    server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
    server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
        "name": "move_to", "arguments": {"x": x, "y": y, "speed": speed}}})


ENTRY_POINTS = {
    "execute": _execute,
    "serve --execute": _serve_execute,
    "voice": _voice,
    "plan --simulate --auto-deploy": _plan_simulate_deploy,
    "skill compose": _skill_compose,
    "test-skill": _test_skill,
    "mcp": _mcp,
}


@pytest.mark.parametrize("name", list(ENTRY_POINTS))
def test_over_limit_speed_reaches_the_robot_clamped(name: str, monkeypatch, moves) -> None:
    ENTRY_POINTS[name](monkeypatch, *OVER)
    assert moves, f"{name}: nothing reached the robot"
    for call in moves:
        assert call["speed"] is not None and call["speed"] <= LIMIT + 1e-9, (name, moves)


@pytest.mark.parametrize("name", list(ENTRY_POINTS))
def test_target_in_no_go_zone_never_reaches_the_robot(name: str, monkeypatch, moves) -> None:
    ENTRY_POINTS[name](monkeypatch, *INTO)
    inside = [
        c for c in moves
        if c["x"] is not None and ZONE["x_min"] <= c["x"] <= ZONE["x_max"]
        and ZONE["y_min"] <= c["y"] <= ZONE["y_max"]
    ]
    assert inside == [], (name, moves)
    # Not refused because the command never got that far: an enforcer saw it.
    assert any(e._violations for e in ENFORCERS), f"{name}: no enforcer refused the move"


def test_plan_only_moves_nothing(monkeypatch, moves) -> None:
    _main(monkeypatch, "plan", _task(*OVER))
    assert moves == []

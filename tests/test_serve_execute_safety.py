"""`apyrobo serve --execute` runs every task through the safety enforcer.

Execute mode handed the planned task to the raw robot, so a front-end could
send "go to (3, 4) at speed 5" and the robot was sent 5 m/s, and a no-go zone
in the policy was never checked. Each task now runs through a SafetyEnforcer
(the "default" policy, or the one the server is given), and `apyrobo serve`
takes --max-speed for it.
"""

from __future__ import annotations

import io
import json
import sys

import pytest

from apyrobo import cli
from apyrobo.core.robot import Robot
from apyrobo.orchestration import OrchestrationServer
from apyrobo.orchestration.adapter import MockOrchestrationAdapter, OrchestrationMessage
from apyrobo.safety.enforcer import SafetyEnforcer, SafetyPolicy
from apyrobo.skills.agent import Agent


@pytest.fixture
def moves(monkeypatch):
    calls: list[dict] = []
    original = Robot.move

    def spy(self, *args, **kwargs):
        calls.append({"robot": self.robot_id, **kwargs})
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Robot, "move", spy)
    return calls


def _serve(tasks, **kwargs):
    adapter = MockOrchestrationAdapter(tasks=tasks)
    server = OrchestrationServer(adapter, Agent(provider="rule"), execute_tasks=True, **kwargs)
    server.run()
    return server, adapter.sent


def test_task_speed_is_clamped_to_the_default_policy(moves):
    _, sent = _serve(["go to (3, 4) at speed 5"])
    assert sent[0].metadata["execution"]["status"] == "completed"
    assert moves and all(c["speed"] is not None and c["speed"] <= 1.5 for c in moves), moves


def test_task_speed_is_clamped_to_the_given_policy(moves):
    _, sent = _serve(["go to (3, 4) at speed 5"], safety_policy=SafetyPolicy(max_speed=0.2))
    assert sent[0].metadata["execution"]["status"] == "completed"
    assert moves and all(c["speed"] is not None and c["speed"] <= 0.2 for c in moves), moves


def test_default_speed_is_clamped_too(moves):
    _serve(["go to (3, 4)"], safety_policy=SafetyPolicy(max_speed=0.2))
    assert moves and all(c["speed"] is not None and c["speed"] <= 0.2 for c in moves), moves


def test_speed_under_the_limit_is_kept(moves):
    _serve(["go to (3, 4) at speed 0.3"], safety_policy=SafetyPolicy(max_speed=1.0))
    assert [c["speed"] for c in moves] == [pytest.approx(0.3)]


def test_no_go_zone_is_refused_and_the_robot_stays(moves):
    robot = Robot.discover("mock://zone_bot")
    policy = SafetyPolicy(collision_zones=[{"x_min": 2, "x_max": 4, "y_min": 2, "y_max": 4}])
    _, sent = _serve(
        ["go to (3, 3)"], default_robot=robot, default_robot_uri="mock://zone_bot",
        safety_policy=policy,
    )
    execution = sent[0].metadata["execution"]
    assert execution["status"] == "failed"
    assert execution["steps_completed"] == 0
    assert moves == []
    assert robot._adapter.position == (0.0, 0.0)


def test_robot_named_in_the_message_is_limited_too(moves):
    _serve(
        [OrchestrationMessage(task="go to (1, 1) at speed 5", robot_uri="mock://other_bot")],
        safety_policy=SafetyPolicy(max_speed=0.2),
    )
    assert moves and all(c["robot"] == "other_bot" for c in moves), moves
    assert all(c["speed"] is not None and c["speed"] <= 0.2 for c in moves), moves


def test_one_enforcer_per_robot_keeps_its_record():
    robot = Robot.discover("mock://record_bot")
    server, _ = _serve(
        ["go to (1, 1) at speed 5", "go to (2, 2) at speed 5"],
        default_robot=robot, default_robot_uri="mock://record_bot",
        safety_policy=SafetyPolicy(max_speed=0.2),
    )
    enforcer = server._enforcer_for(robot)
    assert len([i for i in enforcer.interventions if i["type"] == "speed_clamped"]) == 2


def test_an_enforcer_given_as_robot_is_used_as_is(moves):
    enforcer = SafetyEnforcer(Robot.discover("mock://wrapped"), policy=SafetyPolicy(max_speed=0.1))
    server, _ = _serve(
        ["go to (1, 1) at speed 5"], default_robot=enforcer, default_robot_uri="mock://wrapped",
        safety_policy=SafetyPolicy(max_speed=1.0),
    )
    assert server._enforcer_for(enforcer) is enforcer
    assert [c["speed"] for c in moves] == [pytest.approx(0.1)]


def test_plan_only_does_not_move(moves):
    adapter = MockOrchestrationAdapter(tasks=["go to (3, 4) at speed 5"])
    OrchestrationServer(adapter, Agent(provider="rule")).run()
    assert moves == []


def _cli(monkeypatch, *extra: str, stdin: str = "") -> None:
    monkeypatch.setattr(sys, "argv", ["apyrobo", "serve", "--execute", *extra])
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    cli.main()


def test_cli_max_speed_caps_the_robot(monkeypatch, moves, capsys):
    task = json.dumps({"task": "go to (3, 4) at speed 5"}) + "\n"
    _cli(monkeypatch, "--max-speed", "0.2", stdin=task)
    assert moves and all(c["speed"] is not None and c["speed"] <= 0.2 for c in moves), moves
    reply = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert reply["metadata"]["execution"]["status"] == "completed"


def test_cli_default_limit_is_1_5(monkeypatch, moves):
    _cli(monkeypatch, stdin=json.dumps({"task": "go to (3, 4) at speed 5"}) + "\n")
    assert moves and all(c["speed"] is not None and c["speed"] <= 1.5 for c in moves), moves


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "0", "-1"])
def test_cli_bad_max_speed_is_refused_before_moving(monkeypatch, moves, capsys, bad):
    with pytest.raises(SystemExit) as exc:
        _cli(monkeypatch, f"--max-speed={bad}", stdin=json.dumps({"task": "go to (1, 1)"}) + "\n")
    assert exc.value.code == 2
    assert "--max-speed must be a positive number" in capsys.readouterr().err
    assert moves == []

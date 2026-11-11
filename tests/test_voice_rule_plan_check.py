"""Voice commands are held to the rule-planner checks ``apyrobo mcp`` makes.

The rule planner drives to one place per step and falls back to a status
report on a task it does not recognise. ``apyrobo mcp``, ``apyrobo plan`` and
``apyrobo execute`` refuse such tasks; ``apyrobo voice`` (voice_loop) and
VoiceAgent ran them: "go to (3, 3) then go to (-2, 1)" drove only to (-2, 1),
and "go 2 meters forward" ended in "completed" with nothing moved.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import TaskStatus
from apyrobo.skills.agent import Agent
from apyrobo.voice import MockVoiceAdapter, VoiceAgent, voice_loop


def _spy_robot() -> tuple[Robot, list[dict]]:
    robot = Robot.discover("mock://turtlebot4")
    moves: list[dict] = []
    real_move = robot.move

    def spy(*args, **kwargs):
        moves.append(dict(kwargs))
        return real_move(*args, **kwargs)

    robot.move = spy  # type: ignore[method-assign]
    return robot, moves


def _loop(text: str, robot: Robot | None = None):
    robot, moves = (robot, []) if robot is not None else _spy_robot()
    adapter = MockVoiceAdapter([text, "stop"])
    turns = voice_loop(
        agent=Agent(provider="rule"),
        robot=robot,
        adapter=adapter,
        max_turns=3,
    )
    return turns, moves, adapter


@pytest.mark.parametrize(
    "task, words",
    [
        ("go to (3, 3) then go to (-2, 1)", "one place"),
        ("go 2 meters forward", "not"),
        ("patrol between (1, 1) and (4, 4)", "one place"),
    ],
)
def test_voice_loop_refuses_a_task_the_rule_planner_would_drop(task, words):
    turns, moves, adapter = _loop(task)
    assert moves == []
    assert len(turns) == 1
    result = turns[0]["result"]
    assert result.status == TaskStatus.FAILED
    assert result.steps_completed == 0
    assert result.error and words in result.error.lower()
    # The reason is spoken, not only recorded.
    assert any(result.error in said for said in adapter.spoken)


def test_voice_loop_refusal_does_not_name_mcp_tools():
    turns, _, _ = _loop("dance a little")
    error = turns[0]["result"].error
    assert error is not None
    assert "move_to and rotate" not in error
    assert "--provider llm" in error


def test_voice_loop_still_runs_a_task_it_reads():
    turns, moves, _ = _loop("go to (2, 3)")
    assert turns[0]["result"].status == TaskStatus.COMPLETED
    assert len(moves) == 1
    assert (moves[0]["x"], moves[0]["y"]) == pytest.approx((2.0, 3.0))


def test_voice_loop_runs_coordinates_the_way_the_planner_reads_them():
    turns, moves, _ = _loop("go to x=3 y=-1")
    assert turns[0]["result"].status == TaskStatus.COMPLETED
    assert (moves[-1]["x"], moves[-1]["y"]) == pytest.approx((3.0, -1.0))


def test_voice_loop_status_ask_still_completes():
    turns, moves, _ = _loop("what is your status")
    assert turns[0]["result"].status == TaskStatus.COMPLETED
    assert moves == []


def test_voice_agent_refuses_a_task_the_rule_planner_would_drop():
    robot, moves = _spy_robot()
    va = VoiceAgent(
        agent=Agent(provider="rule"),
        robot=robot,
        stt=MockVoiceAdapter(responses=["go to (3, 3) then go to (-2, 1)"]),
    )
    summary = va.run()
    assert moves == []
    assert "failed" in summary.lower()
    assert "one place" in summary.lower()


def test_non_rule_agent_is_not_replanned():
    agent = MagicMock()
    agent.execute.return_value = MagicMock(
        status=TaskStatus.COMPLETED, steps_completed=1, steps_total=1, error=None
    )
    adapter = MockVoiceAdapter(["go to (3, 3) then go to (-2, 1)", "stop"])
    voice_loop(agent, Robot.discover("mock://turtlebot4"), adapter, max_turns=3)
    agent.execute.assert_called_once()
    assert agent.execute.call_args.kwargs["task"] == "go to (3, 3) then go to (-2, 1)"
    agent.plan.assert_not_called()


def test_cli_voice_prints_the_refusal(capsys, monkeypatch):
    from apyrobo import cli, voice

    monkeypatch.setattr(
        voice, "MockVoiceAdapter",
        lambda *a, **k: MockVoiceAdapter(["go 2 meters forward", "stop"]),
    )
    args = argparse.Namespace(
        robot="mock://turtlebot4", provider="rule", model=None,
        adapter="mock", listen=True, max_turns=3,
    )
    cli.cmd_voice(args)
    out = capsys.readouterr().out
    assert "Result: failed (0/0)" in out
    assert "--provider llm" in out

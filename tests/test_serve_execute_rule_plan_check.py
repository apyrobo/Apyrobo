"""`apyrobo serve --execute` refuses tasks the rule planner would drop.

The rule planner drives to one place per step and falls back to a status
report on a task it does not recognise. Execute mode ran whatever it planned:
"go to (3, 3) then go to (-2, 1)" drove only to (-2, 1), and "go 2 meters
forward" ended "completed" with nothing moved. With the rule planner, execute
mode now makes the same checks as `apyrobo mcp`, `plan` and `execute`: such a
task is not run and its execution is "failed" with the reason.
"""

from __future__ import annotations

import pytest

from apyrobo.core.robot import Robot
from apyrobo.orchestration import OrchestrationServer
from apyrobo.orchestration.adapter import MockOrchestrationAdapter
from apyrobo.skills.agent import Agent


@pytest.fixture
def moves(monkeypatch):
    calls: list[dict] = []
    original = Robot.move

    def spy(self, *args, **kwargs):
        calls.append({"x": kwargs.get("x", args[0] if args else None),
                      "y": kwargs.get("y", args[1] if len(args) > 1 else None)})
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Robot, "move", spy)
    return calls


def _serve(task, execute=True):
    robot = Robot.discover("mock://plan_check_bot")
    adapter = MockOrchestrationAdapter(tasks=[task])
    server = OrchestrationServer(
        adapter, Agent(provider="rule"), default_robot=robot,
        default_robot_uri="mock://plan_check_bot", execute_tasks=execute,
    )
    server.run()
    return robot, adapter.sent[0].metadata


@pytest.mark.parametrize(
    "task",
    [
        "go to (3, 3) then go to (-2, 1)",
        "go 2 meters forward",
        "patrol between (1, 1) and (4, 4)",
    ],
)
def test_task_the_planner_would_drop_is_not_run(moves, task):
    robot, metadata = _serve(task)
    assert metadata["status"] == "planned"
    execution = metadata["execution"]
    assert execution["status"] == "failed"
    assert execution["steps_completed"] == 0
    assert execution["error"]
    assert moves == []
    assert robot._adapter.position == (0.0, 0.0)


def test_refusal_names_no_mcp_tools(moves):
    _, metadata = _serve("go 2 meters forward")
    error = metadata["execution"]["error"]
    assert "move_to" not in error
    assert "start the server" not in error


def test_several_targets_reason_matches_mcp(moves):
    _, metadata = _serve("go to (3, 3) then go to (-2, 1)")
    from apyrobo.mcp_server import _MANY_TARGETS

    assert metadata["execution"]["error"] == _MANY_TARGETS


def test_single_target_still_runs(moves):
    robot, metadata = _serve("go to (3, 4)")
    assert metadata["execution"]["status"] == "completed"
    assert moves and (moves[-1]["x"], moves[-1]["y"]) == (3.0, 4.0)


def test_coordinates_are_read_as_mcp_reads_them(moves):
    robot, metadata = _serve("go to x=3 y=-1")
    assert metadata["execution"]["status"] == "completed"
    assert moves and (moves[-1]["x"], moves[-1]["y"]) == (3.0, -1.0)


def test_plan_only_mode_is_unchanged(moves):
    _, metadata = _serve("go to (3, 3) then go to (-2, 1)", execute=False)
    assert metadata["status"] == "planned"
    assert "execution" not in metadata
    assert moves == []

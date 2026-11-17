"""The patrol and delivery workflow examples drive to the places they name.

They used to send "navigate to corridor A" and "navigate to office 101 and
deliver package" to the rule planner, which reads only coordinates: on
mock:// the patrol never left the origin and every delivery ended at a
made-up (5, 5), while both scripts printed success without looking at the
result. ``apyrobo mcp``, ``plan`` and ``execute`` refuse such tasks.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from apyrobo.core.schemas import TaskResult, TaskStatus
from apyrobo.mcp_server import build_server

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / "examples" / "workflows"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"wf_{name}", WORKFLOWS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _accepted(task: str) -> bool:
    server = build_server("mock://workflow_check")
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "plan_task", "arguments": {"task": task}},
        }
    )
    return json.loads(response["result"]["content"][0]["text"])["can_proceed"] is True


class _Recorder:
    """Wraps the real Agent and keeps every task and the robot it drove."""

    def __init__(self, module: ModuleType, fail_on: str | None = None) -> None:
        self.tasks: list[str] = []
        self.robot = None
        real = module.Agent
        recorder = self

        class Agent(real):  # type: ignore[misc, valid-type]
            def execute(self, task, robot, **kwargs):  # type: ignore[override]
                recorder.tasks.append(task)
                recorder.robot = robot
                if fail_on is not None and fail_on in task:
                    return TaskResult(task_name=task, status=TaskStatus.FAILED,
                                      error="blocked")
                return super().execute(task=task, robot=robot, **kwargs)

        module.Agent = Agent


def test_patrol_visits_every_named_waypoint_in_order():
    patrol = _load("patrol_loop")
    rec = _Recorder(patrol)
    assert patrol.patrol("mock://patrol", rounds=1, pause_s=0) is True
    expected = [f"go to ({x}, {y})" for x, y in patrol.WAYPOINTS.values()]
    assert rec.tasks == expected
    assert rec.robot.get_position() == list(patrol.WAYPOINTS.values())[-1]


def test_patrol_stops_and_says_so_when_a_waypoint_fails(capsys):
    patrol = _load("patrol_loop")
    x, y = patrol.WAYPOINTS["lab room 1"]
    rec = _Recorder(patrol, fail_on=f"({x}, {y})")
    assert patrol.patrol("mock://patrol", rounds=2, pause_s=0) is False
    assert len(rec.tasks) == 3
    out = capsys.readouterr().out
    assert "lab room 1 not reached: blocked" in out
    assert "[patrol] complete" not in out


def test_delivery_ends_each_run_at_the_destination_and_returns():
    delivery = _load("delivery_run")
    rec = _Recorder(delivery)
    results = delivery.run_deliveries("mock://courier", 3)
    assert [r.delivered for r in results] == [True, True, True]
    assert rec.tasks[-1] == "go to (0.0, 0.0)"
    assert rec.robot.get_position() == delivery.MAILROOM[1]
    for d, task in zip(delivery.DELIVERIES[:3], rec.tasks[:3], strict=True):
        x, y = d["at"]
        assert task.endswith(f"to ({x}, {y})")


def test_delivery_reports_a_failed_drop_as_not_delivered(capsys):
    delivery = _load("delivery_run")
    x, y = delivery.DELIVERIES[1]["at"]
    _Recorder(delivery, fail_on=f"to ({x}, {y})")
    results = delivery.run_deliveries("mock://courier", 3)
    assert [r.delivered for r in results] == [True, False]
    assert results[1].notes == "blocked"
    assert "not delivered: blocked" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["patrol_loop", "delivery_run"])
def test_every_task_the_example_sends_is_one_apyrobo_mcp_accepts(name):
    module = _load(name)
    rec = _Recorder(module)
    if name == "patrol_loop":
        module.patrol("mock://check", rounds=1, pause_s=0)
    else:
        module.run_deliveries("mock://check", len(module.DELIVERIES))
    assert rec.tasks
    for task in rec.tasks:
        assert _accepted(task), task


@pytest.mark.parametrize(
    "task", ["navigate to corridor A", "navigate to office 101 and deliver package"],
)
def test_the_old_example_tasks_were_refused(task):
    assert not _accepted(task)

"""The workflow examples drive to the places they name.

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


def test_charging_cycle_docks_then_returns_to_the_work_spot():
    charging = _load("charging_cycle")
    rec = _Recorder(charging)
    assert charging.charging_cycle("mock://charger", threshold=20.0, max_steps=10) is True
    (dx, dy), (wx, wy) = charging.DOCK, charging.WORK_SPOT
    assert rec.tasks == [f"go to ({dx}, {dy})", f"go to ({wx}, {wy})"]
    assert rec.robot.get_position() == charging.WORK_SPOT


def test_charging_cycle_stops_when_the_dock_is_not_reached(capsys):
    charging = _load("charging_cycle")
    x, y = charging.DOCK
    rec = _Recorder(charging, fail_on=f"({x}, {y})")
    assert charging.charging_cycle("mock://charger", threshold=20.0, max_steps=10) is False
    assert len(rec.tasks) == 1
    out = capsys.readouterr().out
    assert "charging dock not reached: blocked" in out
    assert "cycle complete" not in out


def test_charging_cycle_is_not_complete_while_still_docked(capsys):
    charging = _load("charging_cycle")
    _Recorder(charging)
    assert charging.charging_cycle("mock://charger", threshold=20.0, max_steps=4) is False
    assert "still docked" in capsys.readouterr().out


def test_inspection_reaches_every_checkpoint_in_order():
    inspection = _load("inspection_round")
    rec = _Recorder(inspection)
    records = inspection.run_inspection("mock://inspector", inspection.DEFAULT_CHECKPOINTS)
    assert [r.reached for r in records] == [True] * len(inspection.DEFAULT_CHECKPOINTS)
    expected = [f"go to ({float(x)}, {float(y)})"
                for x, y in (cp["at"] for cp in inspection.DEFAULT_CHECKPOINTS)]
    assert rec.tasks == expected
    assert rec.robot.get_position() == tuple(inspection.DEFAULT_CHECKPOINTS[-1]["at"])


def test_inspection_marks_unreached_and_unplaced_checkpoints(capsys):
    inspection = _load("inspection_round")
    rec = _Recorder(inspection, fail_on="(6.0, 2.0)")
    checkpoints = [
        {"id": "a", "name": "server room", "at": [6.0, 2.0], "checks": ["no water"]},
        {"id": "b", "name": "lab", "checks": ["lights on"]},
        {"id": "c", "name": "dock", "at": [1, 1], "checks": ["clear"]},
    ]
    records = inspection.run_inspection("mock://inspector", checkpoints)
    assert [r.reached for r in records] == [False, False, True]
    assert records[0].notes == "blocked" and records[0].checks == []
    assert "at" in records[1].notes
    assert rec.tasks == ["go to (6.0, 2.0)", "go to (1.0, 1.0)"]
    out = capsys.readouterr().out
    assert "not reached: blocked" in out
    assert "lab (b) skipped" in out


def test_inspection_main_exits_1_when_a_checkpoint_is_missed(tmp_path, monkeypatch):
    inspection = _load("inspection_round")
    _Recorder(inspection)
    config = tmp_path / "cp.json"
    config.write_text(json.dumps([{"id": "a", "name": "lab"}]))
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        ["inspection_round.py", "--robot", "mock://i", "--config", str(config),
         "--output", str(report)],
    )
    with pytest.raises(SystemExit) as exc:
        inspection.main()
    assert exc.value.code == 1
    assert json.loads(report.read_text())[0]["reached"] is False


def test_multi_floor_drives_to_the_elevator_then_the_destination(capsys):
    floors = _load("multi_floor_navigation")
    rec = _Recorder(floors)
    assert floors.navigate_multi_floor("mock://lift", 1, 3) is True
    (ex, ey), (dx, dy) = floors.ELEVATOR, floors.FLOOR_DESTINATIONS[3][1]
    assert rec.tasks == [f"go to ({ex}, {ey})", f"go to ({dx}, {dy})"]
    assert rec.robot.get_position() == (dx, dy)
    assert "(simulated)" in capsys.readouterr().out


def test_multi_floor_same_floor_drives_straight_there():
    floors = _load("multi_floor_navigation")
    rec = _Recorder(floors)
    assert floors.navigate_multi_floor("mock://lift", 2, 2) is True
    x, y = floors.FLOOR_DESTINATIONS[2][1]
    assert rec.tasks == [f"go to ({x}, {y})"]


def test_multi_floor_stops_when_the_elevator_is_not_reached(capsys):
    floors = _load("multi_floor_navigation")
    x, y = floors.ELEVATOR
    rec = _Recorder(floors, fail_on=f"({x}, {y})")
    assert floors.navigate_multi_floor("mock://lift", 1, 3) is False
    assert len(rec.tasks) == 1
    out = capsys.readouterr().out
    assert "elevator not reached: blocked" in out
    assert "arrived" not in out


def test_multi_floor_does_not_pretend_to_ride_a_real_elevator(capsys):
    floors = _load("multi_floor_navigation")
    assert floors.ride_elevator("ros2://husky", 1, 3) is False
    assert "no elevator integration" in capsys.readouterr().out


def test_multi_floor_main_exits_1_for_an_unknown_floor(monkeypatch, capsys):
    floors = _load("multi_floor_navigation")
    rec = _Recorder(floors)
    monkeypatch.setattr("sys.argv", ["multi_floor_navigation.py", "--to-floor", "9"])
    with pytest.raises(SystemExit) as exc:
        floors.main()
    assert exc.value.code == 1
    assert rec.tasks == []
    assert "no destination for floor 9" in capsys.readouterr().out


def test_fleet_sends_each_robot_to_its_stops():
    fleet = _load("fleet_coordination")
    rec = _Recorder(fleet)
    workers = fleet.coordinate_fleet(3, 6)
    stops = fleet.TASK_POOL[:6]
    assert sorted(rec.tasks) == sorted(f"go to ({x}, {y})" for _, (x, y) in stops)
    for i, w in enumerate(workers):
        mine = stops[i::3]
        assert w.completed == [name for name, _ in mine]
        assert w.errors == []
        assert w.robot.get_position() == mine[-1][1]


def test_fleet_reports_a_missed_stop_and_main_exits_1(monkeypatch, capsys):
    fleet = _load("fleet_coordination")
    name, (x, y) = fleet.TASK_POOL[1]
    _Recorder(fleet, fail_on=f"({x}, {y})")
    monkeypatch.setattr("sys.argv", ["fleet_coordination.py", "--fleet", "2", "--tasks", "4"])
    with pytest.raises(SystemExit) as exc:
        fleet.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert f"{name} not reached: blocked" in out
    assert "3/4 reached, 1 not reached" in out


def _run_example(name: str, module: ModuleType) -> None:
    if name == "patrol_loop":
        module.patrol("mock://check", rounds=1, pause_s=0)
    elif name == "delivery_run":
        module.run_deliveries("mock://check", len(module.DELIVERIES))
    elif name == "charging_cycle":
        module.charging_cycle("mock://check", threshold=20.0, max_steps=10)
    elif name == "multi_floor_navigation":
        module.navigate_multi_floor("mock://check", 1, 3)
    elif name == "fleet_coordination":
        module.coordinate_fleet(2, len(module.TASK_POOL))
    else:
        module.run_inspection("mock://check", module.DEFAULT_CHECKPOINTS)


@pytest.mark.parametrize(
    "name",
    [
        "patrol_loop", "delivery_run", "charging_cycle", "inspection_round",
        "multi_floor_navigation", "fleet_coordination",
    ],
)
def test_every_task_the_example_sends_is_one_apyrobo_mcp_accepts(name):
    module = _load(name)
    rec = _Recorder(module)
    _run_example(name, module)
    assert rec.tasks
    for task in rec.tasks:
        assert _accepted(task), task


@pytest.mark.parametrize(
    "task",
    [
        "navigate to corridor A",
        "navigate to office 101 and deliver package",
        "navigate to charging dock",
        "undock from charging station",
        "navigate to server room",
        "navigate to elevator lobby",
        "press button for floor 3",
        "navigate to third floor lab",
        "deliver package to office 101",
        "inspect corridor B",
    ],
)
def test_the_old_example_tasks_were_refused(task):
    assert not _accepted(task)

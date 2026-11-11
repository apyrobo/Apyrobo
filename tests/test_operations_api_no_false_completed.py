"""OperationsApiServer never reports a task 'completed' that it did not run.

POST /tasks used to mark a task 'completed' when the server had no agent, no
robot_id was given, or the robot was not on the swarm bus, though nothing ran.
A task cancelled with DELETE was also overwritten with the run's outcome.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

from apyrobo.operations import OperationsApiServer


def _queue(srv: OperationsApiServer, robot_id: str | None = "tb4") -> str:
    task_id = "t1"
    srv._tasks[task_id] = {
        "task_id": task_id, "task": "go to (1, 1)", "robot_id": robot_id,
        "status": "queued",
    }
    return task_id


def _bus(robots: dict[str, Any]) -> MagicMock:
    bus = MagicMock()
    bus.robot_ids = list(robots)

    def get_robot(rid: str) -> Any:
        if rid not in robots:
            raise KeyError(rid)
        return robots[rid]

    bus.get_robot.side_effect = get_robot
    return bus


def test_no_agent_fails_and_store_records_failure() -> None:
    store = MagicMock()
    srv = OperationsApiServer(state_store=store, swarm_bus=_bus({"tb4": object()}))
    task_id = _queue(srv)
    srv._run_task_background(task_id, "go to (1, 1)", "tb4")
    task = srv._tasks[task_id]
    assert task["status"] == "failed"
    assert "no agent configured" in task["error"]
    store.fail_task.assert_called_once()
    store.complete_task.assert_not_called()


def test_no_robot_id_fails_without_running_agent() -> None:
    agent = MagicMock()
    srv = OperationsApiServer(agent=agent, swarm_bus=_bus({"tb4": object()}))
    task_id = _queue(srv, robot_id=None)
    srv._run_task_background(task_id, "go to (1, 1)", None)
    assert srv._tasks[task_id]["status"] == "failed"
    assert "no robot_id" in srv._tasks[task_id]["error"]
    agent.execute.assert_not_called()


def test_unknown_robot_fails_without_running_agent() -> None:
    agent = MagicMock()
    srv = OperationsApiServer(agent=agent, swarm_bus=_bus({"tb4": object()}))
    task_id = _queue(srv, robot_id="ghost")
    srv._run_task_background(task_id, "go to (1, 1)", "ghost")
    assert srv._tasks[task_id]["status"] == "failed"
    assert "'ghost' is not on the swarm bus" in srv._tasks[task_id]["error"]
    agent.execute.assert_not_called()


def test_no_swarm_bus_fails_without_running_agent() -> None:
    agent = MagicMock()
    srv = OperationsApiServer(agent=agent)
    task_id = _queue(srv)
    srv._run_task_background(task_id, "go to (1, 1)", "tb4")
    assert srv._tasks[task_id]["status"] == "failed"
    agent.execute.assert_not_called()


def test_agent_and_robot_still_run() -> None:
    robot = object()
    result = MagicMock()
    result.status = "completed"
    result.to_dict.return_value = {"status": "completed"}
    agent = MagicMock()
    agent.execute.return_value = result
    srv = OperationsApiServer(agent=agent, swarm_bus=_bus({"tb4": robot}))
    task_id = _queue(srv)
    srv._run_task_background(task_id, "go to (1, 1)", "tb4")
    assert srv._tasks[task_id]["status"] == "completed"
    assert agent.execute.call_args.args[:2] == ("go to (1, 1)", robot)


def test_cancelled_while_queued_never_starts() -> None:
    agent = MagicMock()
    srv = OperationsApiServer(agent=agent, swarm_bus=_bus({"tb4": object()}))
    task_id = _queue(srv)
    srv._tasks[task_id]["status"] = "cancelled"
    srv._run_task_background(task_id, "go to (1, 1)", "tb4")
    assert srv._tasks[task_id]["status"] == "cancelled"
    agent.execute.assert_not_called()


def test_cancel_during_run_is_not_overwritten() -> None:
    started, release = threading.Event(), threading.Event()
    result = MagicMock()
    result.status = "completed"

    def execute(*_a: Any, **_k: Any) -> Any:
        started.set()
        release.wait(5)
        return result

    agent = MagicMock()
    agent.execute.side_effect = execute
    store = MagicMock()
    srv = OperationsApiServer(agent=agent, state_store=store,
                              swarm_bus=_bus({"tb4": object()}))
    task_id = _queue(srv)
    t = threading.Thread(target=srv._run_task_background,
                         args=(task_id, "go to (1, 1)", "tb4"))
    t.start()
    assert started.wait(5)
    srv._tasks[task_id]["status"] = "cancelled"  # what DELETE /tasks/{id} does
    release.set()
    t.join(5)
    assert srv._tasks[task_id]["status"] == "cancelled"
    store.complete_task.assert_not_called()


def test_cancel_during_failing_run_is_not_overwritten() -> None:
    started, release = threading.Event(), threading.Event()

    def execute(*_a: Any, **_k: Any) -> Any:
        started.set()
        release.wait(5)
        raise RuntimeError("drive aborted")

    agent = MagicMock()
    agent.execute.side_effect = execute
    srv = OperationsApiServer(agent=agent, swarm_bus=_bus({"tb4": object()}))
    task_id = _queue(srv)
    t = threading.Thread(target=srv._run_task_background,
                         args=(task_id, "go to (1, 1)", "tb4"))
    t.start()
    assert started.wait(5)
    srv._tasks[task_id]["status"] = "cancelled"
    release.set()
    t.join(5)
    assert srv._tasks[task_id]["status"] == "cancelled"
    assert "error" not in srv._tasks[task_id]

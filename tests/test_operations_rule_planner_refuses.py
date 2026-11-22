"""The REST API and scheduled tasks hold the rule planner to apyrobo mcp's checks.

docs/api_reference.md's POST /tasks example, "deliver package to room 3",
went to a rule-planned Agent, which drove a mock robot to a made-up (5, 5)
and the task was reported "completed". apyrobo mcp, plan, execute, serve and
listen refuse such a task before anything moves; OperationsApiServer and
ScheduledTaskRunner now do the same.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any
from unittest.mock import MagicMock

import pytest

from apyrobo import Agent, Robot, SwarmBus
from apyrobo.operations import OperationsApiServer, ScheduledTaskRunner


def _server(agent: Any) -> tuple[OperationsApiServer, Any, Any]:
    robot = Robot.discover("mock://tb4")
    bus = SwarmBus()
    bus.register(robot)
    store = MagicMock()
    return OperationsApiServer(swarm_bus=bus, agent=agent, state_store=store), robot, store


def _run(srv: OperationsApiServer, task: str) -> dict[str, Any]:
    srv._tasks["t1"] = {"task_id": "t1", "task": task, "robot_id": "tb4", "status": "queued"}
    srv._run_task_background("t1", task, "tb4")
    return srv._tasks["t1"]


@pytest.mark.parametrize(
    "task",
    [
        "deliver package to room 3",  # the docs' own example
        "go to the kitchen",
        "go to (3, 3) then go to (-2, 1)",
        "go 2 meters forward",
    ],
)
def test_unreadable_task_fails_and_robot_stays(task: str) -> None:
    srv, robot, store = _server(Agent(provider="rule"))
    before = robot.get_position()
    entry = _run(srv, task)
    assert entry["status"] == "failed"
    assert "Nothing was run." in entry["error"]
    assert "move_to" not in entry["error"]  # MCP tool names mean nothing here
    assert robot.get_position() == before
    store.fail_task.assert_called_once()
    store.complete_task.assert_not_called()


def test_task_with_coordinates_still_runs() -> None:
    srv, robot, store = _server(Agent(provider="rule"))
    entry = _run(srv, "go to (2, 1)")
    assert entry["status"] == "completed"
    assert robot.get_position() == (2.0, 1.0)
    assert any(c.args[0] == "t1" for c in store.complete_task.call_args_list)


def test_non_rule_agent_gets_the_task_unchanged() -> None:
    agent = MagicMock()
    agent.execute.return_value = MagicMock(status="completed", to_dict=lambda: {})
    srv, _, store = _server(agent)
    entry = _run(srv, "deliver package to room 3")
    assert entry["status"] == "completed"
    assert agent.execute.call_args.args[0] == "deliver package to room 3"
    assert agent.execute.call_args.kwargs["state_store"] is store


def test_documented_post_example_over_http() -> None:
    srv, robot, _ = _server(Agent(provider="rule"))
    srv.port = 0
    srv.start()
    try:
        port = srv._server.server_address[1]
        body = json.dumps({"task": "deliver package to room 3", "robot_id": "tb4"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/tasks", data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 202
            task_id = json.loads(resp.read())["task_id"]
        status: dict[str, Any] = {}
        for _ in range(50):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/tasks/{task_id}", timeout=5) as r:
                status = json.loads(r.read())
            if status["status"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        assert status["status"] == "failed"
        assert "No target" in status["error"]
        assert robot.get_position() == (0.0, 0.0)
    finally:
        srv.stop()


def test_scheduled_task_with_no_target_does_not_move() -> None:
    robot = Robot.discover("mock://tb4")
    store = MagicMock()
    runner = ScheduledTaskRunner(state_store=store)
    runner.add_task("patrol", "* * * * *", "patrol the warehouse", robot, Agent(provider="rule"))
    job = next(j for j in runner._jobs if j["name"] == "patrol")
    runner._execute(job, time.time())
    assert robot.get_position() == (0.0, 0.0)
    store.set.assert_not_called()  # no "last_result" recorded for a task that never ran

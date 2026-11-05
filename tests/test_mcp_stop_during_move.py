"""A stop or a client's cancel reaches the robot while a move is running."""
from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from apyrobo import mcp_server
from apyrobo.mcp_server import build_server
from tests.test_mcp_client_loss import INIT, SlowRobot, move, pipes, send, serve_in_thread, status


def stop(msg_id=4):
    return {
        "jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
        "params": {"name": "stop", "arguments": {}},
    }


def cancel(request_id=2):
    return {
        "jsonrpc": "2.0", "method": "notifications/cancelled",
        "params": {"requestId": request_id, "reason": "abandoned"},
    }


def task(msg_id=3, text="deliver from (1, 0) to (2, 0)"):
    return {
        "jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
        "params": {"name": "execute_task", "arguments": {"task": text}},
    }


class Client:
    """A server on pipes, with replies read by id."""

    def __init__(self, server):
        self.server = server
        self.stdin, self.to_server, self.from_server, self.stdout = pipes()
        self.thread = serve_in_thread(server, self.stdin, self.stdout)
        self.replies = {}
        send(self.to_server, INIT)
        self.reply(1)

    def send(self, *messages):
        send(self.to_server, *messages)

    def reply(self, msg_id, timeout=10):
        deadline = time.monotonic() + timeout
        while msg_id not in self.replies:
            assert time.monotonic() < deadline, f"no reply to {msg_id}: {self.replies}"
            line = self.from_server.readline()
            message = json.loads(line)
            self.replies[message.get("id")] = message
        return self.replies[msg_id]

    def close(self):
        self.to_server.close()
        self.thread.join(10)


@pytest.fixture
def server():
    return build_server(robot_uri="mock://stop-during-move")


@pytest.fixture
def client(server):
    connected = Client(server)
    yield connected
    connected.close()


def text_of(reply):
    return reply["result"]["content"][0]["text"]


def audit_lines(path):
    with open(path) as log:
        return [json.loads(line) for line in log]


def test_stop_reaches_a_running_move_and_gets_its_own_reply(server, client):
    robot = SlowRobot(server)
    client.send(move())
    assert robot.moving.wait(10)
    client.send(stop())
    assert robot.stopped.wait(5), "stop did not reach the running move"
    moved = client.reply(2)
    assert moved["result"]["isError"] is True
    assert text_of(moved).startswith("STOPPED: a stop request arrived during move_to")
    stopped = client.reply(4)
    assert stopped["result"]["isError"] is False
    assert json.loads(text_of(stopped))["status"] == "stopped"


def test_the_next_move_runs_after_a_stop(server, client):
    robot = SlowRobot(server)
    client.send(move())
    assert robot.moving.wait(10)
    client.send(stop())
    client.reply(4)
    assert not server.client_gone
    robot.moving.clear()
    client.send(move(5, 3.0))
    assert robot.moving.wait(5), "a stop ended the session's moves"
    # The first stop's repeats are over: this move is still running.
    time.sleep(mcp_server.HALT_REPEAT_S * 3)
    assert not robot.stopped.is_set()
    client.send(stop(6))
    assert client.reply(5)["result"]["isError"] is True
    assert robot.moves == [(1.0, 0.0), (3.0, 0.0)]


def test_a_move_read_before_the_stop_does_not_start_afterwards(server, client):
    robot = SlowRobot(server)
    client.send(move(2, 1.0))
    assert robot.moving.wait(10)
    robot.moving.clear()
    client.send(move(3, 2.0), stop(4), move(5, 3.0))
    queued = client.reply(3)
    assert queued["result"]["isError"] is True
    assert "a stop came in after this request" in text_of(queued)
    assert client.reply(4)["result"]["isError"] is False
    assert robot.moving.wait(5)
    assert robot.moves == [(1.0, 0.0), (3.0, 0.0)]  # 3.0 was sent after the stop.
    client.send(stop(6))


def test_a_stop_in_a_batch_only_holds_back_what_came_before_it(server, client):
    robot = SlowRobot(server)
    client.send([move(2, 1.0), stop(4), status(9)])
    while not client.replies.get(None):
        client.replies[None] = json.loads(client.from_server.readline())
    replies = {r["id"]: r for r in client.replies[None]}
    assert replies[2]["result"]["isError"] is True
    assert replies[4]["result"]["isError"] is False
    assert robot.moves == []


def test_cancel_stops_the_running_move_and_sends_no_reply(server, client):
    robot = SlowRobot(server)
    client.send(move())
    assert robot.moving.wait(10)
    client.send(cancel(2))
    assert robot.stopped.wait(5), "cancel did not reach the running move"
    client.send(status(9))
    assert client.reply(9)["result"]["isError"] is False
    assert 2 not in client.replies  # MCP: no response for a cancelled request.


def test_cancel_works_the_same_in_the_stateless_era(server, client):
    robot = SlowRobot(server)
    modern = move()
    modern["params"]["_meta"] = {
        mcp_server.META_PROTOCOL_VERSION: mcp_server.MODERN_PROTOCOL_VERSIONS[0],
        mcp_server.META_CLIENT_CAPABILITIES: {},
    }
    client.send(modern)
    assert robot.moving.wait(10)
    client.send(cancel(2), status(9))
    assert robot.stopped.wait(5)
    client.reply(9)
    assert 2 not in client.replies


@pytest.mark.parametrize("request_id", [7, "2", None, {"a": 1}, [2]])
def test_cancel_for_anything_else_changes_nothing(server, client, request_id):
    robot = SlowRobot(server)
    client.send(move())
    assert robot.moving.wait(10)
    client.send(cancel(request_id))
    time.sleep(0.1)
    assert robot.stops == 0
    assert not robot.stopped.is_set()
    client.send(stop())
    assert client.reply(2)["result"]["isError"] is True


def test_a_cancel_for_a_finished_call_does_not_stop_the_next_move(server, client):
    robot = SlowRobot(server)
    client.send(status(9), cancel(9))
    client.reply(9)
    client.send(move(2))
    assert robot.moving.wait(10)
    assert robot.stops == 0
    client.send(stop())
    client.reply(2)


def test_cancel_of_a_queued_move_leaves_the_running_one_and_it_never_starts(server, client):
    robot = SlowRobot(server)
    client.send(move(2, 1.0), move(3, 2.0))
    assert robot.moving.wait(10)
    client.send(cancel(3))
    time.sleep(0.1)
    assert robot.stops == 0
    client.send(stop())
    client.reply(2)
    client.reply(4)
    client.send(status(9))
    client.reply(9)
    assert robot.moves == [(1.0, 0.0)]
    assert 3 not in client.replies


def test_a_failed_stop_keeps_trying_until_the_move_returns(server, client):
    robot = SlowRobot(server)
    calls = []
    good_stop = robot.robot.stop

    def flaky_stop():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("bus error")
        good_stop()

    robot.robot.stop = flaky_stop
    client.send(move())
    assert robot.moving.wait(10)
    send(client.to_server, cancel(2))
    assert robot.stopped.wait(5)
    assert len(calls) >= 3


@pytest.mark.parametrize("message, decision", [(stop(), "stopped"), (cancel(), "cancelled")])
def test_the_audit_log_says_how_the_move_ended(tmp_path, message, decision):
    log = tmp_path / "audit.jsonl"
    server = build_server(robot_uri="mock://stop-audit", audit_log=str(log))
    robot = SlowRobot(server)
    client = Client(server)
    client.send(move())
    assert robot.moving.wait(10)
    client.send(message)
    assert robot.stopped.wait(5)
    client.send(status(9))
    client.reply(9)
    client.close()
    lines = [line for line in audit_lines(log) if line["tool"] == "move_to"]
    assert [line["decision"] for line in lines] == ["started", decision]
    assert "arrived during move_to" in lines[-1]["detail"]


def test_a_cancelled_queued_move_is_audited(tmp_path):
    log = tmp_path / "audit.jsonl"
    server = build_server(robot_uri="mock://stop-audit", audit_log=str(log))
    robot = SlowRobot(server)
    client = Client(server)
    client.send(move(2, 1.0), move(3, 2.0))
    assert robot.moving.wait(10)
    client.send(cancel(3), stop(4))
    client.reply(4)
    client.close()
    decisions = [line["decision"] for line in audit_lines(log) if line["tool"] == "move_to"]
    assert decisions == ["started", "stopped", "cancelled"]


def test_execute_task_drops_its_remaining_steps(server, client):
    robot = SlowRobot(server)
    client.send(task())
    assert robot.moving.wait(10)
    client.send(stop())
    reply = client.reply(3)
    assert reply["result"]["isError"] is True
    assert text_of(reply).startswith("STOPPED: a stop request arrived during execute_task")
    assert client.reply(4)["result"]["isError"] is False
    assert len(robot.moves) == 1


def test_a_stop_is_not_retried_away_by_a_step_with_retries(server, client, monkeypatch):
    from apyrobo.skills import executor

    robot = SlowRobot(server)
    execute_skill = executor.SkillExecutor.execute_skill
    dispatch = executor.SkillExecutor._dispatch_skill
    runs = []

    def with_retries(self, skill, parameters=None):
        skill.retry_count = 3
        return execute_skill(self, skill, parameters)

    def failing_step(self, skill, params):
        runs.append(skill.skill_id)
        dispatch(self, skill, params)
        raise RuntimeError("goal canceled")

    monkeypatch.setattr(executor.SkillExecutor, "execute_skill", with_retries)
    monkeypatch.setattr(executor.SkillExecutor, "_dispatch_skill", failing_step)
    client.send(task())
    assert robot.moving.wait(10)
    client.send(stop())
    assert client.reply(3)["result"]["isError"] is True
    assert len(runs) == 1


def test_the_executor_gives_the_abort_reason_and_runs_nothing(server):
    from apyrobo.skills.executor import SkillExecutor

    graph, _ = server.toolbox._plan("deliver from (1, 0) to (2, 0)")
    executor = SkillExecutor(server.toolbox.enforcer, abort_reason=lambda: "stopped by a test")
    result = executor.execute_graph(graph)
    assert result.error == "stopped by a test"
    assert result.steps_completed == 0


def test_stop_with_no_motion_running_is_an_ordinary_call(server, client):
    client.send(stop())
    assert client.reply(4)["result"]["isError"] is False
    client.send(move(5, 1.0))
    assert client.reply(5)["result"]["isError"] is False


def test_a_finished_move_with_no_stop_is_still_a_success(client):
    client.send(move())
    assert client.reply(2)["result"]["isError"] is False


def test_bad_lines_do_not_upset_the_reader(server, client):
    robot = SlowRobot(server)
    client.send(move())
    assert robot.moving.wait(10)
    client.to_server.write('{not json\n[1, "x", null]\n"stop"\n[' * 1 + "\n")
    client.to_server.flush()
    client.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": 3})
    client.send({"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "stop"}})
    time.sleep(0.1)
    assert robot.stops == 0
    client.send(stop())
    assert client.reply(2)["result"]["isError"] is True


def test_without_a_reader_serving_is_unchanged():
    import io

    server = build_server(robot_uri="mock://stop-no-reader")
    lines = [INIT, move(), stop()]
    out = io.StringIO()
    server.serve(io.StringIO("".join(json.dumps(m) + "\n" for m in lines)), out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r["id"] for r in replies] == [1, 2, 4]
    assert all(r["result"].get("isError") is not True for r in replies)


_SLOW_SERVER = """
import sys, threading
from apyrobo.mcp_server import build_server
server = build_server(robot_uri="mock://sdk-cancel", audit_log=sys.argv[1])
robot = server.toolbox.enforcer._robot
stopped = threading.Event()
stop = robot.stop
def slow_move(x, y, speed=None):
    print("moving", file=sys.stderr, flush=True)
    stopped.clear()
    if x > 0 and not stopped.wait(60):
        raise SystemExit("never stopped")
def counted_stop():
    stopped.set()
    stop()
robot.move = slow_move
robot.stop = counted_stop
server.serve(watch_client=True)
"""


def test_the_official_client_abandoning_a_call_stops_the_robot(tmp_path):
    pytest.importorskip("mcp")
    from importlib.metadata import version

    if int(version("mcp").split(".")[0]) < 2:
        pytest.skip("needs mcp>=2.2")
    import anyio
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    log = tmp_path / "audit.jsonl"
    params = StdioServerParameters(
        command=sys.executable, args=["-c", _SLOW_SERVER, str(log)],
    )

    async def run():
        with anyio.fail_after(60):
            async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                with anyio.move_on_after(1.0) as scope:
                    await session.call_tool("move_to", {"x": 1.0, "y": 0.0})
                assert scope.cancelled_caught
                # The server is still there, and the next move runs.
                return await session.call_tool("move_to", {"x": -1.0, "y": 0.0})

    after = anyio.run(run)
    assert after.is_error is False
    decisions = [line["decision"] for line in audit_lines(log) if line["tool"] == "move_to"]
    assert decisions == ["started", "cancelled", "started", "ran"]


def test_real_process_stops_a_move_from_stdin(tmp_path):
    """No threads of ours in the way: the CLI's own server, a real pipe."""
    script = _SLOW_SERVER
    log = tmp_path / "audit.jsonl"
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(log)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        process.stdin.write(json.dumps(INIT) + "\n")
        process.stdin.write(json.dumps(move()) + "\n")
        process.stdin.flush()
        assert process.stdout.readline()
        assert "moving" in process.stderr.readline()
        process.stdin.write(json.dumps(stop()) + "\n")
        process.stdin.flush()
        replies = [json.loads(process.stdout.readline()) for _ in range(2)]
        assert [r["id"] for r in replies] == [2, 4]
        assert replies[0]["result"]["isError"] is True
    finally:
        process.stdin.close()
        process.wait(20)
        process.stdout.close()
        process.stderr.close()

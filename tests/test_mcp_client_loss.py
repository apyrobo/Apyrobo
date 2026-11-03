"""The robot stops when the MCP client goes away, even during a move."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from apyrobo.mcp_server import build_server

INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {},
               "clientInfo": {"name": "t", "version": "0"}},
}


def move(msg_id=2, x=1.0):
    return {
        "jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
        "params": {"name": "move_to", "arguments": {"x": x, "y": 0.0}},
    }


def status(msg_id=9):
    return {
        "jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
        "params": {"name": "robot_status", "arguments": {}},
    }


class SlowRobot:
    """Wraps the mock's move: it blocks until stopped, like a Nav2 goal."""

    def __init__(self, server):
        self.robot = server.toolbox.enforcer._robot
        self.moving = threading.Event()
        self.stopped = threading.Event()
        self.moves = []
        self.stops = 0
        self._stop = self.robot.stop
        self.robot.move = self._blocking_move
        self.robot.stop = self._counting_stop

    def _blocking_move(self, x, y, speed=None):
        self.moves.append((x, y))
        self.stopped.clear()
        self.moving.set()
        assert self.stopped.wait(10), "the move was never stopped"

    def _counting_stop(self):
        self.stops += 1
        self.stopped.set()
        self._stop()


@pytest.fixture
def server():
    return build_server(robot_uri="mock://client-loss")


def pipes():
    read_in, write_in = os.pipe()
    read_out, write_out = os.pipe()
    return (
        os.fdopen(read_in, "r"), os.fdopen(write_in, "w"),
        os.fdopen(read_out, "r"), os.fdopen(write_out, "w"),
    )


def send(stream, *messages):
    stream.write("".join(json.dumps(m) + "\n" for m in messages))
    stream.flush()


def serve_in_thread(server, stdin, stdout):
    thread = threading.Thread(
        target=server.serve, args=(stdin, stdout), kwargs={"watch_client": True}, daemon=True,
    )
    thread.start()
    return thread


def test_eof_during_a_blocking_move_stops_the_robot(server):
    robot = SlowRobot(server)
    stdin, client_in, client_out, stdout = pipes()
    thread = serve_in_thread(server, stdin, stdout)
    send(client_in, INIT, move())
    assert robot.moving.wait(10)
    assert robot.stops == 0
    client_in.close()
    assert robot.stopped.wait(5), "EOF during a move did not stop the robot"
    thread.join(10)
    assert not thread.is_alive()
    assert server.client_gone


def test_move_queued_behind_the_running_one_does_not_start(server):
    robot = SlowRobot(server)
    stdin, client_in, client_out, stdout = pipes()
    thread = serve_in_thread(server, stdin, stdout)
    send(client_in, INIT, move(2, 1.0), move(3, 2.0), status())
    assert robot.moving.wait(10)
    client_in.close()
    thread.join(10)
    stdout.close()
    replies = {r["id"]: r for r in map(json.loads, client_out.read().splitlines())}
    assert robot.moves == [(1.0, 0.0)]
    assert replies[3]["result"]["isError"] is True
    assert "client disconnected" in replies[3]["result"]["content"][0]["text"]
    assert replies[9]["result"]["isError"] is False  # reads still answer


@pytest.mark.parametrize("tool, arguments", [
    ("rotate", {"angle_rad": 1.0}),
    ("execute_task", {"task": "go to (1, 0)"}),
])
def test_every_motion_tool_is_refused_after_loss(server, tool, arguments):
    server.client_gone = True
    reply = server._tools_call(1, {"name": tool, "arguments": arguments})
    assert reply["result"]["isError"] is True
    assert "client disconnected" in reply["result"]["content"][0]["text"]


def test_stop_repeats_until_the_motion_call_ends(server):
    """A move that began just as the first stop went out is stopped too."""
    robot = SlowRobot(server)
    server._motion_running = True
    server.client_lost("test")
    deadline = time.monotonic() + 5
    while robot.stops < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    server._motion_running = False
    assert robot.stops >= 2


def test_client_lost_twice_is_one_halt(server):
    robot = SlowRobot(server)
    server.client_lost("a")
    server.client_lost("b")
    time.sleep(0.1)
    assert robot.stops == 1


def test_a_failing_stop_does_not_kill_the_halt_loop(server):
    calls = []

    def bad_stop():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("no link")
        server._motion_running = False

    server.toolbox.enforcer.stop = bad_stop
    server._motion_running = True
    server.client_lost("test")
    deadline = time.monotonic() + 5
    while len(calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(calls) >= 2


def test_broken_stdout_counts_as_client_loss(server):
    robot = SlowRobot(server)
    stdin, client_in, client_out, stdout = pipes()
    client_out.close()
    send(client_in, INIT)
    thread = serve_in_thread(server, stdin, stdout)
    thread.join(10)
    assert not thread.is_alive()
    assert server.client_gone
    assert robot.stopped.wait(5)


def test_serve_without_watching_keeps_in_memory_scripts_working(server):
    import io

    out = io.StringIO()
    server.serve(io.StringIO(json.dumps(INIT) + "\n" + json.dumps(move()) + "\n"), out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert replies[1]["result"]["isError"] is False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP"])
def test_real_process_stops_and_exits_on_signal(name):
    proc = subprocess.Popen(
        [sys.executable, "-m", "apyrobo", "mcp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    send(proc.stdin, INIT)
    json.loads(proc.stdout.readline())
    signum = getattr(signal, name)
    proc.send_signal(signum)
    assert proc.wait(15) == 128 + signum
    stderr = proc.stderr.read()
    assert "Fatal Python error" not in stderr
    assert "stopping the robot" in stderr
    proc.stdin.close()


def test_real_process_refuses_a_piped_move_and_exits_cleanly():
    payload = "".join(json.dumps(m) + "\n" for m in (INIT, move()))
    proc = subprocess.run(
        [sys.executable, "-m", "apyrobo", "mcp"],
        input=payload, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    replies = [json.loads(line) for line in proc.stdout.splitlines()]
    assert replies[1]["result"]["isError"] is True
    assert "client disconnected" in replies[1]["result"]["content"][0]["text"]

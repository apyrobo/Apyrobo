"""robot_status answers during a move, and nothing about that gets in a stop's way."""
from __future__ import annotations

import json
import threading

import pytest

from apyrobo.mcp_server import build_server
from tests.test_mcp_client_loss import (
    INIT,
    SlowRobot,
    move,
    pipes,
    send,
    serve_in_thread,
    status,
)

STOP = {
    "jsonrpc": "2.0", "id": 7, "method": "tools/call",
    "params": {"name": "stop", "arguments": {}},
}
CANCEL_MOVE = {
    "jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 2},
}


class Session:
    """A server on real pipes, driven as a client would."""

    def __init__(self, server):
        self.server = server
        self.robot = SlowRobot(server)
        self.stdin, self.client_in, self.client_out, self.stdout = pipes()
        self.thread = serve_in_thread(server, self.stdin, self.stdout)
        self.read_lines = []
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.seen = threading.Condition()

    def _read(self):
        for line in self.client_out:
            with self.seen:
                decoded = json.loads(line)
                self.read_lines.extend(decoded if isinstance(decoded, list) else [decoded])
                self.seen.notify_all()

    def wait_for(self, msg_id, timeout=10):
        with self.seen:
            assert self.seen.wait_for(
                lambda: any(r.get("id") == msg_id for r in self.read_lines), timeout
            ), f"no reply to {msg_id}; got {self.read_lines}"
            return next(r for r in self.read_lines if r.get("id") == msg_id)

    def ids(self):
        with self.seen:
            return [r.get("id") for r in self.read_lines]

    def start_move(self):
        send(self.client_in, INIT)
        self.wait_for(1)
        send(self.client_in, move())
        assert self.robot.moving.wait(10)

    def close(self):
        self.client_in.close()
        self.thread.join(10)
        assert not self.thread.is_alive()
        self.stdout.close()
        self.reader.join(10)
        self.client_out.close()


@pytest.fixture
def session():
    active = Session(build_server(robot_uri="mock://status-during-move"))
    yield active
    if not active.client_in.closed:
        active.close()


def payload(reply):
    return json.loads(reply["result"]["content"][0]["text"])


def test_status_is_answered_while_the_move_is_still_running(session):
    session.start_move()
    send(session.client_in, status())
    reply = session.wait_for(9)
    assert reply["result"]["isError"] is False
    assert payload(reply)["position"] == {"x": 0.0, "y": 0.0}
    assert session.robot.stops == 0
    assert 2 not in session.ids()  # the move has not answered yet


def test_status_shows_the_position_change_during_the_move(session):
    session.start_move()
    robot = session.robot.robot
    for msg_id, x in ((9, 0.25), (10, 0.75)):
        robot.get_position = lambda x=x: (x, 0.0)
        send(session.client_in, status(msg_id))
        assert payload(session.wait_for(msg_id))["position"]["x"] == x


def test_status_during_a_move_leaves_the_move_stoppable(session):
    """A read must not clear the flag that a cancel and a stop act on."""
    session.start_move()
    send(session.client_in, status())
    session.wait_for(9)
    assert session.server._motion_running
    send(session.client_in, CANCEL_MOVE)
    assert session.robot.stopped.wait(5), "the cancel did not reach the move"


def test_stop_after_a_status_ends_the_move_and_gets_its_reply(session):
    session.start_move()
    send(session.client_in, status(), STOP)
    assert session.robot.stopped.wait(5)
    assert session.wait_for(2)["result"]["isError"] is True
    assert session.wait_for(7)["result"]["isError"] is False
    session.wait_for(9)


def test_a_status_read_that_hangs_does_not_hold_up_a_stop(session):
    session.start_move()
    release, entered = threading.Event(), threading.Event()
    robot = session.robot.robot

    def hanging_position():
        entered.set()
        assert release.wait(10)
        return (0.5, 0.0)

    robot.get_position = hanging_position
    send(session.client_in, status())
    assert entered.wait(5)
    send(session.client_in, STOP)
    assert session.robot.stopped.wait(5), "a hung status read held up the stop"
    release.set()
    assert payload(session.wait_for(9))["position"]["x"] == 0.5
    session.wait_for(2)
    session.wait_for(7)


def test_a_second_status_waits_for_the_move_while_one_is_out(session):
    session.start_move()
    release, entered = threading.Event(), threading.Event()
    robot = session.robot.robot

    def hanging_position():
        entered.set()
        assert release.wait(10)
        return (0.5, 0.0)

    robot.get_position = hanging_position
    send(session.client_in, status(9))
    assert entered.wait(5)
    send(session.client_in, status(10), STOP)
    assert session.robot.stopped.wait(5)  # so status 10 has been read
    release.set()
    session.wait_for(9)
    session.wait_for(2)
    session.wait_for(10)
    session.wait_for(7)
    assert session.ids().index(2) < session.ids().index(10)


def test_a_status_in_a_batch_waits_for_the_move(session):
    session.start_move()
    send(session.client_in, [status()])
    send(session.client_in, STOP)
    assert session.robot.stopped.wait(5)
    session.wait_for(9)
    assert session.ids().index(2) < session.ids().index(9)


def test_a_failing_status_read_is_an_error_and_the_move_goes_on(session):
    session.start_move()
    session.robot.robot.get_position = lambda: 1 / 0
    send(session.client_in, status())
    reply = session.wait_for(9)
    assert reply["result"]["isError"] is True
    assert "ZeroDivisionError" in reply["result"]["content"][0]["text"]
    assert session.server._motion_running
    assert session.robot.stops == 0


def test_a_stateless_client_gets_its_status_during_a_move(session):
    session.start_move()
    request = status()
    request["params"]["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
    }
    send(session.client_in, request)
    reply = session.wait_for(9)
    assert reply["result"]["resultType"] == "complete"
    assert payload(reply)["robot_uri"].startswith("mock://")


def test_many_status_reads_never_mix_with_other_replies(session):
    session.start_move()
    for msg_id in range(100, 160):
        send(session.client_in, status(msg_id))
    send(session.client_in, STOP)
    assert session.robot.stopped.wait(5)
    for msg_id in range(100, 160):
        session.wait_for(msg_id)  # _read() fails on a line that is not JSON
    session.wait_for(2)
    session.wait_for(7)


def test_status_before_any_move_is_served_in_turn(session):
    send(session.client_in, INIT, status())
    assert session.wait_for(9)["result"]["isError"] is False

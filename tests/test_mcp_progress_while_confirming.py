"""--confirm-moves: progress notifications keep a call alive while the user decides.

A client that times out idle requests (Cursor: 60 s) would otherwise drop the
tools/call, and the question with it, while the user is still reading it.
Only a request that carries a progressToken gets them, and the robot does
not move until the user says yes.
"""
from __future__ import annotations

import time

import pytest

from apyrobo.mcp_server import build_server
from tests.test_mcp_client_loss import send
from tests.test_mcp_status_during_move import STOP, Session

INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-11-25", "capabilities": {"elicitation": {"form": {}}},
               "clientInfo": {"name": "t", "version": "0"}},
}
ASK_ID = "apyrobo-confirm-1"


def move_with_token(token, msg_id=2):
    return {
        "jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
        "params": {"name": "move_to", "arguments": {"x": 1.0, "y": 0.0},
                   "_meta": {"progressToken": token}},
    }


def answer(action):
    return {"jsonrpc": "2.0", "id": ASK_ID, "result": {"action": action, "content": {}}}


@pytest.fixture
def session():
    server = build_server(robot_uri="mock://confirm-progress", confirm_moves=True)
    server.progress_interval = 0.05
    active = Session(server)
    send(active.client_in, INIT)
    active.wait_for(1)
    yield active
    if not active.client_in.closed:
        active.close()


def progress(session):
    with session.seen:
        return [r for r in session.read_lines if r.get("method") == "notifications/progress"]


def wait_for_progress(session, count, timeout=10):
    with session.seen:
        assert session.seen.wait_for(lambda: len(progress(session)) >= count, timeout), (
            f"fewer than {count} progress notifications; got {session.read_lines}"
        )


def asked(session):
    session.wait_for(ASK_ID)


def test_progress_is_sent_while_waiting_for_the_answer(session):
    send(session.client_in, move_with_token("tok-1"))
    asked(session)
    wait_for_progress(session, 3)
    notes = progress(session)
    assert {n["params"]["progressToken"] for n in notes} == {"tok-1"}
    assert [n["params"]["progress"] for n in notes] == list(range(1, len(notes) + 1))
    message = notes[0]["params"]["message"]
    assert "move_to waiting for your confirmation" in message
    assert "has not moved" in message
    assert not session.robot.moving.is_set()
    assert 2 not in session.ids()


def test_after_a_yes_the_move_continues_the_same_count(session):
    send(session.client_in, move_with_token("tok"))
    asked(session)
    wait_for_progress(session, 2)
    send(session.client_in, answer("accept"))
    assert session.robot.moving.wait(10)
    with session.seen:
        assert session.seen.wait_for(
            lambda: any("running for" in n["params"]["message"] for n in progress(session)), 10
        )
    counts = [n["params"]["progress"] for n in progress(session)]
    assert counts == sorted(set(counts)), f"progress must keep increasing: {counts}"
    send(session.client_in, STOP)
    assert session.robot.stopped.wait(5)
    session.wait_for(2)


@pytest.mark.parametrize("action", ["decline", "cancel"])
def test_progress_ends_with_the_answer_and_the_robot_stays(session, action):
    send(session.client_in, move_with_token("tok"))
    asked(session)
    wait_for_progress(session, 1)
    send(session.client_in, answer(action))
    reply = session.wait_for(2)
    assert reply["result"]["content"][0]["text"]
    with session.seen:
        at_reply = next(i for i, r in enumerate(session.read_lines) if r.get("id") == 2)
    time.sleep(0.3)
    with session.seen:
        after = [r for r in session.read_lines[at_reply:]
                 if r.get("method") == "notifications/progress"]
    assert after == [], f"progress sent after the reply: {after}"
    assert not session.robot.moving.is_set()


def test_no_token_means_no_progress_while_waiting(session):
    message = move_with_token("x")
    del message["params"]["_meta"]
    send(session.client_in, message)
    asked(session)
    time.sleep(0.3)
    assert progress(session) == []
    send(session.client_in, answer("decline"))
    session.wait_for(2)

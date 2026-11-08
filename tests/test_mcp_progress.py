"""notifications/progress during a move, only when the client asks, and never in a stop's way."""
from __future__ import annotations

import time

import pytest

from apyrobo.mcp_server import build_server
from tests.test_mcp_client_loss import INIT, move, send
from tests.test_mcp_status_during_move import CANCEL_MOVE, STOP, Session


def move_with_token(token, msg_id=2, x=1.0):
    message = move(msg_id, x)
    message["params"]["_meta"] = {"progressToken": token}
    return message


@pytest.fixture
def session():
    server = build_server(robot_uri="mock://progress")
    server.progress_interval = 0.05
    active = Session(server)
    yield active
    if not active.client_in.closed:
        active.close()


def progress(session):
    with session.seen:  # a Condition over an RLock, so safe inside wait_for
        return [
            r for r in session.read_lines if r.get("method") == "notifications/progress"
        ]


def wait_for_progress(session, count, timeout=10):
    with session.seen:
        assert session.seen.wait_for(lambda: len(progress(session)) >= count, timeout), (
            f"fewer than {count} progress notifications; got {session.read_lines}"
        )


def start(session, message):
    send(session.client_in, INIT)
    session.wait_for(1)
    send(session.client_in, message)
    assert session.robot.moving.wait(10)


def test_a_move_with_a_progress_token_reports_progress_while_it_runs(session):
    start(session, move_with_token("tok-1", x=3.0))
    wait_for_progress(session, 3)
    notes = progress(session)
    assert {n["params"]["progressToken"] for n in notes} == {"tok-1"}
    counts = [n["params"]["progress"] for n in notes]
    assert counts == sorted(set(counts)), f"progress must increase: {counts}"
    assert "move_to running for" in notes[0]["params"]["message"]
    assert "3.00 m from the target" in notes[0]["params"]["message"]
    assert "id" not in notes[0]
    assert session.robot.stops == 0
    assert 2 not in session.ids()  # still moving: no reply yet


def test_an_integer_token_is_echoed_as_sent(session):
    start(session, move_with_token(42))
    wait_for_progress(session, 1)
    assert progress(session)[0]["params"]["progressToken"] == 42


def test_no_token_means_no_progress(session):
    start(session, move())
    time.sleep(0.3)
    assert progress(session) == []


@pytest.mark.parametrize("token", [True, None, 1.5, {"a": 1}])
def test_a_token_that_is_not_a_string_or_integer_is_ignored(session, token):
    start(session, move_with_token(token))
    time.sleep(0.3)
    assert progress(session) == []


def test_a_stop_still_ends_the_move_and_progress_ends_with_the_reply(session):
    start(session, move_with_token("tok"))
    wait_for_progress(session, 2)
    send(session.client_in, STOP)
    assert session.robot.stopped.wait(5), "the stop did not reach the move"
    reply = session.wait_for(2)
    assert reply["result"]["isError"] is True
    with session.seen:
        lines = list(session.read_lines)
    at_reply = next(i for i, r in enumerate(lines) if r.get("id") == 2)
    time.sleep(0.3)
    with session.seen:
        after = [
            r for r in session.read_lines[at_reply:]
            if r.get("method") == "notifications/progress"
        ]
    assert after == [], f"progress sent after the move's reply: {after}"


def test_a_cancel_still_stops_the_move(session):
    start(session, move_with_token("tok"))
    wait_for_progress(session, 1)
    send(session.client_in, CANCEL_MOVE)
    assert session.robot.stopped.wait(5), "the cancel did not reach the move"


def test_the_next_move_reports_under_its_own_token(session):
    start(session, move_with_token("first"))
    wait_for_progress(session, 1)
    send(session.client_in, STOP)
    session.wait_for(2)
    session.wait_for(7)
    session.robot.moving.clear()
    send(session.client_in, move_with_token("second", msg_id=3))
    assert session.robot.moving.wait(10)
    with session.seen:
        assert session.seen.wait_for(
            lambda: any(
                r["params"]["progressToken"] == "second" for r in progress(session)
            ),
            10,
        )
    send(session.client_in, STOP | {"id": 8})
    session.wait_for(3)
    time.sleep(0.3)
    tokens = [n["params"]["progressToken"] for n in progress(session)]
    first_after_second = tokens.index("second")
    assert "first" not in tokens[first_after_second:]


def test_status_reads_still_answer_while_progress_is_sent(session):
    start(session, move_with_token("tok"))
    wait_for_progress(session, 1)
    send(session.client_in, {
        "jsonrpc": "2.0", "id": 9, "method": "tools/call",
        "params": {"name": "robot_status", "arguments": {}},
    })
    assert session.wait_for(9)["result"]["isError"] is False

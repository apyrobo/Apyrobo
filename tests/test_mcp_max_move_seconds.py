"""apyrobo mcp --max-move-seconds stops a motion call that runs too long."""
from __future__ import annotations

import subprocess
import sys
import time

import pytest

from apyrobo.mcp_server import build_server
from tests.test_mcp_client_loss import SlowRobot, move
from tests.test_mcp_stop_during_move import Client, audit_lines, stop, task, text_of


@pytest.fixture
def make_client():
    clients = []

    def make(**kwargs):
        server = build_server(robot_uri="mock://max-move", **kwargs)
        robot = SlowRobot(server)
        client = Client(server)
        clients.append(client)
        return server, robot, client

    yield make
    for client in clients:
        client.close()


def test_a_move_past_the_limit_is_stopped_and_reported(make_client):
    server, robot, client = make_client(max_move_seconds=0.3)
    started = time.monotonic()
    client.send(move())
    assert robot.moving.wait(10)
    reply = client.reply(2)
    elapsed = time.monotonic() - started
    assert robot.stopped.is_set()
    assert reply["result"]["isError"] is True
    text = text_of(reply)
    assert text.startswith(
        "STOPPED: move_to was still running after --max-move-seconds (0.3 s)"
    )
    assert "Last known position: (" in text
    assert elapsed < 5


def test_the_next_move_gets_its_own_limit(make_client):
    server, robot, client = make_client(max_move_seconds=0.3)
    client.send(move())
    assert text_of(client.reply(2)).startswith("STOPPED:")
    robot.moving.clear()
    client.send(move(5, 3.0))
    assert robot.moving.wait(5), "a timed-out move ended the session's moves"
    assert text_of(client.reply(5)).startswith("STOPPED: move_to was still running")
    assert robot.moves == [(1.0, 0.0), (3.0, 0.0)]


def test_a_move_that_finishes_in_time_is_not_touched(make_client):
    server, robot, client = make_client(max_move_seconds=5)

    def quick_move(x, y, speed=None):
        robot.moves.append((x, y))

    robot.robot.move = quick_move
    client.send(move())
    reply = client.reply(2)
    assert reply["result"]["isError"] is False
    time.sleep(0.2)
    assert robot.stops == 0


def test_without_the_flag_there_is_no_limit(make_client):
    server, robot, client = make_client()
    assert server.max_move_seconds is None
    client.send(move())
    assert robot.moving.wait(10)
    time.sleep(0.5)
    assert not robot.stopped.is_set()
    client.send(stop())
    assert text_of(client.reply(2)).startswith("STOPPED: a stop request arrived")


def test_execute_task_drops_its_remaining_steps(make_client):
    server, robot, client = make_client(max_move_seconds=0.3)
    client.send(task())
    reply = client.reply(3)
    assert reply["result"]["isError"] is True
    assert text_of(reply).startswith("STOPPED: execute_task was still running")
    assert robot.moves == [(1.0, 0.0)]


def test_the_audit_log_says_timed_out(make_client, tmp_path):
    log = tmp_path / "audit.jsonl"
    server, robot, client = make_client(max_move_seconds=0.3, audit_log=log)
    client.send(move())
    client.reply(2)
    decisions = [line["decision"] for line in audit_lines(log)]
    assert decisions == ["started", "timed_out"]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_a_limit_that_is_not_a_positive_number_is_refused(value):
    with pytest.raises(ValueError, match="--max-move-seconds"):
        build_server(robot_uri="mock://max-move", max_move_seconds=value)


def test_the_cli_refuses_a_bad_limit_before_serving():
    result = subprocess.run(
        [sys.executable, "-m", "apyrobo", "mcp", "--max-move-seconds", "0"],
        input="", capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 1
    assert "--max-move-seconds" in result.stderr
    assert result.stdout == ""


def test_the_cli_passes_the_limit_to_the_server():
    result = subprocess.run(
        [sys.executable, "-m", "apyrobo", "mcp", "--max-move-seconds", "90"],
        input="", capture_output=True, text=True, timeout=60,
    )
    assert "max_move_seconds=90" in result.stderr
    assert result.stdout == ""

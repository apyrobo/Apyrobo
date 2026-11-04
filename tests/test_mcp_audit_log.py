"""apyrobo mcp --audit-log: a line for each motion decision, or no motion."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from apyrobo.mcp_server import AuditLog, build_server
from tests.mcp_stdio import talk


def call(name, msg_id=2, **arguments):
    return {
        "jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def initialize(capabilities=None):
    return {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": capabilities or {},
                   "clientInfo": {"name": "t", "version": "0"}},
    }


def lines(path):
    return [json.loads(raw) for raw in path.read_text().splitlines()]


def decisions(path):
    return [(e["tool"], e["decision"]) for e in lines(path)]


def is_error(reply):
    return reply["result"]["isError"]


@pytest.fixture
def log(tmp_path):
    return tmp_path / "audit.jsonl"


def make(log, **kwargs):
    kwargs.setdefault("agent", None)
    server = build_server(audit_log=log, **kwargs)
    server.handle(initialize())
    return server


def test_a_plain_move_logs_started_then_ran(log):
    server = make(log, max_speed=1.0)
    assert not is_error(server.handle(call("move_to", x=1.0, y=0.0, speed=0.5)))
    started, ran = lines(log)
    assert started["decision"] == "started"
    assert ran["decision"] == "ran"
    assert ran["tool"] == "move_to"
    assert ran["arguments"] == {"x": 1.0, "y": 0.0, "speed": 0.5}
    assert ran["enforced_speed"] == 0.5
    assert ran["policy"] == server.toolbox.enforcer.policy.name
    assert ran["time"].endswith("+00:00")


def test_a_clamped_move_logs_the_speed_that_was_enforced(log):
    server = make(log, max_speed=0.5)
    server.handle(call("move_to", x=1.0, y=0.0, speed=9.0))
    last = lines(log)[-1]
    assert last["decision"] == "clamped"
    assert last["arguments"]["speed"] == 9.0
    assert last["enforced_speed"] == 0.5


def test_the_logged_speed_is_the_one_the_robot_was_given(log):
    server = make(log, policy="strict")  # its limit lowers the speed
    given = []
    server.toolbox.enforcer._robot.move = lambda **kw: given.append(kw["speed"])
    server.handle(call("move_to", x=1.0, y=0.0, speed=9.0))
    assert lines(log)[-1]["enforced_speed"] == given[0] == 0.5


def test_no_speed_asked_for_logs_none(log):
    make(log).handle(call("move_to", x=1.0, y=0.0))
    assert lines(log)[-1]["enforced_speed"] is None


def test_the_log_can_be_rotated_while_the_server_runs(log):
    server = make(log)
    server.handle(call("stop"))
    log.rename(log.with_suffix(".1"))
    server.handle(call("stop"))
    assert len(lines(log)) == 1


def test_a_clamped_rotation_is_logged_too(log):
    server = make(log)
    limit = server.toolbox.enforcer.policy.max_angular_speed
    server.handle(call("rotate", angle_rad=1.0, speed=limit + 5))
    assert lines(log)[-1]["decision"] == "clamped"
    assert lines(log)[-1]["enforced_speed"] == limit


@pytest.mark.parametrize("target", [(3.0, 3.0), (6.0, 6.0)], ids=["inside", "across"])
def test_a_refused_move_says_so_and_enforces_no_speed(log, target):
    server = make(log, no_go_zones=[{"x_min": 2, "y_min": 2, "x_max": 4, "y_max": 4}])
    reply = server.handle(call("move_to", x=target[0], y=target[1]))
    assert is_error(reply)
    last = lines(log)[-1]
    assert last["decision"] == "refused"
    assert last["enforced_speed"] is None
    assert "REFUSED" in last["detail"]


def test_a_move_outside_the_allowed_areas_is_refused_in_the_log(log):
    server = make(log, allowed_areas=[{"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}])
    server.handle(call("move_to", x=5.0, y=5.0))
    assert decisions(log)[-1] == ("move_to", "refused")


def test_bad_arguments_are_a_failure_not_a_refusal(log):
    server = make(log)
    server.handle(call("move_to", x=1.0))
    assert decisions(log)[-1] == ("move_to", "failed")
    assert "'y'" in lines(log)[-1]["detail"]


def test_an_adapter_error_is_a_failure(log):
    server = make(log)

    def broken(**_):
        raise RuntimeError("Nav2 aborted the goal")

    server.toolbox.enforcer._robot.move = broken
    assert is_error(server.handle(call("move_to", x=1.0, y=0.0)))
    assert decisions(log)[-1] == ("move_to", "failed")
    assert "Nav2 aborted" in lines(log)[-1]["detail"]


def test_read_only_tools_are_not_logged(log):
    server = make(log)
    server.handle(call("robot_status"))
    server.handle(call("safety_report"))
    assert lines(log) == []


def test_stop_is_logged_without_a_started_line(log):
    server = make(log)
    server.handle(call("stop"))
    assert decisions(log) == [("stop", "ran")]


def test_nan_in_the_arguments_keeps_the_line_strict_json(log):
    server = make(log)
    server.handle(call("move_to", x=float("nan"), y=0.0))
    for raw in log.read_text().splitlines():
        json.loads(raw, parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
    assert lines(log)[-1]["arguments"]["x"] == "nan"


def test_a_long_error_is_cut_short(log):
    server = make(log)

    def broken(**_):
        raise RuntimeError("x" * 5000)

    server.toolbox.enforcer._robot.move = broken
    server.handle(call("move_to", x=1.0, y=0.0))
    assert len(lines(log)[-1]["detail"]) == 500


def test_each_session_appends(log):
    make(log).handle(call("stop"))
    make(log).handle(call("stop"))
    assert len(lines(log)) == 2


def test_a_user_decline_is_logged_and_nothing_starts(log):
    server = build_server(audit_log=log, confirm_moves=True, agent=None)
    server.handle(initialize({"elicitation": {"form": {}}}))
    answers = iter([{"action": "decline"}])
    server._elicit = lambda *_: next(answers)
    server._lines, server._out = iter(()), object()
    reply = server.handle(call("move_to", x=1.0, y=0.0))
    assert not is_error(reply)
    assert decisions(log) == [("move_to", "declined")]


def test_no_elicitation_support_is_logged_as_refused(log):
    server = build_server(audit_log=log, confirm_moves=True, agent=None)
    server.handle(initialize())
    assert is_error(server.handle(call("move_to", x=1.0, y=0.0)))
    assert decisions(log) == [("move_to", "refused")]


def test_a_modern_client_asked_to_confirm_leaves_no_line_until_it_answers(log):
    server = build_server(audit_log=log, confirm_moves=True, agent=None)
    meta = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {"elicitation": {"form": {}}},
    }
    params = {"name": "move_to", "arguments": {"x": 1.0, "y": 0.0}, "_meta": meta}
    asked = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": params})
    assert asked["result"]["resultType"] == "input_required"
    assert lines(log) == []
    answer = {
        **params,
        "requestState": asked["result"]["requestState"],
        "inputResponses": {"confirm_move": {"action": "decline"}},
    }
    server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": answer})
    assert decisions(log) == [("move_to", "declined")]


class Broken:
    path = "audit.jsonl"

    def record(self, entry):
        raise OSError(28, "No space left on device")


def test_motion_is_refused_when_the_log_cannot_be_written():
    server = build_server(agent=None)
    server.audit_log = Broken()
    moves = []
    server.toolbox.enforcer._robot.move = lambda **kw: moves.append(kw)
    for request in (call("move_to", x=1.0, y=0.0), call("rotate", angle_rad=1.0)):
        reply = server.handle(request)
        assert is_error(reply)
        assert "No space left" in reply["result"]["content"][0]["text"]
    assert moves == []


def test_stop_still_works_when_the_log_cannot_be_written():
    server = build_server(agent=None)
    server.audit_log = Broken()
    stops = []
    server.toolbox.enforcer._robot.stop = lambda: stops.append(1)
    assert not is_error(server.handle(call("stop")))
    assert stops == [1]


def test_motion_is_refused_again_after_the_log_breaks_during_a_move(log):
    server = make(log)
    audit = server.audit_log

    def breaks_mid_move(**_):
        server.audit_log = Broken()

    server.toolbox.enforcer._robot.move = breaks_mid_move
    assert not is_error(server.handle(call("move_to", x=1.0, y=0.0)))
    assert is_error(server.handle(call("move_to", x=2.0, y=0.0)))
    server.audit_log = audit
    assert decisions(log) == [("move_to", "started")]


def test_the_motion_after_a_client_loss_is_refused_in_the_log(log):
    server = make(log)
    server.client_gone = True
    assert is_error(server.handle(call("move_to", x=1.0, y=0.0)))
    assert decisions(log) == [("move_to", "refused")]


def test_a_full_disk_refuses_motion():
    if not os.path.exists("/dev/full"):
        pytest.skip("needs /dev/full")
    server = build_server(audit_log="/dev/full", agent=None)
    moves = []
    server.toolbox.enforcer._robot.move = lambda **kw: moves.append(kw)
    assert is_error(server.handle(call("move_to", x=1.0, y=0.0)))
    assert moves == []


def test_a_pipe_needs_no_sync(tmp_path):
    read_end, write_end = os.pipe()
    with os.fdopen(read_end) as reader:
        audit = AuditLog(f"/dev/fd/{write_end}")
        audit.record({"tool": "stop"})
        os.close(write_end)
        assert json.loads(reader.readline()) == {"tool": "stop"}


def test_a_real_server_writes_the_log_and_reports_it_at_start(tmp_path, log):
    cmd = [sys.executable, "-m", "apyrobo.cli", "mcp"]
    cmd += ["--audit-log", str(log), "--max-speed", "0.5"]
    done = talk(
        cmd,
        "\n".join(json.dumps(m) for m in (initialize(), call("move_to", x=1.0, y=0.0, speed=9)))
        + "\n",
        last_id=2,
    )
    assert f"audit_log={log}" in done.stderr
    assert decisions(log) == [("move_to", "started"), ("move_to", "clamped")]


def test_a_path_that_cannot_be_written_stops_the_start(tmp_path):
    done = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "mcp", "--audit-log", str(tmp_path)],
        input="", capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 1
    assert "Traceback" not in done.stderr
    assert "Error:" in done.stderr

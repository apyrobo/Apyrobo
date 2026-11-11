"""--confirm-moves: a task refused before the question carries a refusal code.

Without --confirm-moves a refused execute_task step carries structuredContent
(code, policy, violation, step, steps_completed, steps_total). With it, a
task whose plan the policy refuses is refused before the user is asked; that
refusal carries the same fields, with steps_completed 0 since nothing ran.
The text is unchanged.
"""

from __future__ import annotations

import io
import json

import pytest

from apyrobo.mcp_server import build_server
from apyrobo.safety.enforcer import SafetyEnforcer, SafetyPolicy

ZONE = {"x_min": 2.0, "y_min": 2.0, "x_max": 4.0, "y_max": 4.0}
ELICIT = {"elicitation": {"form": {}}}
META = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": ELICIT,
}


@pytest.fixture
def server():
    return build_server(max_speed=0.2, no_go_zones=[ZONE], confirm_moves=True)


def task_call(server, task):
    params = {"name": "execute_task", "arguments": {"task": task}, "_meta": META}
    return server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params})


def legacy_reply(server, task, version="2025-11-25"):
    messages = [
        {"jsonrpc": "2.0", "id": 0, "method": "initialize",
         "params": {"protocolVersion": version, "capabilities": ELICIT}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "execute_task", "arguments": {"task": task}}},
    ]
    stdin = io.StringIO("".join(json.dumps(m) + "\n" for m in messages))
    stdout = io.StringIO()
    server.serve(stdin, stdout)
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert all(m.get("method") != "elicitation/create" for m in lines)
    return next(m for m in lines if m.get("id") == 1)["result"]


def test_a_no_go_target_refused_without_asking_carries_the_code(server):
    result = task_call(server, "go to (3, 3)")["result"]
    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured["status"] == "refused"
    assert structured["code"] == "collision_zone"
    assert structured["policy"] == server.toolbox.enforcer.policy.name
    assert structured["violation"]["zone"] == ZONE
    assert structured["step"] == 1
    assert structured["steps_completed"] == 0
    assert structured["steps_total"] >= 1
    assert "at step 1: Position (3, 3) is inside collision zone" in result["content"][0]["text"]
    json.dumps(result)


def test_a_refused_later_step_names_that_step(server):
    result = task_call(server, "deliver the package from (1, 0) to (3, 3)")["result"]
    structured = result["structuredContent"]
    assert structured["code"] == "collision_zone"
    assert structured["step"] == 3
    assert structured["steps_completed"] == 0
    assert structured["steps_total"] == 4


def test_a_route_through_a_zone_carries_the_path_code(server):
    result = task_call(server, "go to (6, 6)")["result"]
    structured = result["structuredContent"]
    assert structured["code"] == "collision_zone_path"
    assert structured["violation"]["zone"] == ZONE
    assert structured["violation"]["requested_position"] == [6.0, 6.0]


def test_legacy_session_gets_the_code_and_older_versions_text_only():
    task = "deliver the package from (1, 0) to (3, 3)"
    new = legacy_reply(build_server(max_speed=0.2, no_go_zones=[ZONE], confirm_moves=True), task)
    old = legacy_reply(
        build_server(max_speed=0.2, no_go_zones=[ZONE], confirm_moves=True), task, "2025-03-26"
    )
    assert new["structuredContent"]["step"] == 3
    assert "structuredContent" not in old
    assert new["content"] == old["content"]


def test_an_allowed_task_and_an_unplannable_one_carry_no_code(server):
    asked = task_call(server, "go to (1, 1)")["result"]
    assert asked["resultType"] == "input_required"
    unread = task_call(server, "do not go to (1,1)")["result"]
    assert unread["isError"] is True
    assert "structuredContent" not in unread


# -- the enforcer side ---------------------------------------------------------


def _enforcer(**policy):
    from apyrobo.core.robot import Robot

    return SafetyEnforcer(Robot.discover("mock://tb"), policy=SafetyPolicy(name="t", **policy))


@pytest.mark.parametrize(
    "target, code",
    [((3.0, 3.0), "collision_zone"), ((6.0, 6.0), "collision_zone_path")],
)
def test_move_violation_matches_move_problem_and_records_nothing(target, code):
    enforcer = _enforcer(collision_zones=[ZONE])
    violation = enforcer.move_violation(*target)
    assert violation["type"] == code
    assert violation["message"] == enforcer.move_problem(*target)
    assert enforcer.violations == []


def test_move_violation_outside_allowed_area_and_non_finite():
    area = {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}
    enforcer = _enforcer(allowed_areas=[area])
    outside = enforcer.move_violation(5.0, 5.0)
    assert outside["type"] == "outside_allowed_area"
    assert outside["allowed_areas"] == [area]
    assert outside["message"] == enforcer.move_problem(5.0, 5.0)
    bad = enforcer.move_violation(float("nan"), 0.0)
    assert bad["type"] == "invalid_command"
    assert bad["message"] == enforcer.move_problem(float("nan"), 0.0)
    assert enforcer.move_violation(0.5, 0.5) is None
    assert enforcer.violations == []

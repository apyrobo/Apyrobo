"""--confirm-moves on execute_task asks about the plan, not the words.

The question lists each step that will run, with its target and the speed
the policy allows. A task whose plan the policy refuses is refused without
asking, before any step runs. The plan the user said yes to is the plan that
runs: the planner is not asked again between the question and the move.
"""

from __future__ import annotations

import io
import json

import pytest

from apyrobo.mcp_server import CONFIRM_INPUT_KEY, INVALID_PARAMS, build_server

ZONE = {"x_min": 2.0, "y_min": 2.0, "x_max": 4.0, "y_max": 4.0}
ELICIT = {"elicitation": {"form": {}}}
META = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": ELICIT,
}


@pytest.fixture
def server():
    return build_server(max_speed=0.2, no_go_zones=[ZONE], confirm_moves=True)


def task_call(server, task, **extra):
    params = {"name": "execute_task", "arguments": {"task": task}, "_meta": META, **extra}
    return server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params})


def answer(server, task, state, action="accept"):
    return task_call(
        server, task, inputResponses={CONFIRM_INPUT_KEY: {"action": action}}, requestState=state
    )


def question(result):
    assert result["resultType"] == "input_required", result
    return result["inputRequests"][CONFIRM_INPUT_KEY]["params"]["message"]


def position(server):
    return server.toolbox.enforcer.get_position()


def count_plans(server, monkeypatch):
    calls = []
    plan = server.toolbox.agent.plan

    def counted(*args, **kwargs):
        calls.append(args[0])
        return plan(*args, **kwargs)

    monkeypatch.setattr(server.toolbox.agent, "plan", counted)
    return calls


# -- the question ------------------------------------------------------------


def test_the_question_lists_each_step_with_target_and_clamped_speed(server):
    message = question(task_call(server, "go to (1, 1)")["result"])
    assert "'go to (1, 1)'" in message
    assert "1. Drive to x=1.0, y=1.0 at 0.2 m/s (requested 0.5, clamped" in message


def test_a_delivery_shows_every_step_in_order(server):
    message = question(task_call(server, "deliver the package from (1, 0) to (1, 1)")["result"])
    lines = message.splitlines()[1:]
    assert lines[0].startswith("1. Drive to x=1.0, y=0.0 at 0.2 m/s")
    assert lines[1].startswith("2. Pick Object")
    assert lines[2].startswith("3. Drive to x=1.0, y=1.0 at 0.2 m/s")
    assert lines[3].startswith("4. Place Object")


def test_a_turn_shows_its_angle(server):
    message = question(task_call(server, "turn left 90 degrees")["result"])
    assert "1. Turn by 1.57" in message
    assert "rad/s" in message


# -- refused without asking ----------------------------------------------------


def test_a_no_go_target_is_refused_without_asking(server):
    start = position(server)
    result = task_call(server, "go to (3, 3)")["result"]
    assert result["resultType"] == "complete"
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert text.startswith("REFUSED by safety policy")
    assert "at step 1: Position (3, 3) is inside collision zone" in text
    assert "The robot did not move." in text
    assert position(server) == start


def test_a_refused_later_step_stops_the_whole_task_before_it_starts(server):
    # Without the check, step 1 would drive to (1, 0) with no yes.
    start = position(server)
    result = task_call(server, "deliver the package from (1, 0) to (3, 3)")["result"]
    assert result["isError"] is True
    assert "at step 3: Position (3, 3) is inside collision zone" in result["content"][0]["text"]
    assert position(server) == start


def test_a_route_through_a_zone_is_refused_without_asking(server):
    start = position(server)
    result = task_call(server, "go to (6, 6)")["result"]
    assert result["isError"] is True
    assert "crosses collision zone" in result["content"][0]["text"]
    assert position(server) == start


def test_a_task_the_planner_would_misread_is_refused_without_asking(server):
    start = position(server)
    result = task_call(server, "do not go to (1,1)")["result"]
    assert result["resultType"] == "complete"
    assert result["isError"] is True
    assert position(server) == start


def test_without_confirm_moves_a_refused_task_still_runs_up_to_the_refusal():
    plain = build_server(max_speed=0.2, no_go_zones=[ZONE])
    reply = plain.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "execute_task",
                   "arguments": {"task": "deliver the package from (1, 0) to (3, 3)"}},
    })
    text = reply["result"]["content"][0]["text"]
    assert json.loads(text.split("\n\n", 1)[1])["position"] == {"x": 1.0, "y": 0.0}


# -- the plan that runs ------------------------------------------------------


def test_a_yes_runs_the_plan_shown_and_plans_once(server, monkeypatch):
    calls = count_plans(server, monkeypatch)
    first = task_call(server, "go to (1, 1)")["result"]
    question(first)
    result = answer(server, "go to (1, 1)", first["requestState"])["result"]
    assert result["isError"] is False, result
    payload = json.loads(result["content"][0]["text"])
    assert payload["status"] == "completed"
    assert payload["plan"][0]["parameters"]["x"] == 1.0
    assert position(server) == pytest.approx((1.0, 1.0))
    assert len(calls) == 1
    assert server.toolbox.approved_plan is None


def test_the_approved_plan_runs_even_if_the_planner_would_now_answer_differently(
    server, monkeypatch
):
    first = task_call(server, "go to (1, 1)")["result"]
    question(first)

    def other_plan(*args, **kwargs):
        raise AssertionError("planned again after the yes")

    monkeypatch.setattr(server.toolbox.agent, "plan", other_plan)
    result = answer(server, "go to (1, 1)", first["requestState"])["result"]
    assert result["isError"] is False, result
    assert position(server) == pytest.approx((1.0, 1.0))


@pytest.mark.parametrize("action", ["decline", "cancel"])
def test_a_no_runs_nothing(server, action):
    start = position(server)
    first = task_call(server, "go to (1, 1)")["result"]
    result = answer(server, "go to (1, 1)", first["requestState"], action)["result"]
    assert json.loads(result["content"][0]["text"])["status"] == "declined"
    assert position(server) == start
    assert server.toolbox.approved_plan is None


def test_a_yes_cannot_be_resent(server):
    first = task_call(server, "go to (1, 1)")["result"]
    answer(server, "go to (1, 1)", first["requestState"])
    again = answer(server, "go to (1, 1)", first["requestState"])
    assert again["error"]["code"] == INVALID_PARAMS


def test_a_yes_for_one_task_does_not_run_another(server):
    start = position(server)
    first = task_call(server, "go to (1, 1)")["result"]
    other = answer(server, "go to (1, 0)", first["requestState"])
    assert other["error"]["code"] == INVALID_PARAMS
    assert position(server) == start


def test_a_forged_state_is_rejected(server):
    task_call(server, "go to (1, 1)")
    reply = answer(server, "go to (1, 1)", "123.abc.def")
    assert reply["error"]["code"] == INVALID_PARAMS


# -- initialize-era sessions ---------------------------------------------------


def serve_script(server, messages):
    head = [
        {"jsonrpc": "2.0", "id": 0, "method": "initialize",
         "params": {"protocolVersion": "2025-11-25", "capabilities": ELICIT}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]
    stdin = io.StringIO("".join(json.dumps(m) + "\n" for m in head + messages))
    stdout = io.StringIO()
    server.serve(stdin, stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def legacy_call(task):
    return {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "execute_task", "arguments": {"task": task}}}


def legacy_answer(action):
    return {"jsonrpc": "2.0", "id": "apyrobo-confirm-1",
            "result": {"action": action, "content": {}}}


def test_legacy_question_shows_the_plan_and_a_yes_runs_it_once(server, monkeypatch):
    calls = count_plans(server, monkeypatch)
    lines = serve_script(server, [legacy_call("go to (1, 1)"), legacy_answer("accept")])
    request = lines[1]
    assert request["method"] == "elicitation/create"
    assert "1. Drive to x=1.0, y=1.0 at 0.2 m/s" in request["params"]["message"]
    reply = next(m for m in lines if m.get("id") == 1)
    assert reply["result"]["isError"] is False
    assert position(server) == pytest.approx((1.0, 1.0))
    assert len(calls) == 1


def test_legacy_refused_plan_is_refused_without_asking(server):
    start = position(server)
    lines = serve_script(server, [legacy_call("deliver the package from (1, 0) to (3, 3)")])
    assert all(m.get("method") != "elicitation/create" for m in lines)
    reply = next(m for m in lines if m.get("id") == 1)
    assert reply["result"]["isError"] is True
    assert "at step 3" in reply["result"]["content"][0]["text"]
    assert position(server) == start


# -- speed through Nav2 ----------------------------------------------------------


def test_through_nav2_the_question_does_not_promise_a_speed(server):
    server.toolbox.robot._adapter.speed_limit_applies = False
    message = question(task_call(server, "go to (1, 1)")["result"])
    assert "Nav2's controller" in message
    assert "0.2 m/s (requested" not in message
    move = server.handle({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "move_to", "arguments": {"x": 1.0, "y": 0.0, "speed": 0.1},
                   "_meta": META},
    })["result"]
    assert "Nav2's controller" in question(move)

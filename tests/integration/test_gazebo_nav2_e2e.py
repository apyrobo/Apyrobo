"""
End-to-end integration test: natural-language task → rule-based plan →
Nav2 NavigateToPose → a physics-simulated robot navigates in Gazebo.

This is the flagship-stack proof. Where test_gazebo_turtlebot.py shows the
adapter's cmd_vel fallback moves a physics robot, this test runs the whole
pipeline a user touches:

    Agent(provider="rule").execute("navigate to (-1.2, -0.5)", robot)
      → plan: navigate_to skill with extracted coordinates
      → SkillExecutor → builtins navigate_to → robot.move()
      → ROS2Adapter prefers the live Nav2 NavigateToPose action
      → slam_toolbox + Nav2 drive the TurtleBot3 burger through
        turtlebot3_world, with physics in the loop.

Run inside the Docker 'gazebo-nav' profile:

    docker compose -f docker/docker-compose.yml --profile gazebo-nav up \
        --abort-on-container-exit --exit-code-from gazebo-nav-test

Assertions are progress-shaped, not exact-pose-shaped: SLAM and physics are
in the loop, so the point is that the NL task measurably gets the robot to
the goal region via Nav2 — not centimeter repeatability.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import subprocess
import sys
import tempfile
from typing import Any

import pytest

# Gated behind APYROBO_GAZEBO_NAV, set only by the compose 'gazebo-nav-test'
# service — mirrors the APYROBO_GAZEBO gate in test_gazebo_turtlebot.py and
# keeps this file out of the plain integration job and the cmd_vel-only
# gazebo job (where no Nav2 server exists).
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("APYROBO_GAZEBO_NAV"),
        reason="Nav2-in-Gazebo sim not present — set APYROBO_GAZEBO_NAV=1 "
        "(docker compose 'gazebo-nav' profile) to run",
    ),
]

# Spawn pose in gazebo_nav_boot.sh and the goal the NL task names: a short
# hop through free corridor, well inside the default 60 s skill timeout.
START = (-2.0, -0.5)
GOAL = (-1.2, -0.5)
TASK = f"navigate to ({GOAL[0]}, {GOAL[1]})"


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@pytest.fixture(scope="module")
def robot():
    from apyrobo.core.robot import Robot
    from apyrobo.core.ros2_bridge import _ROS2NodeManager

    # Nav2 activation after SLAM bringup can be slow on CI — wait generously
    # for the action server; the compose healthcheck has already seen it.
    bot = Robot.discover(
        "ros2://burger",
        nav2_server_wait_sec=90.0,
        odom_wait_sec=60.0,
    )
    yield bot
    with contextlib.suppress(Exception):
        _ROS2NodeManager.shutdown()


@pytest.mark.integration
def test_nav2_action_server_discovered(robot):
    """The adapter found the live NavigateToPose server (not the fallback)."""
    adapter = robot._adapter
    assert adapter._has_nav2, (
        "NavigateToPose action server not found — Nav2 bringup failed or "
        "the adapter fell back to cmd_vel. Check gazebo-nav-sim logs."
    )
    assert adapter._has_odom, "No /odom from Gazebo — sim not up"


@pytest.mark.integration
def test_nl_task_plans_navigate_skill(robot):
    """The rule agent turns the NL task into navigate_to with the coords."""
    from apyrobo.skills.agent import Agent

    graph = Agent(provider="rule").plan(TASK, robot)
    order = graph.get_execution_order()
    nav_steps = [s for s in order if "navigate" in s.name.lower() or "navigate" in s.skill_id.lower()]
    assert nav_steps, f"no navigate skill planned for {TASK!r}: {[s.name for s in order]}"
    # Runtime parameters live on the graph (provider coords merged over the
    # catalog defaults), not on Skill.parameters — same lookup the executor
    # uses (SkillGraph.get_parameters).
    params = graph.get_parameters(nav_steps[-1].skill_id)
    assert params.get("x") == GOAL[0] and params.get("y") == GOAL[1], (
        f"coordinates not extracted into the plan: {params}"
    )


@pytest.mark.integration
def test_nl_task_navigates_the_robot_via_nav2(robot):
    """The whole pipeline: NL task in, robot at the goal region out."""
    from apyrobo.core.schemas import TaskStatus
    from apyrobo.skills.agent import Agent

    adapter = robot._adapter
    start = adapter.position
    d_before = _dist(start, GOAL)

    agent = Agent(provider="rule")
    result = agent.execute(task=TASK, robot=robot)

    assert result.status == TaskStatus.COMPLETED, (
        f"NL task did not complete: {result.status} error={result.error!r}"
    )

    end = adapter.position
    d_after = _dist(end, GOAL)
    travelled = _dist(start, end)

    assert travelled > 0.3, (
        f"robot barely moved (Δ={travelled:.2f} m) — Nav2 accepted the goal "
        f"but nothing drove the physics robot"
    )
    assert d_after < d_before - 0.3, (
        f"no progress toward goal: {d_before:.2f} m → {d_after:.2f} m "
        f"(start={start}, end={end}, goal={GOAL})"
    )
    assert d_after < 0.5, (
        f"did not reach the goal region: {d_after:.2f} m away "
        f"(end={end}, goal={GOAL})"
    )


# The MCP leg drives the robot back to its spawn pose through `apyrobo mcp`,
# the way Claude, Cursor or any MCP client would, with a speed limit and a
# no-go zone set on the command line. The zone sits away from the route.
MCP_MAX_SPEED = 0.22
MCP_NO_GO = (0.0, 0.0, 1.0, 1.0)
MCP_NO_GO_TARGET = (0.5, 0.5)


class _StdioMCPClient:
    """Newline-delimited JSON-RPC over the server's stdin/stdout."""

    def __init__(self, cmd: list[str], stderr: Any) -> None:
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True
        )
        self._next_id = 1

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        msg_id = self._next_id
        self._next_id += 1
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}) + "\n"
        )
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        assert line, f"apyrobo mcp closed the connection during {method}"
        reply = json.loads(line)
        assert reply.get("id") == msg_id, f"unexpected reply {reply!r}"
        assert "error" not in reply, f"{method} failed: {reply['error']}"
        return reply["result"]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        payload = json.loads(result["content"][0]["text"]) if not result.get("isError") else {
            "error": result["content"][0]["text"]
        }
        return bool(result.get("isError")), payload

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


@pytest.mark.integration
def test_mcp_client_drives_the_robot_via_nav2(robot):
    """An MCP client drives the Gazebo robot through `apyrobo mcp --robot ros2://`.

    Same Nav2 stack as above, reached over MCP instead of the Python API:
    an over-limit move runs with the speed clamp reported, Nav2 takes the
    goal and the physics robot gets there; a target in the no-go zone is
    refused and the robot does not move.
    """
    adapter = robot._adapter
    start = adapter.position
    target = START
    d_before = _dist(start, target)
    zone = ",".join(str(v) for v in MCP_NO_GO)
    cmd = [
        sys.executable, "-m", "apyrobo", "--verbose", "mcp",
        "--robot", "ros2://burger",
        "--max-speed", str(MCP_MAX_SPEED),
        "--no-go", zone,
    ]
    with tempfile.TemporaryFile(mode="w+") as log:
        client = _StdioMCPClient(cmd, stderr=log)
        try:
            init = client.request("initialize", {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "apyrobo-nav2-e2e", "version": "1.0"},
            })
            assert init["serverInfo"]["name"]
            tools = {t["name"] for t in client.request("tools/list", {})["tools"]}
            assert {"move_to", "stop", "safety_report"} <= tools

            is_error, moved = client.call_tool(
                "move_to", {"x": target[0], "y": target[1], "speed": 1.0}
            )
            assert not is_error, f"move_to failed over MCP: {moved}"
            assert moved["safety_interventions"], (
                f"1.0 m/s against a {MCP_MAX_SPEED} m/s limit was not reported as clamped: {moved}"
            )

            mid = adapter.position
            reported = (moved["position"]["x"], moved["position"]["y"])
            assert _dist(reported, mid) < 0.3, (
                f"MCP reported {reported} but odometry says {mid}"
            )
            assert not [i for i in moved["safety_interventions"]
                        if i.get("type") == "watchdog_triggered"], (
                f"odometry watchdog fired on a healthy Nav2 move: {moved}"
            )
            d_after = _dist(mid, target)
            assert _dist(start, mid) > 0.3, (
                f"robot barely moved over MCP (start={start}, end={mid}, target={target})"
            )
            assert d_after < d_before - 0.3 and d_after < 0.5, (
                f"MCP move did not reach the target region: {d_before:.2f} m → "
                f"{d_after:.2f} m (end={mid}, target={target})"
            )

            is_error, refused = client.call_tool(
                "move_to", {"x": MCP_NO_GO_TARGET[0], "y": MCP_NO_GO_TARGET[1]}
            )
            assert is_error and "REFUSED" in refused["error"], (
                f"no-go target was not refused over MCP: {refused}"
            )
            assert _dist(mid, adapter.position) < 0.15, "robot moved after a refused command"

            is_error, _ = client.call_tool("stop", {})
            assert not is_error
            is_error, report = client.call_tool("safety_report", {})
            assert not is_error and report, "safety_report returned nothing"
        finally:
            client.close()
        log.seek(0)
        server_log = log.read()
    assert "Nav2: sending goal" in server_log, (
        "apyrobo mcp did not send the move through Nav2 (fell back to cmd_vel?). "
        "Server log tail:\n" + server_log[-2000:]
    )


@pytest.mark.integration
def test_stop_after_nav_is_safe(robot):
    """stop() stays safe and the adapter queryable after Nav2 activity."""
    robot.stop()
    robot.stop()  # idempotent
    pos = robot.get_position()
    assert isinstance(pos, tuple) and len(pos) == 2
    assert isinstance(robot._adapter.get_health(), dict)

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
import threading
import time
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
    nav_steps = [
        s for s in order if "navigate" in s.name.lower() or "navigate" in s.skill_id.lower()
    ]
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
# Well under the controller's own top speed (0.26 m/s in nav2_bringup's DWB
# params), so a limit that did not reach Nav2 would show in the measured peak.
MCP_MAX_SPEED = 0.1
# The limit governs what Nav2's controller commands, so that is what the
# test holds tight: the controller's output may only round over the limit.
MCP_COMMAND_TOLERANCE = 0.01
# /odom is wheel odometry with physics in the loop: a stall-and-recover can
# spike a single reading over the limit (0.132 m/s seen in CI) while the
# commands stayed under it. Odometry is checked loosely, over a short window.
MCP_ODOM_TOLERANCE = 0.03
MCP_ODOM_WINDOW_S = 0.5
# Nav2's bringup remaps the controller's output to /cmd_vel_nav and feeds it
# through the velocity smoother to /cmd_vel. Without a smoother the
# controller publishes /cmd_vel itself.
NAV2_CONTROLLER_TOPICS = ("/cmd_vel_nav", "/cmd_vel")
MCP_NO_GO = (0.0, 0.0, 1.0, 1.0)
MCP_NO_GO_TARGET = (0.5, 0.5)


@contextlib.contextmanager
def _odom_speeds(adapter: Any):
    """Collect (time, forward speed) from /odom while the block runs."""
    from nav_msgs.msg import Odometry

    from apyrobo.core.ros2_bridge import _sensor_qos

    speeds: list[tuple[float, float]] = []
    sub = adapter._node.create_subscription(
        Odometry, "/odom",
        lambda msg: speeds.append((time.monotonic(), abs(msg.twist.twist.linear.x))),
        _sensor_qos(),
    )
    try:
        yield speeds
    finally:
        with contextlib.suppress(Exception):
            adapter._node.destroy_subscription(sub)


def _controller_topic(adapter: Any) -> str:
    """The topic Nav2's controller publishes its velocity commands on."""
    for topic in NAV2_CONTROLLER_TOPICS:
        if adapter._node.count_publishers(topic) > 0:
            return topic
    return NAV2_CONTROLLER_TOPICS[-1]


@contextlib.contextmanager
def _commanded_speeds(adapter: Any, topic: str):
    """Collect the forward speed Nav2's controller commands while the block runs."""
    from geometry_msgs.msg import Twist

    speeds: list[float] = []
    sub = adapter._node.create_subscription(
        Twist, topic, lambda msg: speeds.append(abs(msg.linear.x)), 10,
    )
    try:
        yield speeds
    finally:
        with contextlib.suppress(Exception):
            adapter._node.destroy_subscription(sub)


def _peak_moving_average(samples: list[tuple[float, float]], window_s: float) -> float:
    """Highest mean speed over any `window_s` stretch of (time, speed) samples."""
    peak = 0.0
    start = 0
    total = 0.0
    for end, (t, v) in enumerate(samples):
        total += v
        while samples[start][0] < t - window_s:
            total -= samples[start][1]
            start += 1
        peak = max(peak, total / (end - start + 1))
    return peak


class _StdioMCPClient:
    """Newline-delimited JSON-RPC over the server's stdin/stdout."""

    def __init__(self, cmd: list[str], stderr: Any) -> None:
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True
        )
        self._next_id = 1

    def write(self, message: dict[str, Any]) -> None:
        """Send one message without waiting for an answer."""
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
        self.proc.stdin.flush()

    def send(self, method: str, params: dict[str, Any]) -> int:
        """Send a request and return its id; read the reply with `reply_to`."""
        msg_id = self._next_id
        self._next_id += 1
        self.write({"id": msg_id, "method": method, "params": params})
        return msg_id

    def reply_to(self, msg_id: int, method: str) -> dict[str, Any]:
        """Read the next reply, which has to be the one for `msg_id`."""
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        assert line, f"apyrobo mcp closed the connection during {method}"
        reply = json.loads(line)
        assert reply.get("id") == msg_id, f"unexpected reply {reply!r}"
        assert "error" not in reply, f"{method} failed: {reply['error']}"
        return reply["result"]

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return self.reply_to(self.send(method, params), method)

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
    an over-limit move is clamped, Nav2 takes the goal and the physics robot
    gets there no faster than the limit, measured on the controller's
    commands and, loosely, on /odom; a target in the
    no-go zone is refused and the robot does not move.
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

            cmd_topic = _controller_topic(adapter)
            with _odom_speeds(adapter) as speeds, \
                    _commanded_speeds(adapter, cmd_topic) as commanded:
                is_error, moved = client.call_tool(
                    "move_to", {"x": target[0], "y": target[1], "speed": 1.0}
                )
            assert not is_error, f"move_to failed over MCP: {moved}"
            assert moved["safety_interventions"], (
                f"1.0 m/s against a {MCP_MAX_SPEED} m/s limit was not reported as clamped: {moved}"
            )
            # The limit reaches Nav2's controller through its speed_limit
            # topic, so the reply reports a limited drive, and the robot
            # really drove no faster than the limit.
            assert moved.get("speed_limit_applied") is not False, (
                f"Nav2's controller did not take the speed limit: {moved}"
            )
            assert len(speeds) > 20, f"too few /odom samples during the move: {len(speeds)}"
            odom_peak = max(v for _, v in speeds)
            odom_avg_peak = _peak_moving_average(speeds, MCP_ODOM_WINDOW_S)
            cmd_peak = max(commanded, default=0.0)
            print(
                f"Nav2 speed under a {MCP_MAX_SPEED} m/s limit: {cmd_topic} peak "
                f"{cmd_peak:.3f} m/s over {len(commanded)} commands; /odom raw peak "
                f"{odom_peak:.3f} m/s, {MCP_ODOM_WINDOW_S} s average peak {odom_avg_peak:.3f} m/s"
            )
            assert odom_peak > 0.03, (
                f"/odom never showed the robot moving (peak {odom_peak:.3f} m/s)"
            )
            assert commanded, f"no velocity commands seen on {cmd_topic} during the move"
            assert cmd_peak <= MCP_MAX_SPEED + MCP_COMMAND_TOLERANCE, (
                f"Nav2's controller commanded {cmd_peak:.3f} m/s on {cmd_topic} under a "
                f"{MCP_MAX_SPEED} m/s limit"
            )
            assert odom_avg_peak <= MCP_MAX_SPEED + MCP_ODOM_TOLERANCE, (
                f"robot averaged {odom_avg_peak:.3f} m/s over {MCP_ODOM_WINDOW_S} s on /odom "
                f"through Nav2 under a {MCP_MAX_SPEED} m/s limit (raw peak {odom_peak:.3f})"
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
    assert f"Nav2 speed limit set to {MCP_MAX_SPEED:.2f} m/s" in server_log, (
        "apyrobo mcp did not set Nav2's speed limit. Server log tail:\n" + server_log[-2000:]
    )


# Spans the whole world between the robot and GOAL, so every route to GOAL
# crosses it while GOAL itself stays outside. The route check runs after the
# MCP leg above has brought the robot back to START.
ROUTE_NO_GO = (-1.7, -50.0, -1.5, 50.0)


@pytest.mark.integration
def test_mcp_refuses_a_goal_whose_nav2_route_crosses_a_zone(robot):
    """Nav2's planned route is checked: the target is clear, the way there is not."""
    adapter = robot._adapter
    start = adapter.position
    assert start[0] < ROUTE_NO_GO[0], f"robot is not behind the zone: {start}"
    cmd = [
        sys.executable, "-m", "apyrobo", "--verbose", "mcp",
        "--robot", "ros2://burger",
        "--no-go=" + ",".join(str(v) for v in ROUTE_NO_GO),
    ]
    with tempfile.TemporaryFile(mode="w+") as log:
        client = _StdioMCPClient(cmd, stderr=log)
        try:
            client.request("initialize", {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "apyrobo-nav2-e2e", "version": "1.0"},
            })
            is_error, refused = client.call_tool("move_to", {"x": GOAL[0], "y": GOAL[1]})
        finally:
            client.close()
        log.seek(0)
        server_log = log.read()
    assert is_error and "crosses collision zone" in refused["error"], (
        f"a route through the zone was not refused: {refused}\n" + server_log[-2000:]
    )
    assert "Nav2: sending goal" not in server_log, "the goal reached Nav2 despite the refusal"
    assert _dist(start, adapter.position) < 0.15, "robot moved after a refused command"


# How far the robot has to get before the stop is sent, and how far short of
# GOAL it must end up. GOAL is 0.8 m from START, several seconds at this speed.
# The second move starts where the cancelled one settled, and a cancel lets the
# robot run on for a while (Nav2 has to confirm it), so with 0.15 m the stop
# could land only ~0.27 m from GOAL and settle just inside the 0.25 m check.
# 0.08 m is still well above the 0.05 m a settled robot may drift, so the stop
# and the cancel both still land while the robot is driving.
STOP_AFTER_M = 0.08
STOP_SHORT_OF_GOAL_M = 0.25
STOP_MAX_SPEED = 0.22


def _wait_until_driving(adapter: Any, start: tuple[float, float]) -> None:
    """Block until the physics robot is on its way, so the stop lands mid-move."""
    deadline = time.time() + 30.0
    while _dist(start, adapter.position) < STOP_AFTER_M:
        assert time.time() < deadline, f"robot never started moving from {start}"
        time.sleep(0.05)


def _assert_stopped_short(adapter: Any, action: str) -> tuple[float, float]:
    """The robot settles, stays put and has not reached GOAL."""
    time.sleep(1.5)  # Nav2's last velocity command and a little coasting
    settled = adapter.position
    time.sleep(2.0)
    assert _dist(settled, adapter.position) < 0.05, (
        f"robot still moving 1.5 s after {action}: {settled} then {adapter.position}"
    )
    assert _dist(settled, GOAL) > STOP_SHORT_OF_GOAL_M, (
        f"robot reached the goal region despite {action}: {settled}, goal={GOAL}"
    )
    return settled


@pytest.mark.integration
def test_mcp_cancel_and_stop_end_a_running_nav2_goal(robot):
    """A client's cancel, then a stop call, each halt the robot partway to the goal.

    Both arrive while move_to is still running, and Nav2 has to confirm the
    goal as cancelled. The cancelled move gets no reply: the next line the
    server writes must be the answer to the call after it.
    """
    adapter = robot._adapter
    cmd = [
        sys.executable, "-m", "apyrobo", "--verbose", "mcp",
        "--robot", "ros2://burger",
        "--max-speed", str(STOP_MAX_SPEED),
    ]
    with tempfile.TemporaryFile(mode="w+") as log:
        client = _StdioMCPClient(cmd, stderr=log)
        # A move that is not stopped blocks the readline below; fail, don't hang.
        watchdog = threading.Timer(120.0, client.proc.kill)
        watchdog.start()
        try:
            client.request("initialize", {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "apyrobo-nav2-e2e", "version": "1.0"},
            })
            move = {"name": "move_to", "arguments": {"x": GOAL[0], "y": GOAL[1]}}

            start = adapter.position
            move_id = client.send("tools/call", move)
            _wait_until_driving(adapter, start)
            client.write({
                "method": "notifications/cancelled",
                "params": {"requestId": move_id, "reason": "client gave up"},
            })
            after_cancel = _assert_stopped_short(adapter, "the cancel")
            is_error, report = client.call_tool("safety_report", {})
            assert not is_error and report, "server did not answer after a cancelled move"

            move_id = client.send("tools/call", move)
            _wait_until_driving(adapter, after_cancel)
            stop_id = client.send("tools/call", {"name": "stop", "arguments": {}})
            stopped = client.reply_to(move_id, "move_to")
            assert stopped.get("isError"), f"a stopped move did not return an error: {stopped}"
            assert stopped["content"][0]["text"].startswith("STOPPED:"), stopped
            assert not client.reply_to(stop_id, "stop").get("isError")
            _assert_stopped_short(adapter, "stop")
        finally:
            watchdog.cancel()
            client.close()
        log.seek(0)
        server_log = log.read()
    assert server_log.count("Nav2: sending goal") >= 2, (
        "the moves did not go through Nav2 (fell back to cmd_vel?). "
        "Server log tail:\n" + server_log[-2000:]
    )
    assert server_log.count("Nav2: cancelling current goal") >= 2, server_log[-2000:]
    assert server_log.count("Nav2: goal CANCELLED") >= 2, (
        "Nav2 did not confirm the goals as cancelled:\n" + server_log[-2000:]
    )


@pytest.mark.integration
def test_position_comes_from_the_map_frame_under_nav2(robot):
    """With localisation up, the adapter reads the map-to-base transform, not /odom."""
    adapter = robot._adapter
    deadline = time.time() + 15.0
    while adapter.position_frame != "map" and time.time() < deadline:
        time.sleep(0.1)
    assert adapter.position_frame == "map", "no fresh map-to-base transform from TF"
    time.sleep(1.0)  # past Nav2's feedback: odometry ticks must keep it in map
    assert adapter.position_frame == "map"


@pytest.mark.integration
def test_stop_after_nav_is_safe(robot):
    """stop() stays safe and the adapter queryable after Nav2 activity."""
    robot.stop()
    robot.stop()  # idempotent
    pos = robot.get_position()
    assert isinstance(pos, tuple) and len(pos) == 2
    assert isinstance(robot._adapter.get_health(), dict)

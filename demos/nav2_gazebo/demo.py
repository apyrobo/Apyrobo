"""
Demo: Natural Language → Nav2 → Gazebo
=======================================
The flagship-stack demo, live: a natural-language task is planned by the
rule agent and executed through the real Nav2 ``NavigateToPose`` action,
driving a physics-simulated TurtleBot3 through ``turtlebot3_world``.

Nothing here is mocked — the same pipeline CI verifies on every commit
(tests/integration/test_gazebo_nav2_e2e.py). Requires the live sim, so it
runs inside the ``gazebo-nav`` Docker Compose profile (Linux):

    docker compose -f docker/docker-compose.yml --profile gazebo-nav \
        run --rm gazebo-nav-demo

See demos/nav2_gazebo/README.md.
"""
from __future__ import annotations

import contextlib
import math
import sys
import threading
import time

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import TaskStatus
from apyrobo.skills.agent import Agent

TASK = "navigate to (-1.2, -0.5)"
GOAL = (-1.2, -0.5)


def say(line: str = "") -> None:
    print(line, flush=True)


def main() -> int:
    say("╔══════════════════════════════════════════════════════════╗")
    say("║  APYROBO — natural language → Nav2 → Gazebo (no mocks)   ║")
    say("╚══════════════════════════════════════════════════════════╝")
    say()
    say("① Discovering the robot on the ROS 2 graph …")
    robot = Robot.discover(
        "ros2://burger",
        nav2_server_wait_sec=90.0,
        odom_wait_sec=60.0,
    )
    adapter = robot._adapter
    x, y = adapter.position
    say(f"   ros2://burger — odometry live, position ({x:.2f}, {y:.2f})")
    nav2 = bool(getattr(adapter, "_has_nav2", False))
    say(f"   Nav2 NavigateToPose action server: {'FOUND' if nav2 else 'not found (cmd_vel fallback)'}")
    say()

    agent = Agent(provider="rule")

    say(f"② Planning the task: {TASK!r}  (rule-based, no LLM, no API key)")
    graph = agent.plan(TASK, robot)
    for skill in graph.get_execution_order():
        params = graph.get_parameters(skill.skill_id)
        say(f"   plan → {skill.name}  {params}")
    say()

    say("③ Executing — Nav2 plans the path, physics does the rest:")
    start = adapter.position
    done = threading.Event()

    def ticker() -> None:
        while not done.wait(2.0):
            px, py = adapter.position
            d = math.hypot(px - GOAL[0], py - GOAL[1])
            say(f"   … robot at ({px:+.2f}, {py:+.2f}) — {d:.2f} m from goal")

    t = threading.Thread(target=ticker, daemon=True)
    t.start()
    t0 = time.time()
    result = agent.execute(task=TASK, robot=robot)
    done.set()
    t.join(timeout=1)

    ex, ey = adapter.position
    dist = math.hypot(ex - GOAL[0], ey - GOAL[1])
    travelled = math.hypot(ex - start[0], ey - start[1])
    say()
    if result.status == TaskStatus.COMPLETED and dist < 0.5:
        say(f"✅ GOAL REACHED — ({ex:+.2f}, {ey:+.2f}), {dist:.2f} m from goal, "
            f"{travelled:.2f} m travelled in {time.time() - t0:.1f} s")
        code = 0
    else:
        say(f"❌ {result.status.value} — ended {dist:.2f} m from goal "
            f"(error: {result.error or 'n/a'})")
        code = 1

    robot.stop()
    # Prompt DDS teardown (see ros2_bridge module docs) — without it the C++
    # layer prints "terminate called without an active exception" on exit.
    from apyrobo.core.ros2_bridge import _ROS2NodeManager
    with contextlib.suppress(Exception):
        _ROS2NodeManager.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Hello Robot — the simplest complete APYROBO program.

Demonstrates:
  - Discovering a mock robot (no ROS 2 or hardware needed)
  - Querying what the robot can do
  - Executing a natural-language task with the rule-based agent
  - Reading the task result

Run from the repo root:
    python examples/01_hello_robot.py
"""

from apyrobo import Robot, Agent


def main() -> None:
    # ── 1. Discover ──────────────────────────────────────────
    robot = Robot.discover("mock://turtlebot4")
    caps = robot.capabilities()

    print(f"Robot : {caps.name}")
    print(f"Skills: {[c.name for c in caps.capabilities]}")
    print(f"Speed : {caps.max_speed} m/s max")
    print()

    # ── 2. Execute a task ────────────────────────────────────
    # "rule" provider needs no API key — great for testing.
    # Swap to provider="llm", model="claude-sonnet-4-20250514" for a real LLM.
    agent = Agent(provider="rule")

    print("Task: 'navigate to position 3, 5 and pick up the object'")
    result = agent.execute(
        task="deliver package from position 1, 2 to position 3, 5",
        robot=robot,
        on_event=lambda e: print(f"  [{e.status.value:>10}] {e.skill_id}: {e.message}"),
    )

    # ── 3. Print result ──────────────────────────────────────
    print()
    print(f"Status : {result.status.value}")
    print(f"Steps  : {result.steps_completed}/{result.steps_total}")
    if result.error:
        print(f"Error  : {result.error}")

    pos = robot.get_position()
    print(f"Position after task: ({pos[0]:.1f}, {pos[1]:.1f})")


if __name__ == "__main__":
    main()

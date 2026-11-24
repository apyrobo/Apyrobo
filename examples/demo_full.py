#!/usr/bin/env python3
"""
APYROBO — Full Pipeline Demo (mock mode, no ROS 2 required).

This is the demo from the roadmap: discover a robot, plan a task in
natural language, execute it through a safety policy with live status
streaming, then watch the policy refuse a task that ends in a restricted area.

    python examples/demo_full.py
"""

from apyrobo import Agent, Robot, SafetyEnforcer, SafetyPolicy

# ── 1. Discover ──────────────────────────────────────────────
robot = Robot.discover("mock://turtlebot4")
print(f"Robot: {robot.capabilities().name}")
print(f"  Skills: {[c.name for c in robot.capabilities().capabilities]}")
print(f"  Sensors: {[s.sensor_id for s in robot.capabilities().sensors]}")
print()

# ── 2. Safety ────────────────────────────────────────────────
safe_robot = SafetyEnforcer(robot, policy=SafetyPolicy(
    name="warehouse",
    max_speed=1.0,
    collision_zones=[
        {"x_min": 8, "x_max": 10, "y_min": 8, "y_max": 10},  # restricted area
    ],
    human_proximity_limit=0.5,
))
print(f"Safety: {safe_robot.policy}")
print()

# ── 3. Plan + Execute ───────────────────────────────────────
agent = Agent(provider="rule")

def on_event(event):
    print(f"  → [{event.status.value:>10}] {event.skill_id}: {event.message}")

# Pass safe_robot, not robot: every move then goes through the policy.
def run(task):
    print(f"Task: {task!r}")
    print("-" * 50)
    result = agent.execute(task=task, robot=safe_robot, on_event=on_event)
    print("-" * 50)
    print(f"Result: {result.status.value}")
    print(f"  Steps: {result.steps_completed}/{result.steps_total}")
    if result.error:
        print(f"  Reason: {result.error}")
    print()


run("deliver package from (1, 2) to (5, 5)")
# (9, 9) is inside the restricted area: the policy refuses the move.
run("go to (9, 9)")

# ── 4. Show safety log ──────────────────────────────────────
if safe_robot.violations:
    print(f"Safety violations: {len(safe_robot.violations)}")
    for v in safe_robot.violations:
        print(f"  ⚠ {v.get('message', v)}")
else:
    print("Safety: No violations ✓")

if safe_robot.interventions:
    print(f"Safety interventions: {len(safe_robot.interventions)}")
    for i in safe_robot.interventions:
        print(f"  ⚡ {i}")
else:
    print("Safety: No interventions needed ✓")

print()
print("Done! This demo runs entirely in-memory with the mock adapter.")
print("To run against a real ROS 2 robot or a running Gazebo, use: "
      "Robot.discover('ros2://turtlebot4') (see docs/first-real-robot.md)")

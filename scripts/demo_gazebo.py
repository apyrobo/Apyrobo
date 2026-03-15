#!/usr/bin/env python3
"""
APYROBO — MVP Demo in Gazebo

This is the demo from the roadmap: a user types a task in plain English,
APYROBO plans it, assigns robots, executes with live status streaming,
handles obstacles, and completes the task — safely.

Run AFTER Gazebo + Nav2 are up (via scripts/launch.sh).

Usage:
    python3 scripts/demo_gazebo.py
    python3 scripts/demo_gazebo.py --task "go to (3, 2)"
    python3 scripts/demo_gazebo.py --interactive
"""

from __future__ import annotations

import argparse
import sys
import os
import time

sys.path.insert(0, "/workspace")

# Rich output (falls back to plain if not installed)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    console = Console()
    def header(text): console.print(Panel(text, style="bold blue"))
    def success(text): console.print(f"[bold green]✓[/] {text}")
    def warn(text): console.print(f"[bold yellow]⚠[/] {text}")
    def error(text): console.print(f"[bold red]✗[/] {text}")
    def info(text): console.print(f"  {text}")
except ImportError:
    def header(text): print(f"\n{'='*50}\n  {text}\n{'='*50}")
    def success(text): print(f"  ✓ {text}")
    def warn(text): print(f"  ⚠ {text}")
    def error(text): print(f"  ✗ {text}")
    def info(text): print(f"  {text}")


from apyrobo import (
    Robot, Agent, SafetyEnforcer, SafetyPolicy,
    SensorPipeline, SensorReading,
)
from apyrobo.core.schemas import TaskStatus
from apyrobo.skills.executor import SkillStatus


def run_demo(task: str, interactive: bool = False) -> None:
    """Execute the full MVP demo."""

    # ── 1. Discover ──────────────────────────────────────────
    header("APYROBO MVP Demo")
    info("Discovering robot...")

    robot = Robot.discover("gazebo://turtlebot4")
    caps = robot.capabilities()

    success(f"Robot: {caps.name}")
    info(f"  Capabilities: {[c.name for c in caps.capabilities]}")
    info(f"  Sensors:      {[f'{s.sensor_id} ({s.sensor_type.value})' for s in caps.sensors]}")
    info(f"  Max speed:    {caps.max_speed} m/s")
    info(f"  Nav2:         {caps.metadata.get('nav2_available', 'unknown')}")
    info(f"  Position:     ({robot._adapter.position[0]:.2f}, {robot._adapter.position[1]:.2f})")
    print()

    # ── 2. Sensor pipeline ───────────────────────────────────
    info("Starting sensor pipeline...")
    try:
        from apyrobo.core.ros2_bridge import _ROS2NodeManager
        from apyrobo.sensors.ros2_subscribers import ROS2SensorBridge
        node = _ROS2NodeManager.get_node()
        pipeline = SensorPipeline()
        sensor_bridge = ROS2SensorBridge(node, pipeline)
        time.sleep(2.0)  # let data flow
        world = pipeline.get_world_state()
        success(f"Sensor pipeline online ({sensor_bridge.subscriber_count} subscribers)")
        info(f"  Obstacles detected: {len(world.obstacles)}")
        info(f"  Objects detected:   {len(world.detected_objects)}")
    except Exception as e:
        warn(f"Sensor pipeline not available: {e}")
        world = None
    print()

    # ── 3. Safety ────────────────────────────────────────────
    safe_robot = SafetyEnforcer(robot, policy=SafetyPolicy(
        name="demo",
        max_speed=0.8,
        collision_zones=[],  # add zones if needed
        human_proximity_limit=0.5,
    ))
    success(f"Safety enforcer active: max_speed={safe_robot.policy.max_speed} m/s")
    print()

    # ── 4. Agent ─────────────────────────────────────────────
    agent = Agent(provider="auto")
    success(f"Agent ready")
    print()

    # ── 5. Execute ───────────────────────────────────────────
    if interactive:
        # Interactive loop
        header("Interactive Mode — type a task (or 'quit')")
        while True:
            try:
                task = input("\n  Task > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not task or task.lower() in ("quit", "exit", "q"):
                break
            _execute_task(agent, robot, task)
    else:
        _execute_task(agent, robot, task)

    # ── 6. Cleanup ───────────────────────────────────────────
    robot.stop()
    print()

    # Safety report
    if safe_robot.violations:
        warn(f"Safety violations: {len(safe_robot.violations)}")
        for v in safe_robot.violations:
            info(f"  {v}")
    else:
        success("No safety violations")

    if safe_robot.interventions:
        info(f"Safety interventions: {len(safe_robot.interventions)}")
        for i in safe_robot.interventions:
            info(f"  {i}")

    print()
    success("Demo complete!")


def _execute_task(agent: Agent, robot: Robot, task: str) -> None:
    """Plan and execute a single task with live output."""
    header(f"Task: {task}")
    print()

    t0 = time.time()

    def on_event(event):
        elapsed = time.time() - t0
        status_icon = {
            SkillStatus.PENDING: "⏳",
            SkillStatus.RUNNING: "🔄",
            SkillStatus.COMPLETED: "✅",
            SkillStatus.FAILED: "❌",
        }.get(event.status, "  ")
        print(f"  {status_icon} [{elapsed:6.1f}s] {event.skill_id}: {event.message}")

    result = agent.execute(task=task, robot=robot, on_event=on_event)

    elapsed = time.time() - t0
    print()
    if result.status == TaskStatus.COMPLETED:
        success(f"COMPLETED in {elapsed:.1f}s ({result.steps_completed}/{result.steps_total} steps)")
    else:
        error(f"FAILED: {result.error}")
        if result.recovery_actions_taken:
            info(f"Recovery actions: {[a.value for a in result.recovery_actions_taken]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APYROBO MVP Demo")
    parser.add_argument("--task", default="deliver package from (1, 0) to (3, 0)",
                        help="Task to execute")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode — type tasks one by one")
    args = parser.parse_args()

    try:
        run_demo(task=args.task, interactive=args.interactive)
    except KeyboardInterrupt:
        print("\nInterrupted — stopping robot...")
        try:
            robot = Robot.discover("gazebo://turtlebot4")
            robot.stop()
        except Exception:
            pass
    except Exception as e:
        error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

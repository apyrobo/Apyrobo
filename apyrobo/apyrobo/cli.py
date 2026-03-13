#!/usr/bin/env python3
"""
APYROBO CLI — command-line interface for robot discovery, planning, and execution.

Usage:
    python -m apyrobo.cli discover mock://turtlebot4
    python -m apyrobo.cli plan "deliver package to room 3" --robot mock://tb4
    python -m apyrobo.cli execute "go to (3, 2)" --robot mock://tb4
    python -m apyrobo.cli skills --list
    python -m apyrobo.cli config --generate > apyrobo.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import TaskStatus
from apyrobo.skills.agent import Agent
from apyrobo.skills.skill import BUILTIN_SKILLS
from apyrobo.skills.executor import SkillStatus
from apyrobo.safety.enforcer import SafetyEnforcer, SafetyPolicy
from apyrobo.safety.confidence import ConfidenceEstimator
from apyrobo.config import ApyroboConfig


def cmd_discover(args: argparse.Namespace) -> None:
    """Discover a robot and show its capabilities."""
    robot = Robot.discover(args.uri)
    caps = robot.capabilities()

    print(f"Robot: {caps.name}")
    print(f"  ID:         {caps.robot_id}")
    print(f"  Max speed:  {caps.max_speed} m/s")
    print(f"  Metadata:   {caps.metadata}")
    print()
    print("Capabilities:")
    for c in caps.capabilities:
        print(f"  - {c.name} ({c.capability_type.value}): {c.description}")
    print()
    print("Sensors:")
    for s in caps.sensors:
        topic = f" → {s.topic}" if s.topic else ""
        hz = f" @ {s.hz}Hz" if s.hz else ""
        print(f"  - {s.sensor_id} ({s.sensor_type.value}){topic}{hz}")
    print()
    print("Joints:")
    if caps.joints:
        for j in caps.joints:
            print(f"  - {j.name} ({j.joint_id})")
    else:
        print("  (none)")


def cmd_plan(args: argparse.Namespace) -> None:
    """Plan a task and show the skill graph (without executing)."""
    robot = Robot.discover(args.robot)
    agent = Agent(provider=args.provider)
    graph = agent.plan(args.task, robot)

    print(f"Task: {args.task!r}")
    print(f"Plan: {len(graph)} skills")
    print()

    order = graph.get_execution_order()
    for i, skill in enumerate(order, 1):
        params = graph.get_parameters(skill.skill_id)
        print(f"  {i}. {skill.name} ({skill.skill_id})")
        if params:
            for k, v in params.items():
                print(f"       {k}: {v}")

    # Confidence check
    estimator = ConfidenceEstimator()
    report = estimator.assess(graph, robot)
    print()
    print(f"Confidence: {report.confidence:.0%} (risk: {report.risk_level})")
    if report.risks:
        for r in report.risks:
            print(f"  ⚠ {r.name}: {r.description}")
    print(f"Proceed: {'yes' if report.can_proceed else 'NO — too risky'}")


def cmd_execute(args: argparse.Namespace) -> None:
    """Plan and execute a task with live output."""
    robot = Robot.discover(args.robot)

    # Safety
    policy = SafetyPolicy(max_speed=args.max_speed)
    enforcer = SafetyEnforcer(robot, policy=policy)

    # Agent
    agent = Agent(provider=args.provider)

    # Confidence check
    graph = agent.plan(args.task, robot)
    estimator = ConfidenceEstimator()
    report = estimator.assess(graph, robot)

    print(f"Task:       {args.task!r}")
    print(f"Robot:      {robot.robot_id}")
    print(f"Confidence: {report.confidence:.0%} (risk: {report.risk_level})")
    if report.risks:
        for r in report.risks:
            print(f"  ⚠ {r.name}: {r.description}")

    if not report.can_proceed and not args.force:
        print("\nAborted — confidence too low. Use --force to override.")
        sys.exit(1)

    print()
    print("Executing...")
    print("-" * 50)

    t0 = time.time()

    def on_event(event):
        elapsed = time.time() - t0
        icon = {
            SkillStatus.PENDING: "⏳",
            SkillStatus.RUNNING: "🔄",
            SkillStatus.COMPLETED: "✅",
            SkillStatus.FAILED: "❌",
        }.get(event.status, "  ")
        print(f"  {icon} [{elapsed:5.1f}s] {event.skill_id}: {event.message}")

    result = agent.execute(task=args.task, robot=robot, on_event=on_event)
    duration = time.time() - t0

    print("-" * 50)
    print(f"Result: {result.status.value}")
    print(f"  Steps:    {result.steps_completed}/{result.steps_total}")
    print(f"  Duration: {duration:.1f}s")
    if result.error:
        print(f"  Error:    {result.error}")

    robot.stop()

    if result.status != TaskStatus.COMPLETED:
        sys.exit(1)


def cmd_skills(args: argparse.Namespace) -> None:
    """List available skills."""
    if args.list:
        print("Built-in Skills:")
        for skill in BUILTIN_SKILLS.values():
            print(f"  {skill.skill_id}")
            print(f"    Name: {skill.name}")
            print(f"    Capability: {skill.required_capability.value}")
            print(f"    Description: {skill.description}")
            if skill.preconditions:
                print(f"    Preconditions: {[c.name for c in skill.preconditions]}")
            if skill.postconditions:
                print(f"    Postconditions: {[c.name for c in skill.postconditions]}")
            print()

    if args.export:
        skill = BUILTIN_SKILLS.get(args.export)
        if skill is None:
            print(f"Unknown skill: {args.export}")
            sys.exit(1)
        print(skill.to_json())


def cmd_config(args: argparse.Namespace) -> None:
    """Generate or show configuration."""
    if args.generate:
        config = ApyroboConfig()
        print(config.to_yaml())
    elif args.file:
        config = ApyroboConfig.from_file(args.file)
        print(config)
        print()
        print(config.to_yaml())
    else:
        config = ApyroboConfig.from_env()
        print(config)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="apyrobo",
        description="APYROBO — AI orchestration layer for robotics",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    sub = parser.add_subparsers(dest="command")

    # discover
    p_disc = sub.add_parser("discover", help="Discover a robot")
    p_disc.add_argument("uri", help="Robot URI (e.g. mock://turtlebot4)")

    # plan
    p_plan = sub.add_parser("plan", help="Plan a task (no execution)")
    p_plan.add_argument("task", help="Task description in natural language")
    p_plan.add_argument("--robot", default="mock://turtlebot4")
    p_plan.add_argument("--provider", default="rule")

    # execute
    p_exec = sub.add_parser("execute", help="Plan and execute a task")
    p_exec.add_argument("task", help="Task description")
    p_exec.add_argument("--robot", default="mock://turtlebot4")
    p_exec.add_argument("--provider", default="rule")
    p_exec.add_argument("--max-speed", type=float, default=1.5)
    p_exec.add_argument("--force", action="store_true", help="Execute even if low confidence")

    # skills
    p_skills = sub.add_parser("skills", help="Manage skills")
    p_skills.add_argument("--list", action="store_true", help="List all skills")
    p_skills.add_argument("--export", type=str, help="Export a skill as JSON")

    # config
    p_config = sub.add_parser("config", help="Configuration management")
    p_config.add_argument("--generate", action="store_true", help="Generate default config")
    p_config.add_argument("--file", type=str, help="Load and display a config file")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "discover": cmd_discover,
        "plan": cmd_plan,
        "execute": cmd_execute,
        "skills": cmd_skills,
        "config": cmd_config,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()

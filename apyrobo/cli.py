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
from apyrobo.skills.skill import Skill, BUILTIN_SKILLS
from apyrobo.skills.executor import SkillStatus
from apyrobo.skills.package import SkillPackage
from apyrobo.skills.registry import SkillRegistry, PackageConflict
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
# Package management commands
# ---------------------------------------------------------------------------

def _get_registry(args: argparse.Namespace) -> SkillRegistry:
    """Get or create a SkillRegistry, respecting --registry-dir."""
    registry_dir = getattr(args, "registry_dir", None)
    return SkillRegistry(registry_dir)


def cmd_pkg_init(args: argparse.Namespace) -> None:
    """Initialise a new skill package in a directory."""
    pkg = SkillPackage.init(
        name=args.name,
        version=args.version or "0.1.0",
        description=args.description or "",
        author=args.author or "",
        directory=args.directory or f"./{args.name}",
    )
    out_dir = args.directory or f"./{args.name}"
    print(f"Initialised package: {pkg.name}@{pkg.version}")
    print(f"  Directory: {out_dir}")
    print(f"  Manifest:  {out_dir}/skill-package.json")
    print(f"  Skills:    {out_dir}/skills/")
    print()
    print("Next steps:")
    print(f"  1. Add skill JSON files to {out_dir}/skills/")
    print(f"  2. Edit {out_dir}/skill-package.json to list them")
    print(f"  3. Run: apyrobo pkg pack {out_dir}")


def cmd_pkg_pack(args: argparse.Namespace) -> None:
    """Pack a package directory into a .skillpkg archive."""
    pkg = SkillPackage.load(args.directory)
    errors = pkg.validate()
    if errors:
        print(f"Validation errors in {args.directory}:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    output = args.output  # may be None
    archive_path = pkg.pack(args.directory, output)
    print(f"Packed: {pkg.name}@{pkg.version}")
    print(f"  Skills: {', '.join(pkg.skill_ids)}")
    print(f"  Archive: {archive_path}")


def cmd_pkg_install(args: argparse.Namespace) -> None:
    """Install a skill package from an archive or directory."""
    registry = _get_registry(args)
    try:
        pkg = registry.install(args.source, force=args.force)
        print(f"Installed: {pkg.name}@{pkg.version}")
        print(f"  Skills: {', '.join(pkg.skill_ids)}")

        unmet = registry.check_dependencies(pkg)
        if unmet:
            print()
            print("Unmet dependencies:")
            for dep in unmet:
                print(f"  - {dep}")
    except PackageConflict as e:
        print(f"Conflict: {e}")
        print("Use --force to overwrite.")
        sys.exit(1)
    except Exception as e:
        print(f"Install failed: {e}")
        sys.exit(1)


def cmd_pkg_remove(args: argparse.Namespace) -> None:
    """Remove an installed skill package."""
    registry = _get_registry(args)
    if registry.remove(args.name):
        print(f"Removed: {args.name}")
    else:
        print(f"Package not installed: {args.name}")
        sys.exit(1)


def cmd_pkg_list(args: argparse.Namespace) -> None:
    """List installed skill packages."""
    registry = _get_registry(args)
    packages = registry.list_packages()

    if not packages:
        print("No packages installed.")
        print("Install with: apyrobo pkg install <path>")
        return

    for pkg_info in packages:
        name = pkg_info["name"]
        version = pkg_info.get("version", "?")
        desc = pkg_info.get("description", "")
        skills = pkg_info.get("skills", [])
        print(f"  {name}@{version}  ({len(skills)} skills)")
        if desc:
            print(f"    {desc}")
        if args.verbose_list:
            print(f"    Skills: {', '.join(skills)}")
            tags = pkg_info.get("tags", [])
            if tags:
                print(f"    Tags: {', '.join(tags)}")
            deps = pkg_info.get("dependencies", {})
            if deps:
                dep_strs = [f"{k} {v}" for k, v in deps.items()]
                print(f"    Dependencies: {', '.join(dep_strs)}")


def cmd_pkg_info(args: argparse.Namespace) -> None:
    """Show detailed info about an installed package."""
    registry = _get_registry(args)
    pkg = registry.get(args.name)
    if pkg is None:
        print(f"Package not installed: {args.name}")
        sys.exit(1)

    print(f"Package: {pkg.name}")
    print(f"  Version:      {pkg.version}")
    print(f"  Description:  {pkg.description or '(none)'}")
    print(f"  Author:       {pkg.author or '(none)'}")
    print(f"  License:      {pkg.license}")
    print(f"  Homepage:     {pkg.homepage or '(none)'}")
    print(f"  Capabilities: {', '.join(pkg.required_capabilities) or '(none)'}")
    print(f"  Min APYROBO:  {pkg.min_apyrobo_version}")
    print(f"  Tags:         {', '.join(pkg.tags) or '(none)'}")
    print()
    print(f"Skills ({len(pkg.skills)}):")
    for skill in pkg.skills:
        print(f"  - {skill.skill_id}: {skill.name}")
        if skill.description:
            print(f"    {skill.description}")
    if pkg.dependencies:
        print()
        print("Dependencies:")
        for dep_name, constraint in pkg.dependencies.items():
            installed = registry.is_installed(dep_name)
            status = "installed" if installed else "MISSING"
            print(f"  - {dep_name} {constraint} ({status})")


def cmd_pkg_search(args: argparse.Namespace) -> None:
    """Search installed packages."""
    registry = _get_registry(args)
    results = registry.search(args.query)
    if not results:
        print(f"No packages match: {args.query!r}")
        return

    print(f"Results for {args.query!r}:")
    for r in results:
        name = r["name"]
        version = r.get("version", "?")
        desc = r.get("description", "")
        print(f"  {name}@{version}")
        if desc:
            print(f"    {desc}")


def cmd_pkg_validate(args: argparse.Namespace) -> None:
    """Validate a skill package directory."""
    try:
        pkg = SkillPackage.load(args.directory)
    except Exception as e:
        print(f"Failed to load package from {args.directory}: {e}")
        sys.exit(1)

    errors = pkg.validate()
    if errors:
        print(f"Package {pkg.name}@{pkg.version} has errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"Package {pkg.name}@{pkg.version} is valid.")
    print(f"  Skills: {', '.join(pkg.skill_ids)}")
    print(f"  Dependencies: {len(pkg.dependencies)}")
    print(f"  Tags: {', '.join(pkg.tags) or '(none)'}")


# Reference to the pkg argparser, set during main() so cmd_pkg can print help.
_p_pkg: argparse.ArgumentParser | None = None

_PKG_COMMANDS = {
    "init": cmd_pkg_init,
    "pack": cmd_pkg_pack,
    "install": cmd_pkg_install,
    "remove": cmd_pkg_remove,
    "list": cmd_pkg_list,
    "info": cmd_pkg_info,
    "search": cmd_pkg_search,
    "validate": cmd_pkg_validate,
}


def cmd_pkg(args: argparse.Namespace) -> None:
    """Dispatch to the appropriate pkg sub-command."""
    if args.pkg_command is None:
        if _p_pkg is not None:
            _p_pkg.print_help()
        else:
            print("Usage: apyrobo pkg <subcommand>")
        return
    _PKG_COMMANDS[args.pkg_command](args)


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

    # pkg — skill package management
    p_pkg = sub.add_parser("pkg", help="Skill package management")
    p_pkg.add_argument("--registry-dir", type=str, default=None,
                       help="Override registry directory")
    pkg_sub = p_pkg.add_subparsers(dest="pkg_command")

    # pkg init
    p_pkg_init = pkg_sub.add_parser("init", help="Create a new skill package")
    p_pkg_init.add_argument("name", help="Package name (lowercase, hyphenated)")
    p_pkg_init.add_argument("--version", default="0.1.0")
    p_pkg_init.add_argument("--description", default="")
    p_pkg_init.add_argument("--author", default="")
    p_pkg_init.add_argument("--directory", default=None,
                            help="Output directory (default: ./<name>)")

    # pkg pack
    p_pkg_pack = pkg_sub.add_parser("pack", help="Pack a package into .skillpkg")
    p_pkg_pack.add_argument("directory", help="Package directory")
    p_pkg_pack.add_argument("--output", default=None, help="Output .skillpkg path")

    # pkg install
    p_pkg_install = pkg_sub.add_parser("install", help="Install a package")
    p_pkg_install.add_argument("source", help="Path to .skillpkg or package directory")
    p_pkg_install.add_argument("--force", action="store_true",
                               help="Overwrite existing package")

    # pkg remove
    p_pkg_remove = pkg_sub.add_parser("remove", help="Remove an installed package")
    p_pkg_remove.add_argument("name", help="Package name")

    # pkg list
    p_pkg_list = pkg_sub.add_parser("list", help="List installed packages")
    p_pkg_list.add_argument("-v", "--verbose-list", action="store_true",
                            help="Show skills and tags")

    # pkg info
    p_pkg_info = pkg_sub.add_parser("info", help="Show package details")
    p_pkg_info.add_argument("name", help="Package name")

    # pkg search
    p_pkg_search = pkg_sub.add_parser("search", help="Search packages")
    p_pkg_search.add_argument("query", help="Search query")

    # pkg validate
    p_pkg_validate = pkg_sub.add_parser("validate", help="Validate a package directory")
    p_pkg_validate.add_argument("directory", help="Package directory")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.command is None:
        parser.print_help()
        return

    # Store pkg parser reference for cmd_pkg help
    global _p_pkg
    _p_pkg = p_pkg

    commands = {
        "discover": cmd_discover,
        "plan": cmd_plan,
        "execute": cmd_execute,
        "skills": cmd_skills,
        "config": cmd_config,
        "pkg": cmd_pkg,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()

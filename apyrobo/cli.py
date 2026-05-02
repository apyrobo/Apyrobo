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
import subprocess
import sys
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

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


# ---------------------------------------------------------------------------
# Provider name resolution
# ---------------------------------------------------------------------------

# Friendly short-names and their (provider, model) translations.
_PROVIDER_ALIASES: dict[str, tuple[str, str]] = {
    "anthropic": ("llm", "claude-sonnet-4-20250514"),
    "claude": ("llm", "claude-sonnet-4-20250514"),
    "openai": ("llm", "gpt-4o"),
    "gpt": ("llm", "gpt-4o"),
    "gpt4": ("llm", "gpt-4o"),
    "gpt-4": ("llm", "gpt-4o"),
    "ollama": ("llm", "ollama/llama3"),
}

_PROVIDER_TABLE = """\
  Provider name       Equivalent to
  ─────────────────────────────────────────────────────────────
  rule                Built-in rule-based planner (no API key)
  llm                 LiteLLM provider — set --model explicitly
  tool_calling        LLM with structured tool-calling
  multi_turn          LLM with clarifying question support
  anthropic           llm --model claude-sonnet-4-20250514
  claude              llm --model claude-sonnet-4-20250514
  openai              llm --model gpt-4o
  gpt                 llm --model gpt-4o
  ollama              llm --model ollama/llama3

  For LLM providers, set the matching env var:
    ANTHROPIC_API_KEY   for Anthropic models
    OPENAI_API_KEY      for OpenAI models
"""


def _resolve_provider(provider: str, model: str | None = None) -> tuple[str, str | None]:
    """Expand friendly aliases to (provider, model). Returns (provider, model)."""
    alias = _PROVIDER_ALIASES.get(provider.lower())
    if alias:
        resolved_provider, default_model = alias
        return resolved_provider, model or default_model
    return provider, model


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
    provider, model = _resolve_provider(args.provider, getattr(args, "model", None))
    try:
        agent = Agent(provider=provider, **({"model": model} if model else {}))
    except ValueError as exc:
        print(f"Error: {exc}\n\nAvailable providers:\n{_PROVIDER_TABLE}", file=sys.stderr)
        sys.exit(1)
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
    provider, model = _resolve_provider(args.provider, getattr(args, "model", None))
    try:
        agent = Agent(provider=provider, **({"model": model} if model else {}))
    except ValueError as exc:
        print(f"Error: {exc}\n\nAvailable providers:\n{_PROVIDER_TABLE}", file=sys.stderr)
        sys.exit(1)

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


# ---------------------------------------------------------------------------
# apyrobo connect — one-command connection test
# ---------------------------------------------------------------------------

def _connect_with_timeout(uri: str, timeout: float) -> tuple[Any, float, str | None]:
    """Run Robot.discover + robot.connect in a background thread with a wall-clock timeout.

    Returns (robot, elapsed_s, error_message).
    On success: (robot, elapsed_s, None).
    On failure: (None,  elapsed_s, "<reason>").
    """
    robot_box: list[Any] = [None]
    elapsed_box: list[float] = [timeout]
    error_box: list[str | None] = [None]

    def _attempt() -> None:
        t0 = time.monotonic()
        try:
            r = Robot.discover(uri)
            r.connect()
            robot_box[0] = r
        except Exception as exc:
            error_box[0] = str(exc)
        finally:
            elapsed_box[0] = time.monotonic() - t0

    thread = threading.Thread(target=_attempt, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return None, timeout, f"Connection timed out after {timeout:.0f}s"
    return robot_box[0], elapsed_box[0], error_box[0]


def cmd_connect(args: argparse.Namespace) -> None:
    """Connect to a robot and optionally run a verification suite."""
    uri: str = args.uri
    timeout: float = getattr(args, "timeout", 10.0)
    as_json: bool = getattr(args, "json", False)
    verify: bool = getattr(args, "verify", False)

    if not as_json:
        print(f"Connecting to {uri}...")

    robot, connect_time, error = _connect_with_timeout(uri, timeout)

    if error:
        if as_json:
            print(json.dumps({
                "uri": uri,
                "connected": False,
                "error": error,
                "connect_time_s": round(connect_time, 3),
            }))
        else:
            print(f"{_icon('fail')} {error}")
        sys.exit(1)

    if not as_json:
        print(f"{_icon('pass')} Connected in {connect_time:.1f}s")

    if not verify:
        if as_json:
            print(json.dumps({
                "uri": uri,
                "connected": True,
                "connect_time_s": round(connect_time, 3),
            }))
        return

    # --verify: run the full check suite
    checks: list[dict[str, Any]] = []

    if not as_json:
        print()
        print(f"apyrobo connect --verify {uri}")
        print(_RULE)

    # 1. Position
    try:
        pos = robot.get_position()
        checks.append({"name": "position", "status": "pass", "value": list(pos)})
        if not as_json:
            print(f"{_icon('pass')} {'Position':<14} ({pos[0]:.2f}, {pos[1]:.2f})")
    except Exception as exc:
        checks.append({"name": "position", "status": "fail", "value": None, "error": str(exc)})
        if not as_json:
            print(f"{_icon('fail')} {'Position':<14} failed: {exc}")

    # 2. Battery
    try:
        health_data = robot.get_health()
        battery = health_data.get("battery_pct")
        if battery is not None:
            status = "warn" if battery < 20 else "pass"
            checks.append({"name": "battery", "status": status, "value": round(float(battery), 1)})
            if not as_json:
                print(f"{_icon(status)} {'Battery':<14} {battery:.0f}%")
        else:
            checks.append({"name": "battery", "status": "warn", "value": None})
            if not as_json:
                print(f"{_icon('warn')} {'Battery':<14} not available")
    except Exception:
        checks.append({"name": "battery", "status": "warn", "value": None})
        if not as_json:
            print(f"{_icon('warn')} {'Battery':<14} not available")

    # 3. Capabilities / Skills
    try:
        caps = robot.capabilities()
        cap_list = caps.capabilities
        names = [c.name for c in cap_list]
        count = len(names)
        display = ", ".join(names[:3])
        if count > 3:
            display += f", +{count - 3} more"
        checks.append({"name": "capabilities", "status": "pass", "value": names})
        if not as_json:
            print(f"{_icon('pass')} {'Capabilities':<14} {display}  ({count} skills)")
    except Exception as exc:
        checks.append({"name": "capabilities", "status": "fail", "value": None, "error": str(exc)})
        if not as_json:
            print(f"{_icon('fail')} {'Capabilities':<14} failed: {exc}")

    # 4. Round-trip latency — p50 of 3 calls
    try:
        raw: list[float] = []
        for _ in range(3):
            t = time.monotonic()
            robot.get_position()
            raw.append(time.monotonic() - t)
        raw.sort()
        p50_ms = raw[len(raw) // 2] * 1000
        checks.append({"name": "latency_ms_p50", "status": "pass", "value": round(p50_ms, 1)})
        if not as_json:
            print(f"{_icon('pass')} {'Latency':<14} {p50_ms:.0f}ms p50")
    except Exception as exc:
        checks.append({"name": "latency_ms_p50", "status": "fail", "value": None, "error": str(exc)})
        if not as_json:
            print(f"{_icon('fail')} {'Latency':<14} failed: {exc}")

    # 5. Health monitor — wait 2 s then sample is_healthy (ros2:// only)
    health_monitor = robot.health
    if health_monitor is not None:
        time.sleep(2)
        is_healthy = health_monitor.is_healthy
        status = "pass" if is_healthy else "warn"
        label = "online" if is_healthy else "no odom received"
        checks.append({"name": "health_monitor", "status": status, "value": is_healthy})
        if not as_json:
            print(f"{_icon(status)} {'Health':<14} {label}")
    else:
        checks.append({"name": "health_monitor", "status": "pass", "value": None, "note": "not applicable"})
        if not as_json:
            print(f"{_icon('pass')} {'Health':<14} not monitored")

    if not as_json:
        print(_RULE)

    passed = sum(1 for c in checks if c["status"] == "pass")
    warnings = sum(1 for c in checks if c["status"] == "warn")
    failures = sum(1 for c in checks if c["status"] == "fail")

    if as_json:
        print(json.dumps({
            "uri": uri,
            "connected": True,
            "connect_time_s": round(connect_time, 3),
            "checks": checks,
            "summary": {"passed": passed, "warnings": warnings, "failures": failures},
        }, indent=2))
    else:
        print(f"{passed} checks passed · {warnings} warnings · {failures} failures")

    if failures > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# apyrobo doctor — environment diagnostics
# ---------------------------------------------------------------------------

_RULE = "─" * 38
_LLM_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")


@dataclass
class _CheckResult:
    status: str        # "pass" | "warn" | "fail"
    message: str       # single-line detail shown next to the icon
    hint: str | None = None  # fix hint, indented below on warn/fail


def _icon(status: str) -> str:
    return {"pass": "✅ ", "warn": "⚠️ ", "fail": "❌ "}.get(status, "   ")


def _check_python_version() -> _CheckResult:
    vi = sys.version_info
    vs = f"{vi[0]}.{vi[1]}.{vi[2]}"
    if (vi[0], vi[1]) >= (3, 10):
        return _CheckResult("pass", f"Python {vs}")
    return _CheckResult(
        "fail", f"Python {vs} (3.10+ required)",
        hint="Upgrade Python: https://www.python.org/downloads/",
    )


def _check_apyrobo_install() -> _CheckResult:
    try:
        import apyrobo  # noqa: PLC0415
        return _CheckResult("pass", f"apyrobo {apyrobo.__version__}")
    except Exception as exc:
        return _CheckResult(
            "fail", f"apyrobo not importable: {exc}",
            hint="pip install apyrobo",
        )


def _check_rclpy() -> tuple[_CheckResult, bool]:
    """Returns (result, rclpy_available)."""
    try:
        import rclpy  # noqa: F401, PLC0415
        return _CheckResult("pass", "rclpy available"), True
    except ImportError:
        return _CheckResult(
            "warn", "rclpy not found",
            hint=(
                "Run inside Docker to use ros2://: "
                "docker compose -f docker/docker-compose.yml exec apyrobo bash. "
                "Once inside, test: apyrobo connect --verify ros2://<robot>"
            ),
        ), False


def _check_ros_domain_id() -> _CheckResult:
    domain_id = os.environ.get("ROS_DOMAIN_ID")
    if domain_id:
        return _CheckResult("pass", f"ROS_DOMAIN_ID={domain_id}")
    return _CheckResult(
        "warn", "ROS_DOMAIN_ID not set (defaults to 0, may clash)",
        hint="export ROS_DOMAIN_ID=42  (any unique integer per ROS network)",
    )


def _check_mock_adapter() -> _CheckResult:
    try:
        t0 = time.monotonic()
        Robot.discover("mock://test")
        elapsed = time.monotonic() - t0
        if elapsed < 1.0:
            return _CheckResult("pass", "Mock adapter ok")
        return _CheckResult("warn", f"Mock adapter slow ({elapsed:.2f}s)")
    except Exception as exc:
        return _CheckResult(
            "fail", f"Mock adapter failed: {exc}",
            hint="Reinstall apyrobo: pip install --force-reinstall apyrobo",
        )


def _check_llm_api_key() -> _CheckResult:
    found = [k for k in _LLM_KEYS if os.environ.get(k)]
    if found:
        return _CheckResult("pass", f"LLM API key present ({found[0]})")
    return _CheckResult(
        "warn",
        f"No LLM API key found ({', '.join(_LLM_KEYS)})",
        hint="Set one to use the LLM agent",
    )


def _check_docker() -> _CheckResult:
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=3,
        )
        if proc.returncode == 0:
            return _CheckResult("pass", "Docker available")
        return _CheckResult(
            "warn", "Docker not responding",
            hint="Docker not found — ros2:// integration tests won't run.",
        )
    except FileNotFoundError:
        return _CheckResult(
            "warn", "Docker not found",
            hint="Docker not found — ros2:// integration tests won't run.",
        )
    except subprocess.TimeoutExpired:
        return _CheckResult(
            "warn", "Docker timed out",
            hint="Docker not found — ros2:// integration tests won't run.",
        )


def _check_skill_registry() -> _CheckResult:
    try:
        urllib.request.urlopen("http://localhost:8080/health", timeout=2)
        return _CheckResult("pass", "Skill registry reachable at localhost:8080")
    except Exception:
        return _CheckResult(
            "warn", "Skill registry not reachable at localhost:8080",
            hint="Start with: apyrobo registry start",
        )


def run_doctor_checks() -> list[_CheckResult]:
    """Run all environment checks. Exposed for testing."""
    results: list[_CheckResult] = []
    results.append(_check_python_version())
    results.append(_check_apyrobo_install())
    rclpy_result, rclpy_ok = _check_rclpy()
    results.append(rclpy_result)
    if rclpy_ok:
        results.append(_check_ros_domain_id())
    results.append(_check_mock_adapter())
    results.append(_check_llm_api_key())
    results.append(_check_docker())
    results.append(_check_skill_registry())
    return results


def cmd_doctor(args: argparse.Namespace) -> None:
    """Run environment diagnostics (also aliased as `apyrobo diagnose`)."""
    print("apyrobo doctor")
    print(_RULE)

    results = run_doctor_checks()

    for result in results:
        print(f"{_icon(result.status)} {result.message}")
        if result.hint and result.status != "pass":
            print(f"    → {result.hint}")

    passed = sum(1 for r in results if r.status == "pass")
    warnings = sum(1 for r in results if r.status == "warn")
    failures = sum(1 for r in results if r.status == "fail")

    print(_RULE)
    print(f"{passed} passed · {warnings} warnings · {failures} failures")

    if failures > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# apyrobo diagnose --robot — full diagnostic report
# ---------------------------------------------------------------------------

class _LogCapture(logging.Handler):
    """Buffer the last N warning/error log records."""

    def __init__(self, maxlen: int = 20) -> None:
        super().__init__()
        self._records: list[dict[str, str]] = []
        self._maxlen = maxlen
        self.setLevel(logging.WARNING)

    def emit(self, record: logging.LogRecord) -> None:
        import datetime
        entry = {
            "level": record.levelname,
            "message": self.format(record),
            "logger": record.name,
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.timezone.utc
            ).isoformat(),
        }
        self._records.append(entry)
        if len(self._records) > self._maxlen:
            self._records.pop(0)

    def entries(self) -> list[dict[str, str]]:
        return list(self._records)


def _collect_system_info() -> dict[str, Any]:
    import platform
    vi = sys.version_info
    return {
        "python": f"{vi.major}.{vi.minor}.{vi.micro}",
        "os": platform.platform(),
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
    }


def _collect_health_info(robot: Any) -> dict[str, Any]:
    """Read ConnectionHealth state if available."""
    health_mon = getattr(robot, "health", None)
    if health_mon is None:
        return {"available": False}
    try:
        result: dict[str, Any] = {"available": True, "is_healthy": health_mon.is_healthy}
        # last_odom_age_s: compute from internal timestamp if exposed
        last_odom = getattr(health_mon, "_last_odom_time", None)
        if last_odom is not None:
            result["last_odom_age_s"] = round(time.monotonic() - last_odom, 3)
        reconnect = getattr(health_mon, "_reconnect_count", None)
        if reconnect is not None:
            result["reconnect_count"] = reconnect
        return result
    except Exception as exc:
        return {"available": True, "error": str(exc)}


def _collect_recent_tasks(limit: int = 10) -> list[dict[str, Any]]:
    """Query EpisodicStore for recent task history; returns [] on any failure."""
    try:
        from apyrobo.memory.episodic import EpisodicStore
        store = EpisodicStore()
        episodes = store.query(limit=limit, order="DESC")
        return [
            {
                "task": ep.task,
                "robot_id": ep.robot_id,
                "outcome": ep.outcome,
                "duration_s": ep.duration_s,
                "timestamp": ep.timestamp,
                "skills_run": ep.skills_run,
            }
            for ep in episodes
        ]
    except Exception:
        return []


def _run_robot_checks(robot: Any, uri: str) -> list[dict[str, Any]]:
    """Run the same checks as `connect --verify` and return them as dicts."""
    checks: list[dict[str, Any]] = []

    # Position
    try:
        pos = robot.get_position()
        checks.append({"name": "position", "status": "pass", "value": list(pos)})
    except Exception as exc:
        checks.append({"name": "position", "status": "fail", "value": None, "error": str(exc)})

    # Battery
    try:
        health_data = robot.get_health()
        battery = health_data.get("battery_pct")
        if battery is not None:
            status = "warn" if battery < 20 else "pass"
            checks.append({"name": "battery", "status": status, "value": round(float(battery), 1)})
        else:
            checks.append({"name": "battery", "status": "warn", "value": None})
    except Exception:
        checks.append({"name": "battery", "status": "warn", "value": None})

    # Capabilities
    try:
        caps = robot.capabilities()
        names = [c.name for c in caps.capabilities]
        checks.append({"name": "capabilities", "status": "pass", "value": names})
    except Exception as exc:
        checks.append({"name": "capabilities", "status": "fail", "value": None, "error": str(exc)})

    # Latency p50
    try:
        raw: list[float] = []
        for _ in range(3):
            t = time.monotonic()
            robot.get_position()
            raw.append(time.monotonic() - t)
        raw.sort()
        p50_ms = raw[len(raw) // 2] * 1000
        checks.append({"name": "latency_ms_p50", "status": "pass", "value": round(p50_ms, 1)})
    except Exception as exc:
        checks.append({"name": "latency_ms_p50", "status": "fail", "value": None, "error": str(exc)})

    return checks


def cmd_diagnose(args: argparse.Namespace) -> None:
    """Extended diagnostics with optional robot connection and JSON export."""
    import datetime

    uri: str | None = getattr(args, "robot", None)
    out: str | None = getattr(args, "out", None)
    timeout: float = getattr(args, "timeout", 10.0)

    # Install log capture early so we catch warnings during robot connect
    log_capture = _LogCapture(maxlen=20)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_capture)

    report: dict[str, Any] = {
        "generated_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "apyrobo_version": _get_apyrobo_version(),
        "system": _collect_system_info(),
        "robot": None,
        "health": None,
        "recent_tasks": [],
        "log_entries": [],
        "checks": [],
    }

    # Always run environment doctor checks
    doctor_results = run_doctor_checks()
    report["checks"] = [
        {"name": r.message.split()[0] if r.message else "check",
         "status": r.status, "message": r.message}
        for r in doctor_results
    ]

    if uri:
        robot, connect_time, error = _connect_with_timeout(uri, timeout)

        if error or robot is None:
            report["robot"] = {
                "uri": uri,
                "connected": False,
                "connect_time_s": round(connect_time, 3),
                "error": error or "unknown",
            }
        else:
            # Basic adapter state
            robot_info: dict[str, Any] = {
                "uri": uri,
                "connected": True,
                "connect_time_s": round(connect_time, 3),
            }
            try:
                pos = robot.get_position()
                robot_info["position"] = list(pos)
            except Exception:
                robot_info["position"] = None

            try:
                h = robot.get_health()
                robot_info["battery_pct"] = h.get("battery_pct")
            except Exception:
                robot_info["battery_pct"] = None

            report["robot"] = robot_info
            report["health"] = _collect_health_info(robot)
            report["checks"].extend(_run_robot_checks(robot, uri))

        report["recent_tasks"] = _collect_recent_tasks(limit=10)

    # Attach buffered log entries after everything has run
    report["log_entries"] = log_capture.entries()
    root_logger.removeHandler(log_capture)

    payload = json.dumps(report, indent=2, default=str)

    if out == "-" or (not out and uri is None and not sys.stdout.isatty()):
        # --out - : write JSON to stdout
        print(payload)
    elif out == "-":
        print(payload)
    elif out:
        with open(out, "w") as f:
            f.write(payload)
        print(f"Diagnostic report written to {out}")
    else:
        # Default: write to timestamped file
        import datetime as _dt
        ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        default_out = f"apyrobo-diag-{ts}.json"
        with open(default_out, "w") as f:
            f.write(payload)
        print(f"Diagnostic report written to {default_out}")


def _get_apyrobo_version() -> str:
    try:
        import apyrobo
        return apyrobo.__version__
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# apyrobo test-skill — isolated skill test runner
# ---------------------------------------------------------------------------

def cmd_test_skill(args: argparse.Namespace) -> None:
    """Run a skill against a mock robot and report results."""
    skill_id_or_file: str = args.skill
    robot_uri: str = getattr(args, "robot", "mock://test")
    params_json: str | None = getattr(args, "params", None)
    repeat: int = getattr(args, "repeat", 1)

    # Parse params
    params: dict[str, Any] = {}
    if params_json:
        try:
            params = json.loads(params_json)
        except json.JSONDecodeError as exc:
            print(f"Error: --params is not valid JSON: {exc}", file=sys.stderr)
            sys.exit(1)

    # Determine if skill_id_or_file is a file path
    from pathlib import Path as _Path
    skill_file = _Path(skill_id_or_file)
    skill_id = skill_id_or_file

    if skill_file.suffix == ".py" and skill_file.exists():
        # Import the file so @skill decorators run
        import importlib.util
        spec = importlib.util.spec_from_file_location("_test_skill_mod", skill_file)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
        # Use the stem as skill_id if not overridden
        skill_id = skill_file.stem

    # Resolve the handler — check global registry first, then decorated skills
    from apyrobo.skills.handlers import _DEFAULT_REGISTRY, dispatch as _dispatch
    from apyrobo.skills.skill import BUILTIN_SKILLS
    from apyrobo.skills.decorators import get_decorated_skills

    handler = _DEFAULT_REGISTRY.resolve(skill_id)
    if handler is None:
        # Try decorated skills (file may have registered via @skill)
        dec = get_decorated_skills()
        if skill_id in dec:
            _skill_def, fn = dec[skill_id]
            import inspect as _inspect
            accepted = set(_inspect.signature(fn).parameters)

            def handler(robot: Any, p: dict) -> bool:  # type: ignore[misc]
                filtered = {k: v for k, v in p.items() if k in accepted}
                result = fn(**filtered)
                return bool(result) if result is not None else True
        else:
            print(f"Error: skill {skill_id!r} not found in handler registry or @skill registry.",
                  file=sys.stderr)
            print("Hint: register it with @skill_handler or @skill before running.", file=sys.stderr)
            sys.exit(1)

    # Build the Skill metadata (for precondition check display)
    skill_meta = BUILTIN_SKILLS.get(skill_id)
    if skill_meta is None:
        dec = get_decorated_skills()
        if skill_id in dec:
            skill_meta = dec[skill_id][0]

    # Connect to robot
    try:
        robot = Robot.discover(robot_uri)
    except Exception as exc:
        print(f"Error: could not connect to {robot_uri!r}: {exc}", file=sys.stderr)
        sys.exit(1)

    # Merge default params from skill metadata with user-supplied params
    merged_params: dict[str, Any] = {}
    if skill_meta is not None:
        merged_params.update(skill_meta.parameters)
    merged_params.update(params)

    _W = 38
    print("─" * _W)
    print(f"Skill:    {skill_id}")
    print(f"Robot:    {robot_uri}")
    print(f"Runs:     {repeat}")
    print()

    times: list[float] = []
    passed = 0

    for i in range(1, repeat + 1):
        t0 = time.monotonic()
        exc_info: str | None = None
        retval: Any = None
        try:
            retval = handler(robot, merged_params)
            ok = bool(retval) if retval is not None else True
        except Exception as exc:
            ok = False
            exc_info = str(exc)

        elapsed = time.monotonic() - t0
        times.append(elapsed)
        if ok:
            passed += 1

        icon = "✅" if ok else "❌"
        detail = f"{retval}" if exc_info is None else f"raised: {exc_info}"
        print(f"  Run {i}  {icon}  {elapsed:.3f}s   {detail}")

    print()
    avg = sum(times) / len(times) if times else 0.0
    min_t = min(times) if times else 0.0
    max_t = max(times) if times else 0.0
    print(f"Passed: {passed}/{repeat}   Avg: {avg:.3f}s   Min: {min_t:.3f}s   Max: {max_t:.3f}s")
    print("─" * _W)

    if passed < repeat:
        sys.exit(1)


def cmd_voice(args: argparse.Namespace) -> None:
    """VC-01: Interactive voice control demo."""
    from apyrobo.voice import (
        MockVoiceAdapter, WhisperAdapter, OpenAIVoiceAdapter, voice_loop,
    )

    robot = Robot.discover(args.robot)
    provider, model = _resolve_provider(args.provider, getattr(args, "model", None))
    try:
        agent = Agent(provider=provider, **({"model": model} if model else {}))
    except ValueError as exc:
        print(f"Error: {exc}\n\nAvailable providers:\n{_PROVIDER_TABLE}", file=sys.stderr)
        sys.exit(1)

    adapter_map = {
        "mock": lambda: MockVoiceAdapter(["go to (2, 3)", "stop"]),
        "whisper": lambda: WhisperAdapter(),
        "openai": lambda: OpenAIVoiceAdapter(),
    }
    adapter = adapter_map[args.adapter]()

    if args.listen or args.adapter != "mock":
        print(f"Voice mode: {args.adapter} adapter")
        print(f"Robot:      {robot.robot_id}")
        print("Listening... (say 'stop' to exit)")
        print("-" * 50)

        def on_listen(text: str) -> None:
            print(f"  Heard: {text!r}")

        def on_result(result: Any) -> None:
            print(f"  Result: {result.status.value} "
                  f"({result.steps_completed}/{result.steps_total})")

        turns = voice_loop(
            agent=agent,
            robot=robot,
            adapter=adapter,
            max_turns=args.max_turns,
            on_listen=on_listen,
            on_result=on_result,
        )
        print("-" * 50)
        print(f"Completed {len(turns)} turn(s)")
    else:
        print("Use --listen to start voice interaction")


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

    # connect — one-command connection test
    p_conn = sub.add_parser("connect", help="Test connection to a robot")
    p_conn.add_argument("uri", help="Robot URI (e.g. ros2://turtlebot4, mock://test)")
    p_conn.add_argument("--verify", action="store_true",
                        help="Run full verification suite (position, battery, skills, latency, health)")
    p_conn.add_argument("--timeout", type=float, default=10.0, metavar="N",
                        help="Seconds to wait for connection (default 10)")
    p_conn.add_argument("--json", action="store_true", dest="json",
                        help="Machine-readable JSON output")

    # doctor / diagnose — environment diagnostics
    sub.add_parser("doctor", help="Diagnose the local environment and show fix hints")
    p_diag = sub.add_parser(
        "diagnose",
        help="Full diagnostic report (optionally connects to a robot)",
    )
    p_diag.add_argument(
        "--robot", metavar="URI", default=None,
        help="Robot URI to connect to (e.g. mock://turtlebot4)",
    )
    p_diag.add_argument(
        "--out", metavar="FILE", default=None,
        help="Output path for JSON report; use '-' for stdout",
    )
    p_diag.add_argument(
        "--timeout", type=float, default=10.0, metavar="SECS",
        help="Robot connection timeout in seconds (default: 10)",
    )

    # test-skill — isolated skill test runner
    p_ts = sub.add_parser(
        "test-skill",
        help="Run a skill against a mock robot and print a test report",
    )
    p_ts.add_argument(
        "skill", metavar="SKILL",
        help="Skill ID (e.g. 'move_to') or path to a .py skill file",
    )
    p_ts.add_argument(
        "--robot", metavar="URI", default="mock://turtlebot4",
        help="Robot URI (default: mock://turtlebot4)",
    )
    p_ts.add_argument(
        "--params", metavar="JSON", default="{}",
        help="Skill parameters as a JSON object (default: {})",
    )
    p_ts.add_argument(
        "--repeat", type=int, default=1, metavar="N",
        help="Number of times to run the skill (default: 1)",
    )

    # voice — VC-01
    p_voice = sub.add_parser("voice", help="Interactive voice control")
    p_voice.add_argument("--robot", default="mock://turtlebot4")
    p_voice.add_argument("--provider", default="rule")
    p_voice.add_argument("--adapter", default="mock",
                         choices=["whisper", "openai", "mock"],
                         help="Voice adapter backend")
    p_voice.add_argument("--listen", action="store_true",
                         help="Start interactive voice demo")
    p_voice.add_argument("--max-turns", type=int, default=None,
                         help="Maximum conversation turns")

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
        "connect": cmd_connect,
        "doctor": cmd_doctor,
        "diagnose": cmd_diagnose,
        "test-skill": cmd_test_skill,
        "voice": cmd_voice,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()

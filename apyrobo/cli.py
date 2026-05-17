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
import urllib.parse
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
    _profile_name = getattr(args, "profile", None)
    if _profile_name:
        from apyrobo.profiles import get_profile as _gp
        _profile = _gp(_profile_name)
        _default_model = _profile.llm_model
    else:
        _default_model = None
    provider, model = _resolve_provider(args.provider, getattr(args, "model", _default_model))
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

    # ST-01: Sim-to-real validation
    if getattr(args, "simulate", False):
        from apyrobo.skills.simtoreal import SimToRealTransfer
        sim_uri = getattr(args, "sim_robot", None) or args.robot
        real_uri = getattr(args, "real_robot", None)
        auto_deploy = getattr(args, "auto_deploy", False)
        transfer = SimToRealTransfer(sim_adapter_uri=sim_uri, real_adapter_uri=real_uri)
        print()
        print(f"Running simulation on {sim_uri!r} …")
        sim_result, deployed = transfer.run(graph, auto_deploy=auto_deploy)
        status_str = "SUCCESS" if sim_result.success else "FAILED"
        print(f"Simulation: {status_str} "
              f"({sim_result.steps_completed}/{sim_result.steps_total} steps)")
        if sim_result.failures:
            for f in sim_result.failures:
                print(f"  ✗ {f}")
        if auto_deploy:
            if deployed:
                print(f"Deployed to real robot {real_uri!r}: YES")
            else:
                print("Deployed to real robot: NO (sim failed or no real-robot URI)")


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

    # Auto-discovery: detect hardware spec from live ROS nodes (best-effort)
    detected_hw: dict[str, Any] = {}
    try:
        from apyrobo.hardware import HardwareRegistry, AutoDiscovery
        _hw_reg = HardwareRegistry()
        _disc = AutoDiscovery(_hw_reg)
        _hw_spec = _disc.detect()
        if _hw_spec:
            detected_hw = {
                "robot_id": _hw_spec.robot_id,
                "model": _hw_spec.model,
                "skill_package": _hw_spec.skill_package,
            }
            if not as_json:
                print(f"  Hardware: {_disc.summary(_hw_spec)}")
            _disc.load_skill_package(_hw_spec)
    except Exception:
        pass

    if not verify:
        if as_json:
            payload: dict[str, Any] = {
                "uri": uri,
                "connected": True,
                "connect_time_s": round(connect_time, 3),
            }
            if detected_hw:
                payload["hardware"] = detected_hw
            print(json.dumps(payload))
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

    _W = 50
    print("─" * _W)
    print(f"Skill:    {skill_id}")
    print(f"Robot:    {robot_uri}")
    print(f"Runs:     {repeat}")

    # Capability mismatch check — emit structured warning before running
    if skill_meta is not None:
        from apyrobo.core.schemas import CapabilityType
        required_cap: Any = getattr(skill_meta, "required_capability", None)
        if required_cap is not None and required_cap != CapabilityType.CUSTOM:
            try:
                caps = robot.capabilities()
                robot_cap_types = {c.capability_type for c in caps.capabilities}
                if required_cap not in robot_cap_types:
                    _CAP_PACKAGES = {
                        CapabilityType.MANIPULATE: "apyrobo-skills-ur / apyrobo-skills-franka",
                        CapabilityType.PICK:       "apyrobo-skills-ur / apyrobo-skills-franka",
                        CapabilityType.PLACE:      "apyrobo-skills-ur / apyrobo-skills-franka",
                        CapabilityType.NAVIGATE:   "apyrobo-skills-turtlebot4 / apyrobo-skills-spot",
                        CapabilityType.DOCK:       "apyrobo-skills-turtlebot4",
                    }
                    print()
                    print("  ⚠  Capability mismatch detected:")
                    print(f"       Skill requires: {required_cap.value!r}")
                    print(f"       Robot provides: {[c.capability_type.value for c in caps.capabilities]}")
                    hint = _CAP_PACKAGES.get(required_cap)
                    if hint:
                        print(f"       Fix:            pip install {hint}")
                    print("  (skill may still run if the robot handles the call gracefully)")
            except Exception:
                pass

    print()

    times: list[float] = []
    passed = 0
    failures: list[str] = []

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
        else:
            failures.append(exc_info or f"returned {retval!r}")

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
        # Structured failure summary with fix hints
        print()
        print("  Failure summary:")
        for msg in set(failures):
            print(f"    • {msg}")
            # Surface common fix hints
            if "AttributeError" in (msg or "") and "gripper" in (msg or "").lower():
                print("      Fix: robot does not support gripper — check robot capabilities")
            elif "TimeoutError" in (msg or "") or "timeout" in (msg or "").lower():
                print("      Fix: increase --timeout or check robot connection")
            elif "CapabilityError" in (msg or "") or "capability" in (msg or "").lower():
                print("      Fix: install the matching skill package for this robot type")
        sys.exit(1)


# ---------------------------------------------------------------------------
# apyrobo registry start — launch the FastAPI skill registry
# ---------------------------------------------------------------------------

def cmd_registry_start(args: argparse.Namespace) -> None:
    """Start the APYROBO skill registry server using uvicorn."""
    port: int = getattr(args, "port", 8080)
    db_path: str = getattr(args, "db", "./registry.db")
    host: str = getattr(args, "host", "0.0.0.0")

    # Surface db_path to the registry via env so SQLAlchemy can pick it up
    os.environ.setdefault("REGISTRY_DB_PATH", db_path)

    try:
        import uvicorn  # type: ignore[import]
    except ImportError:
        print(
            "Error: uvicorn is required to start the registry server.\n"
            "Install it with: pip install 'apyrobo[registry]'",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from apyrobo.registry.server import create_app
        app = create_app()
    except ImportError as exc:
        print(f"Error: could not load registry server: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting APYROBO skill registry on http://{host}:{port}")
    print(f"  Database: {db_path}")
    print("  Press Ctrl+C to stop\n")
    uvicorn.run(app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# apyrobo skill search / publish — remote registry client commands
# ---------------------------------------------------------------------------

def _registry_base_url(args: argparse.Namespace) -> str:
    return getattr(args, "registry", "http://localhost:8080").rstrip("/")


def cmd_skill_search(args: argparse.Namespace) -> None:
    """Search the remote skill registry."""
    query: str = args.query
    base = _registry_base_url(args)
    url = f"{base}/search?q={urllib.parse.quote(query)}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            results = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        print(
            f"Error: could not reach registry at {base}\n"
            f"  Reason: {exc}\n"
            "  Start the registry with: apyrobo registry start",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print(f"No skills found matching {query!r}")
        return

    print(f"Found {len(results)} skill(s) matching {query!r}:\n")
    for pkg in results:
        name = pkg.get("name", "?")
        version = pkg.get("version", "?")
        description = pkg.get("description", "")
        tags = ", ".join(pkg.get("tags", [])) or "—"
        print(f"  {name}  v{version}")
        print(f"    {description}")
        print(f"    Tags: {tags}")
        print()


def cmd_skill_publish(args: argparse.Namespace) -> None:
    """Publish a skill package to the remote registry."""
    base = _registry_base_url(args)
    url = f"{base}/skills"

    name: str = args.name
    version: str = args.version
    description: str = args.description
    author: str = getattr(args, "author", "")
    download_url: str = args.download_url
    token: str = args.token

    # Generate a placeholder checksum so the model validator passes
    import hashlib
    checksum = hashlib.sha256(f"{name}-{version}".encode()).hexdigest()

    payload = json.dumps({
        "package": {
            "name": name,
            "version": version,
            "description": description,
            "author": author,
            "license": "Apache-2.0",
            "tags": [],
            "download_url": download_url,
            "checksum": checksum,
            "apyrobo_version_min": "1.0.0",
        },
        "token": token,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
        print(f"Published: {result.get('name')} v{result.get('version')}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        print(f"Error: publish failed (HTTP {exc.code}): {body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(
            f"Error: could not reach registry at {base}\n"
            f"  Reason: {exc}\n"
            "  Start the registry with: apyrobo registry start",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# apyrobo skill compose — interactive skill composition REPL
# ---------------------------------------------------------------------------

def cmd_skill_compose(args: argparse.Namespace) -> None:
    """Start the interactive skill composition REPL."""
    robot_uri: str = getattr(args, "robot", "mock://turtlebot4")
    library_arg: str | None = getattr(args, "library", None)

    robot = Robot.discover(robot_uri)

    # Build library
    from apyrobo.skills.library import SkillLibrary
    if library_arg:
        # Treat as a Python file path: import it so @skill decorators fire
        from pathlib import Path as _Path
        lib_path = _Path(library_arg)
        if lib_path.exists() and lib_path.suffix == ".py":
            import importlib.util
            spec = importlib.util.spec_from_file_location("_compose_lib", lib_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
        library = SkillLibrary.from_decorated()
    else:
        library = SkillLibrary.from_decorated()

    from apyrobo.skills.compose import ComposeREPL
    ComposeREPL(robot, library).run()


def cmd_listen(args: argparse.Namespace) -> None:
    """VC-01: Listen for one voice utterance, plan from it, and print the plan."""
    from apyrobo.voice import (
        MockVoiceAdapter, WhisperVoiceAdapter, PiperVoiceAdapter,
        OpenAIVoiceAdapter,
    )

    robot = Robot.discover(args.robot)
    provider, model_name = _resolve_provider(args.provider, getattr(args, "model", None))
    try:
        agent = Agent(provider=provider, **({"model": model_name} if model_name else {}))
    except ValueError as exc:
        print(f"Error: {exc}\n\nAvailable providers:\n{_PROVIDER_TABLE}", file=sys.stderr)
        sys.exit(1)

    adapter_name = getattr(args, "adapter", "whisper")
    model_arg = getattr(args, "model", None)

    if adapter_name == "whisper":
        adapter = WhisperVoiceAdapter(model=model_arg or "base")
    elif adapter_name == "piper":
        adapter = PiperVoiceAdapter(model_path=model_arg)
    elif adapter_name == "openai":
        adapter = OpenAIVoiceAdapter()
    else:
        adapter = MockVoiceAdapter(listen_responses=["navigate to (1.0, 2.0)"])

    print(f"Adapter: {adapter_name}  Robot: {robot.robot_id}")
    print("Listening for one command…")

    text = adapter.listen()
    if not text:
        print("(no speech detected)")
        return

    print(f"Heard: {text!r}")
    graph = agent.plan(text, robot)
    print(f"Plan: {len(graph)} skill(s)")
    for i, skill in enumerate(graph.get_execution_order(), 1):
        params = graph.get_parameters(skill.skill_id)
        print(f"  {i}. {skill.name} ({skill.skill_id})")
        for k, v in (params or {}).items():
            print(f"       {k}: {v}")


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
# Serve command
# ---------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> None:
    """Start an orchestration server (stdio adapter by default)."""
    from apyrobo.orchestration import OrchestrationServer, StdioOrchestrationAdapter
    from apyrobo.core.robot import Robot

    robot_uri: str = getattr(args, "robot", "mock://turtlebot4")
    profile_name: str | None = getattr(args, "profile", None)
    provider_name: str = getattr(args, "provider", "rule")

    if profile_name:
        from apyrobo.profiles import get_profile as _gp
        _profile = _gp(profile_name)
        _default_model: str | None = _profile.llm_model
    else:
        _default_model = None

    provider, model = _resolve_provider(provider_name, _default_model)
    try:
        agent = Agent(provider=provider, **({"model": model} if model else {}))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    robot = Robot.discover(robot_uri)
    adapter = StdioOrchestrationAdapter()
    server = OrchestrationServer(adapter, agent, default_robot=robot)

    print(f"apyrobo serve — robot={robot_uri} provider={provider}", file=sys.stderr, flush=True)
    server.run()


# ---------------------------------------------------------------------------
# Profiles command
# ---------------------------------------------------------------------------

def cmd_profiles(args: argparse.Namespace) -> None:
    """List available compute profiles or show details for one profile."""
    from apyrobo.profiles import ProfileRegistry, get_profile as _get_profile

    sub = getattr(args, "profiles_command", None)
    name = getattr(args, "profile_name", None)
    as_json: bool = getattr(args, "json", False)

    reg = ProfileRegistry()

    if sub == "show" and name:
        profile = reg.get(name)
        if profile is None:
            print(f"Error: unknown profile '{name}'. Available: {', '.join(reg.names())}", file=sys.stderr)
            sys.exit(1)
        if as_json:
            print(json.dumps(profile.to_dict(), indent=2))
        else:
            d = profile.to_dict()
            print(f"Profile: {d['name']}")
            print(f"  {d['description']}")
            print(f"  LLM:        {d['llm_provider']} / {d['llm_model']}")
            if d["llm_api_base"]:
                print(f"  API base:   {d['llm_api_base']}")
            print(f"  Context:    {d['max_context_tokens']:,} tokens")
            print(f"  GPU:        {'yes (' + str(d['gpu_vram_gb']) + ' GB VRAM)' if d['gpu_available'] else 'no'}")
            print(f"  RAM:        {d['ram_gb']} GB")
            print(f"  Edge infer: {'yes' if d['edge_inference'] else 'no'}")
        return

    # Default: list all profiles
    profiles = reg.all()
    if as_json:
        print(json.dumps([p.to_dict() for p in profiles], indent=2))
    else:
        print(f"{'NAME':<20} {'LLM MODEL':<35} {'GPU':>5}  DESCRIPTION")
        print("-" * 80)
        for p in profiles:
            gpu = "yes" if p.gpu_available else "no"
            desc = p.description[:40] if len(p.description) > 40 else p.description
            print(f"{p.name:<20} {p.llm_model:<35} {gpu:>5}  {desc}")


# ---------------------------------------------------------------------------
# apyrobo init — scaffold a new pip-installable skill package
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> None:
    """Scaffold a new pip-installable APYROBO skill package."""
    import pathlib
    import textwrap

    raw_name: str = args.name.lower().strip()
    # Normalise to kebab-case for the package name, snake_case for the module
    pkg_name = "apyrobo-skills-" + raw_name.replace("_", "-")
    module_name = "apyrobo_skills_" + raw_name.replace("-", "_")
    out_dir = pathlib.Path(getattr(args, "directory", None) or raw_name)

    if out_dir.exists() and not getattr(args, "force", False):
        print(f"Error: directory '{out_dir}' already exists. Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    src_dir = out_dir / "src" / module_name
    tests_dir = out_dir / "tests"
    src_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    author = getattr(args, "author", "") or ""
    description = getattr(args, "description", "") or f"APYROBO skill package for {raw_name}"

    # pyproject.toml
    (out_dir / "pyproject.toml").write_text(textwrap.dedent(f"""\
        [build-system]
        requires = ["setuptools>=68"]
        build-backend = "setuptools.backends.legacy:build"

        [project]
        name = "{pkg_name}"
        version = "0.1.0"
        description = "{description}"
        requires-python = ">=3.10"
        dependencies = ["apyrobo>=3.0.0"]
        {('authors = [{name = ' + repr(author) + '}]') if author else ""}

        [project.entry-points."apyrobo.skills"]
        {module_name} = "{module_name}:register"

        [tool.pytest.ini_options]
        testpaths = ["tests"]
    """))

    # src/<module>/__init__.py
    (src_dir / "__init__.py").write_text(textwrap.dedent(f"""\
        \"\"\"APYROBO skill package: {pkg_name}.\"\"\"
        from .skills import register

        __all__ = ["register"]
    """))

    # src/<module>/skills.py
    skill_fn_name = raw_name.replace("-", "_")
    (src_dir / "skills.py").write_text(textwrap.dedent(f"""\
        \"\"\"Skills for {pkg_name}.\"\"\"
        from apyrobo.skills.decorators import skill
        from apyrobo.skills.library import SkillLibrary


        @skill(
            name="{skill_fn_name}_hello",
            description="Example skill — say hello from {raw_name}",
            required_capabilities=[],
        )
        def {skill_fn_name}_hello(robot, message: str = "hello") -> bool:
            print(f"[{raw_name}] {{message}}")
            return True


        def register(library: SkillLibrary) -> None:
            \"\"\"Entry-point called by APYROBO on startup.\"\"\"
            library.register_decorated()
    """))

    # tests/__init__.py
    (tests_dir / "__init__.py").write_text("")

    # tests/test_skills.py
    (tests_dir / "test_skills.py").write_text(textwrap.dedent(f"""\
        \"\"\"Smoke tests for {pkg_name}.\"\"\"
        import pytest
        from apyrobo.core.robot import Robot


        @pytest.fixture
        def robot():
            return Robot.discover("mock://test")


        def test_{skill_fn_name}_hello(robot):
            from {module_name}.skills import {skill_fn_name}_hello
            assert {skill_fn_name}_hello(robot) is True


        def test_{skill_fn_name}_hello_custom_message(robot):
            from {module_name}.skills import {skill_fn_name}_hello
            assert {skill_fn_name}_hello(robot, message="world") is True
    """))

    # README.md
    (out_dir / "README.md").write_text(textwrap.dedent(f"""\
        # {pkg_name}

        {description}

        ## Quick start

        ```bash
        pip install -e .
        apyrobo execute "{skill_fn_name} hello" --robot mock://turtlebot4
        ```

        ## Development

        ```bash
        pip install -e ".[dev]"
        pytest
        ```
    """))

    # .github/workflows/ci.yml
    gh_dir = out_dir / ".github" / "workflows"
    gh_dir.mkdir(parents=True, exist_ok=True)
    (gh_dir / "ci.yml").write_text(textwrap.dedent(f"""\
        name: CI

        on: [push, pull_request]

        jobs:
          test:
            runs-on: ubuntu-latest
            steps:
              - uses: actions/checkout@v4
              - uses: actions/setup-python@v5
                with:
                  python-version: "3.12"
              - run: pip install apyrobo pytest
              - run: pip install -e .
              - run: pytest
    """))

    print(f"Created: {out_dir}/")
    print(f"  Package:  {pkg_name}")
    print(f"  Module:   {module_name}")
    print(f"  Skill:    {skill_fn_name}_hello")
    print()
    print("Next steps:")
    print(f"  cd {out_dir}")
    print(f"  pip install -e .")
    print(f"  apyrobo test-skill {skill_fn_name}_hello --robot mock://turtlebot4")
    print(f"  pytest")


# ---------------------------------------------------------------------------
# apyrobo shell — interactive REPL with robot + skills pre-loaded
# ---------------------------------------------------------------------------

def cmd_shell(args: argparse.Namespace) -> None:
    """Drop into a Python REPL with robot, agent, and all skills pre-imported."""
    import code as _code

    robot_uri: str = getattr(args, "robot", "mock://turtlebot4")
    provider_name: str = getattr(args, "provider", "rule")

    print(f"Connecting to {robot_uri!r} …")
    try:
        robot = Robot.discover(robot_uri)
    except Exception as exc:
        print(f"Error: could not connect to {robot_uri!r}: {exc}", file=sys.stderr)
        sys.exit(1)

    provider, model = _resolve_provider(provider_name)
    try:
        agent = Agent(provider=provider, **({"model": model} if model else {}))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    caps = robot.capabilities()

    from apyrobo.skills.skill import BUILTIN_SKILLS
    from apyrobo.skills.executor import SkillGraph

    banner = (
        "\n"
        "  APYROBO Interactive Shell\n"
        "  ─────────────────────────────────────────────────────\n"
        f"  robot    → {caps.name!r} ({robot_uri})\n"
        f"  agent    → {provider} provider\n"
        f"  skills   → {len(BUILTIN_SKILLS)} built-in skills available\n"
        "\n"
        "  Try:\n"
        "    graph = agent.plan('navigate to dock', robot)\n"
        "    graph.get_execution_order()\n"
        "    robot.move(1.0, 0.0)\n"
        "    robot.capabilities()\n"
        "\n"
        "  Type exit() or Ctrl-D to quit.\n"
    )

    local_ns: dict = {
        "robot": robot,
        "agent": agent,
        "Robot": Robot,
        "Agent": Agent,
        "SkillGraph": SkillGraph,  # type: ignore[possibly-undefined]
        "BUILTIN_SKILLS": BUILTIN_SKILLS,
    }

    _code.interact(banner=banner, local=local_ns, exitmsg="Goodbye.")


# ---------------------------------------------------------------------------
# apyrobo tutorial — interactive guided walkthrough (mock mode only)
# ---------------------------------------------------------------------------

_TUTORIAL_STEPS = [
    {
        "title": "Step 1 — Discover a robot",
        "description": (
            "APYROBO talks to robots through URIs like ros2://turtlebot4 or mock://test.\n"
            "The 'mock://' prefix spins up an in-process simulator — no hardware needed."
        ),
        "code": "robot = Robot.discover('mock://turtlebot4')\nprint(robot.capabilities().name)",
        "expect": "TurtleBot4",
    },
    {
        "title": "Step 2 — Inspect capabilities",
        "description": (
            "Every robot exposes its capabilities so the planner knows what it can do.\n"
            "The mock TurtleBot4 supports navigation, cameras, and basic manipulation."
        ),
        "code": "caps = robot.capabilities()\nfor c in caps.capabilities:\n    print(f'  {c.name}: {c.description}')",
        "expect": None,
    },
    {
        "title": "Step 3 — Plan a task",
        "description": (
            "The Agent turns a natural-language task into a skill graph.\n"
            "The built-in rule-based planner works without any API key."
        ),
        "code": "agent = Agent(provider='rule')\ngraph = agent.plan('navigate to the dock', robot)\nfor s in graph.get_execution_order():\n    print(f'  {s.skill_id}: {s.name}')",
        "expect": None,
    },
    {
        "title": "Step 4 — Execute the plan",
        "description": (
            "SkillExecutor runs each skill in the graph, respecting dependencies\n"
            "and enforcing safety policies at every step."
        ),
        "code": (
            "from apyrobo.skills.executor import SkillExecutor\n"
            "from apyrobo.core.schemas import TaskStatus\n"
            "executor = SkillExecutor(robot)\n"
            "result = executor.execute_graph(graph)\n"
            "print('Result:', result.status.value)"
        ),
        "expect": "completed",
    },
    {
        "title": "Step 5 — Write your own skill",
        "description": (
            "Skills are plain Python functions decorated with @skill.\n"
            "They receive the robot and any parameters, and return True on success."
        ),
        "code": (
            "from apyrobo.skills.decorators import skill\n\n"
            "@skill(name='wave', description='Wave the robot arm')\n"
            "def wave(robot, repetitions: int = 3) -> bool:\n"
            "    for i in range(repetitions):\n"
            "        print(f'  wave {i+1}/{repetitions}')\n"
            "    return True\n\n"
            "wave(robot)"
        ),
        "expect": None,
    },
    {
        "title": "Step 6 — Test a skill",
        "description": (
            "apyrobo test-skill runs a skill against a mock robot and prints timing.\n"
            "For your own skill files: apyrobo test-skill my_skill.py"
        ),
        "code": None,  # command-line demo
        "cli": "apyrobo test-skill navigate_to --robot mock://turtlebot4",
        "expect": None,
    },
]


def cmd_tutorial(args: argparse.Namespace) -> None:
    """Interactive guided walkthrough — runs entirely in mock mode, no hardware needed."""
    interactive: bool = not getattr(args, "non_interactive", False)

    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║   APYROBO Interactive Tutorial                   ║")
    print("  ║   Zero hardware required — runs in mock mode     ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()

    from apyrobo.core.robot import Robot as _Robot
    from apyrobo.skills.agent import Agent as _Agent

    ns: dict = {"Robot": _Robot, "Agent": _Agent}

    for i, step in enumerate(_TUTORIAL_STEPS, 1):
        total = len(_TUTORIAL_STEPS)
        print(f"  ┌─ {step['title']} ({i}/{total})")
        print(f"  │")
        for line in step["description"].splitlines():
            print(f"  │  {line}")
        print(f"  │")

        if step.get("cli"):
            print(f"  │  $ {step['cli']}")
        elif step.get("code"):
            for line in step["code"].splitlines():
                print(f"  │  >>> {line}" if not line.startswith(" ") else f"  │  ... {line}")

        print(f"  │")

        if interactive and i < total:
            try:
                inp = input("  └─ Press Enter to continue (or 'q' to quit) … ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if inp in ("q", "quit", "exit"):
                print("  Exiting tutorial. Run 'apyrobo tutorial' any time to restart.")
                break
        else:
            print(f"  └─")

        if step.get("code") and step.get("expect") is None and not step.get("cli"):
            try:
                exec(step["code"], ns)  # noqa: S102
            except Exception:
                pass

        print()

    print("  Tutorial complete!")
    print("  Explore further:")
    print("    apyrobo shell --robot mock://turtlebot4   # interactive REPL")
    print("    apyrobo init my-robot                     # scaffold a skill package")
    print("    apyrobo doctor                            # diagnose your environment")
    print()


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
    p_plan.add_argument("--simulate", action="store_true",
                        help="Validate plan in simulation before reporting")
    p_plan.add_argument("--sim-robot", default=None,
                        help="Simulation robot URI (default: same as --robot)")
    p_plan.add_argument("--real-robot", default=None,
                        help="Real robot URI for deployment after sim validation")
    p_plan.add_argument("--auto-deploy", action="store_true",
                        help="Automatically deploy to real robot if sim succeeds")
    p_plan.add_argument("--profile", default=None, metavar="PROFILE",
                        help="Compute profile (jetson-orin, raspberry-pi, workstation-gpu, cloud, cpu-only)")

    # execute
    p_exec = sub.add_parser("execute", help="Plan and execute a task")
    p_exec.add_argument("task", help="Task description")
    p_exec.add_argument("--robot", default="mock://turtlebot4")
    p_exec.add_argument("--provider", default="rule")
    p_exec.add_argument("--max-speed", type=float, default=1.5)
    p_exec.add_argument("--force", action="store_true", help="Execute even if low confidence")
    p_exec.add_argument("--profile", default=None, metavar="PROFILE",
                        help="Compute profile (jetson-orin, raspberry-pi, workstation-gpu, cloud, cpu-only)")

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

    # registry — skill registry server management
    p_registry = sub.add_parser("registry", help="Manage the APYROBO skill registry server")
    registry_sub = p_registry.add_subparsers(dest="registry_command")
    p_reg_start = registry_sub.add_parser("start", help="Start the registry server")
    p_reg_start.add_argument(
        "--port", type=int, default=8080, metavar="PORT",
        help="Port to listen on (default: 8080)",
    )
    p_reg_start.add_argument(
        "--host", default="0.0.0.0", metavar="HOST",
        help="Bind host (default: 0.0.0.0)",
    )
    p_reg_start.add_argument(
        "--db", default="./registry.db", metavar="PATH",
        help="Path to SQLite database (default: ./registry.db)",
    )

    # skill — remote registry client commands
    p_skill = sub.add_parser("skill", help="Search and publish skills in the registry")
    skill_sub = p_skill.add_subparsers(dest="skill_command")

    p_skill_search = skill_sub.add_parser("search", help="Search the skill registry")
    p_skill_search.add_argument("query", help="Search term")
    p_skill_search.add_argument(
        "--registry", metavar="URL", default="http://localhost:8080",
        help="Registry base URL (default: http://localhost:8080)",
    )

    p_skill_publish = skill_sub.add_parser("publish", help="Publish a skill to the registry")
    p_skill_publish.add_argument("name", help="Package name (e.g. apyrobo-skills-myrobot)")
    p_skill_publish.add_argument("--version", required=True, help="SemVer version string")
    p_skill_publish.add_argument("--description", required=True, help="Package description")
    p_skill_publish.add_argument("--author", default="", help="Author name")
    p_skill_publish.add_argument(
        "--download-url", required=True, dest="download_url",
        help="URL to the wheel or tarball",
    )
    p_skill_publish.add_argument(
        "--token", required=True, help="Registry authentication token",
    )
    p_skill_publish.add_argument(
        "--registry", metavar="URL", default="http://localhost:8080",
        help="Registry base URL (default: http://localhost:8080)",
    )

    p_skill_compose = skill_sub.add_parser(
        "compose",
        help="Interactive REPL for chaining skills into a plan",
    )
    p_skill_compose.add_argument(
        "--robot", metavar="URI", default="mock://turtlebot4",
        help="Robot URI (default: mock://turtlebot4)",
    )
    p_skill_compose.add_argument(
        "--library", metavar="PATH", default=None,
        help="Path to a .py file with @skill-decorated skills to load",
    )

    # listen — VC-01: single-utterance voice→plan
    p_listen = sub.add_parser("listen", help="Listen for one voice command and plan from it")
    p_listen.add_argument("--robot", default="mock://turtlebot4")
    p_listen.add_argument("--provider", default="rule")
    p_listen.add_argument("--adapter", default="whisper",
                          choices=["whisper", "piper", "openai", "mock"],
                          help="Voice STT adapter backend (default: whisper)")
    p_listen.add_argument("--model", default=None,
                          help="Whisper model size (base/small/medium/large) or model path")

    # serve — orchestration server
    p_serve = sub.add_parser("serve", help="Start a stdio orchestration server")
    p_serve.add_argument("--robot", default="mock://turtlebot4", metavar="URI",
                         help="Robot URI (default: mock://turtlebot4)")
    p_serve.add_argument("--provider", default="rule",
                         help="LLM provider (default: rule)")
    p_serve.add_argument("--profile", default=None, metavar="PROFILE",
                         help="Compute profile to apply")

    # profiles
    p_profiles = sub.add_parser("profiles", help="List or inspect compute profiles")
    profiles_sub = p_profiles.add_subparsers(dest="profiles_command")
    p_profiles_show = profiles_sub.add_parser("show", help="Show details for a profile")
    p_profiles_show.add_argument("profile_name", help="Profile name")
    p_profiles_show.add_argument("--json", action="store_true")
    p_profiles.add_argument("--json", action="store_true", help="Output as JSON")

    # init — project scaffold
    p_init = sub.add_parser("init", help="Scaffold a new pip-installable skill package")
    p_init.add_argument("name", help="Robot/platform name (e.g. 'my-robot')")
    p_init.add_argument("--description", default="", help="One-line package description")
    p_init.add_argument("--author", default="", help="Author name")
    p_init.add_argument("--directory", default=None,
                        help="Output directory (default: ./<name>)")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing directory")

    # shell — interactive REPL
    p_shell = sub.add_parser("shell", help="Interactive Python REPL with robot and skills loaded")
    p_shell.add_argument("--robot", default="mock://turtlebot4", metavar="URI",
                         help="Robot URI (default: mock://turtlebot4)")
    p_shell.add_argument("--provider", default="rule",
                         help="LLM provider (default: rule)")

    # tutorial — guided walkthrough
    p_tutorial = sub.add_parser("tutorial", help="Interactive guided tutorial (mock mode, no hardware needed)")
    p_tutorial.add_argument("--non-interactive", action="store_true",
                            help="Run all steps without pausing for input")

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
        "registry": _cmd_registry_dispatch,
        "skill": _cmd_skill_dispatch,
        "listen": cmd_listen,
        "voice": cmd_voice,
        "profiles": cmd_profiles,
        "serve": cmd_serve,
        "init": cmd_init,
        "shell": cmd_shell,
        "tutorial": cmd_tutorial,
    }
    commands[args.command](args)


def _cmd_registry_dispatch(args: argparse.Namespace) -> None:
    sub = getattr(args, "registry_command", None)
    if sub == "start":
        cmd_registry_start(args)
    else:
        print("Usage: apyrobo registry <start>", file=sys.stderr)
        sys.exit(1)


def _cmd_skill_dispatch(args: argparse.Namespace) -> None:
    sub = getattr(args, "skill_command", None)
    if sub == "search":
        cmd_skill_search(args)
    elif sub == "publish":
        cmd_skill_publish(args)
    elif sub == "compose":
        cmd_skill_compose(args)
    else:
        print("Usage: apyrobo skill <search|publish|compose>", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

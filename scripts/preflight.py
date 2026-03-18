#!/usr/bin/env python3
"""
APYROBO Preflight Checker — run before every pilot session.

Validates that the environment is correctly set up:
  - Python package importable
  - Mock pipeline works (no ROS 2 needed)
  - Config file loads
  - Safety policy builds
  - ROS 2 available (skipped outside Docker)
  - Gazebo topics present (/odom, /cmd_vel, /scan)
  - Nav2 action server reachable
  - LLM reachable (if configured)

Usage:
    python3 scripts/preflight.py           # run all checks
    python3 scripts/preflight.py --ros     # require ROS 2 checks to pass
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure the workspace is on PYTHONPATH (works inside Docker or from repo root)
_workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _workspace not in sys.path:
    sys.path.insert(0, _workspace)


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0
_skipped = 0


def _mark(ok: bool, name: str, detail: str = "", hint: str = "") -> bool:
    """Print a check result and update counters."""
    global _passed, _failed, _skipped
    if ok:
        _passed += 1
        print(f"  \033[32m[\u2713]\033[0m {name}")
    else:
        _failed += 1
        print(f"  \033[31m[X]\033[0m {name}: {detail}")
        if hint:
            print(f"      \033[33mRun: {hint}\033[0m")
    return ok


def _skip(name: str, reason: str) -> None:
    global _skipped
    _skipped += 1
    print(f"  \033[33m[-]\033[0m {name}: {reason}")


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------

def check_python_import() -> bool:
    """Can we import the apyrobo package at all?"""
    try:
        import apyrobo  # noqa: F401
        return _mark(True, "Python import")
    except Exception as exc:
        return _mark(False, "Python import", str(exc),
                     hint="pip install -e . (from repo root)")


def check_mock_pipeline() -> bool:
    """Discover a mock robot and query capabilities."""
    try:
        from apyrobo.core.robot import Robot
        robot = Robot.discover("mock://t4")
        caps = robot.capabilities()
        assert caps is not None
        return _mark(True, "Mock pipeline")
    except Exception as exc:
        return _mark(False, "Mock pipeline", str(exc))


def check_config() -> bool:
    """Load the default configuration."""
    try:
        from apyrobo.config import ApyroboConfig
        cfg = ApyroboConfig.from_env()
        assert cfg is not None
        return _mark(True, "Config file")
    except Exception as exc:
        return _mark(False, "Config file", str(exc),
                     hint="cp config/apyrobo.example.yaml apyrobo.yaml")


def check_safety_policy() -> bool:
    """Build a safety policy from configuration."""
    try:
        from apyrobo.config import ApyroboConfig
        cfg = ApyroboConfig.from_env()
        policy = cfg.safety_policy()
        assert policy is not None
        return _mark(True, "Safety policy")
    except Exception as exc:
        return _mark(False, "Safety policy", str(exc))


def check_ros2_available(*, required: bool = False) -> bool:
    """Import rclpy — expected to fail outside the Docker container."""
    try:
        import rclpy  # noqa: F401
        return _mark(True, "ROS 2 available")
    except ImportError:
        if required:
            return _mark(False, "ROS 2 available", "rclpy not found",
                         hint="docker compose -f docker/docker-compose.yml exec apyrobo bash")
        _skip("ROS 2 available",
              "rclpy not found \u2014 are you inside the Docker container?")
        return False


def check_gazebo_topics() -> bool:
    """Verify that essential ROS 2 topics are published (/odom, /cmd_vel, /scan)."""
    try:
        import rclpy
        from rclpy.node import Node

        if not rclpy.ok():
            rclpy.init()

        node = rclpy.create_node("_preflight_topic_probe")
        try:
            # Give DDS a moment to discover
            time.sleep(1.0)
            topics = node.get_topic_names_and_types()
            topic_names = {name for name, _ in topics}
            essential = ["/odom", "/cmd_vel", "/scan"]
            missing = [t for t in essential if t not in topic_names]
            if missing:
                return _mark(False, "Gazebo topics",
                             f"missing {', '.join(missing)}",
                             hint="bash scripts/launch.sh")
            return _mark(True, "Gazebo topics")
        finally:
            node.destroy_node()
    except ImportError:
        _skip("Gazebo topics", "rclpy not available")
        return False
    except Exception as exc:
        return _mark(False, "Gazebo topics", str(exc),
                     hint="bash scripts/launch.sh")


def check_nav2_action_server() -> bool:
    """Check that the Nav2 navigate_to_pose action server is available."""
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.action import ActionClient
        from nav2_msgs.action import NavigateToPose

        if not rclpy.ok():
            rclpy.init()

        node = rclpy.create_node("_preflight_nav2_probe")
        try:
            client = ActionClient(node, NavigateToPose, "navigate_to_pose")
            ready = client.wait_for_server(timeout_sec=3.0)
            client.destroy()
            if ready:
                return _mark(True, "Nav2 ready")
            return _mark(False, "Nav2 ready",
                         "navigate_to_pose action server not responding",
                         hint="bash scripts/launch.sh  (wait ~40s for Nav2)")
        finally:
            node.destroy_node()
    except ImportError:
        _skip("Nav2 ready", "nav2_msgs not available")
        return False
    except Exception as exc:
        return _mark(False, "Nav2 ready", str(exc),
                     hint="bash scripts/launch.sh")


def check_llm_if_configured() -> bool:
    """If an LLM model is configured, verify we can reach it."""
    try:
        from apyrobo.config import ApyroboConfig
        cfg = ApyroboConfig.from_env()

        # Only test if user explicitly configured a model
        model = cfg.agent_model
        provider = cfg.agent_provider
        if provider in ("rule", "auto") and model is None:
            _skip("LLM reachable", "no LLM configured (using rule-based agent)")
            return True

        # Try a minimal litellm completion to verify connectivity
        import litellm
        resp = litellm.completion(
            model=model or "gpt-3.5-turbo",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            timeout=10,
        )
        assert resp.choices
        return _mark(True, "LLM reachable")
    except ImportError:
        _skip("LLM reachable", "litellm not installed")
        return False
    except Exception as exc:
        return _mark(False, "LLM reachable", str(exc),
                     hint="export OPENAI_API_KEY=... or set agent.model in apyrobo.yaml")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="APYROBO preflight checker")
    parser.add_argument("--ros", action="store_true",
                        help="Require ROS 2 checks to pass (fail instead of skip)")
    args = parser.parse_args()

    print()
    print("\033[1m  APYROBO Preflight Checklist\033[0m")
    print("\033[1m  " + "=" * 40 + "\033[0m")
    print()

    # ── Core checks (always run) ──────────────────────────────
    check_python_import()
    check_mock_pipeline()
    check_config()
    check_safety_policy()

    # ── ROS 2 / Gazebo checks (skip gracefully outside Docker) ─
    has_ros = check_ros2_available(required=args.ros)
    if has_ros:
        check_gazebo_topics()
        check_nav2_action_server()
    else:
        if not args.ros:
            _skip("Gazebo topics", "skipped (no ROS 2)")
            _skip("Nav2 ready", "skipped (no ROS 2)")

    # ── Optional: LLM ─────────────────────────────────────────
    check_llm_if_configured()

    # ── Summary ───────────────────────────────────────────────
    print()
    total = _passed + _failed + _skipped
    print(f"\033[1m  Results: {_passed} passed, {_failed} failed, {_skipped} skipped  ({total} total)\033[0m")

    if _failed:
        print(f"\n  \033[31mSome checks failed. Fix the issues above before proceeding.\033[0m")
        return 1

    if _skipped and not args.ros:
        print(f"\n  \033[33mSkipped checks are expected outside Docker.\033[0m")
        print(f"  To run the full stack:")
        print(f"    docker compose -f docker/docker-compose.yml exec apyrobo bash")
        print(f"    python3 scripts/preflight.py --ros")

    print(f"\n  \033[32mPreflight complete \u2014 ready to fly.\033[0m\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

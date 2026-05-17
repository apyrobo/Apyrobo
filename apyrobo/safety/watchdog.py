"""Per-skill watchdog — configurable dead-man switch per skill type.

The existing SafetyEnforcer watchdog checks odometry divergence globally.
SkillWatchdog is complementary: it enforces per-skill-type timeouts and
fires a configurable recovery action when a skill runs too long.

Unlike the timeout inside SkillExecutor (which raises SkillTimeout and
retries), the watchdog fires asynchronously via a background thread and
calls the recovery action regardless of what the skill thread is doing.

Usage::

    watchdog = SkillWatchdog.default(robot)
    with watchdog.arm("navigate_to", skill_id="navigate_to_0"):
        robot.move(5.0, 3.0)
    # watchdog disarmed automatically on __exit__

    # Override timeout for a specific skill type:
    watchdog.set_timeout("pick_object", seconds=30.0,
                         action=WatchdogAction.OPEN_GRIPPER)
"""
from __future__ import annotations

import enum
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from apyrobo.observability import emit_event

logger = logging.getLogger(__name__)


class WatchdogAction(enum.Enum):
    STOP = "stop"             # robot.stop()
    OPEN_GRIPPER = "open_gripper"  # robot.gripper_open()
    HOME = "home"             # home_fn() or robot.stop()
    LOG_ONLY = "log_only"     # emit event, no physical action
    NONE = "none"             # no-op


@dataclass
class WatchdogRule:
    """Timeout and recovery action for one skill type."""
    timeout_seconds: float
    action: WatchdogAction = WatchdogAction.STOP


@dataclass
class WatchdogFiring:
    """Record of a watchdog firing."""
    skill_type: str
    skill_id: str
    timeout_seconds: float
    action: WatchdogAction
    timestamp: float = field(default_factory=lambda: __import__("time").time())


class SkillWatchdog:
    """
    Per-skill-type dead-man switch.

    Each skill type can have its own timeout and recovery action.
    When a skill runs longer than its configured timeout, the watchdog:

    1. Fires the recovery action on the robot
    2. Emits a ``skill.watchdog_fired`` event
    3. Records the firing in ``firings``

    The skill itself is NOT interrupted — the watchdog only responds to
    time overruns. To also interrupt the skill, use SkillExecutor's
    built-in ``_run_with_timeout``.
    """

    # Built-in defaults by skill-type prefix
    _BUILTIN_RULES: dict[str, WatchdogRule] = {
        "navigate_to":    WatchdogRule(timeout_seconds=60.0,  action=WatchdogAction.STOP),
        "pick_object":    WatchdogRule(timeout_seconds=30.0,  action=WatchdogAction.OPEN_GRIPPER),
        "place_object":   WatchdogRule(timeout_seconds=30.0,  action=WatchdogAction.OPEN_GRIPPER),
        "grasp":          WatchdogRule(timeout_seconds=20.0,  action=WatchdogAction.OPEN_GRIPPER),
        "release":        WatchdogRule(timeout_seconds=10.0,  action=WatchdogAction.NONE),
        "report_status":  WatchdogRule(timeout_seconds=10.0,  action=WatchdogAction.LOG_ONLY),
        "capture_image":  WatchdogRule(timeout_seconds=15.0,  action=WatchdogAction.LOG_ONLY),
        "stop":           WatchdogRule(timeout_seconds=5.0,   action=WatchdogAction.NONE),
    }

    def __init__(
        self,
        robot: Any,
        default_timeout: float = 120.0,
        default_action: WatchdogAction = WatchdogAction.STOP,
        home_fn: "Any | None" = None,
    ) -> None:
        self._robot = robot
        self._default_rule = WatchdogRule(
            timeout_seconds=default_timeout,
            action=default_action,
        )
        self._rules: dict[str, WatchdogRule] = {}
        self._home_fn = home_fn
        self._firings: list[WatchdogFiring] = []
        self._lock = threading.Lock()

    @classmethod
    def default(cls, robot: Any) -> "SkillWatchdog":
        """Create a watchdog with built-in skill-type rules."""
        w = cls(robot)
        w._rules = dict(cls._BUILTIN_RULES)
        return w

    def set_timeout(
        self,
        skill_type: str,
        seconds: float,
        action: WatchdogAction = WatchdogAction.STOP,
    ) -> None:
        """Set a custom timeout and recovery action for a skill type."""
        self._rules[skill_type] = WatchdogRule(timeout_seconds=seconds, action=action)

    def rule_for(self, skill_type: str) -> WatchdogRule:
        """Return the rule for a skill type. Checks exact match then prefix match."""
        # Strip suffixed IDs (e.g. "navigate_to_0" → "navigate_to")
        clean = skill_type.rsplit("_", 1)
        base = clean[0] if len(clean) == 2 and clean[1].isdigit() else skill_type

        if base in self._rules:
            return self._rules[base]
        # Prefix scan
        for key, rule in self._rules.items():
            if base.startswith(key):
                return rule
        return self._default_rule

    @contextmanager
    def arm(self, skill_type: str, skill_id: str = "") -> Generator[None, None, None]:
        """Context manager that arms the watchdog for one skill execution.

        Starts a background timer; disarms it automatically on exit.
        If the timer fires, the recovery action runs in the timer thread.
        """
        rule = self.rule_for(skill_type)
        fired = threading.Event()

        def _fire() -> None:
            if fired.is_set():
                return
            fired.set()
            self._on_fire(skill_type, skill_id or skill_type, rule)

        timer = threading.Timer(rule.timeout_seconds, _fire)
        timer.daemon = True
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
            fired.set()  # prevent late fire after yield exits

    def _on_fire(self, skill_type: str, skill_id: str, rule: WatchdogRule) -> None:
        """Called from the timer thread when a watchdog fires."""
        firing = WatchdogFiring(
            skill_type=skill_type,
            skill_id=skill_id,
            timeout_seconds=rule.timeout_seconds,
            action=rule.action,
        )
        with self._lock:
            self._firings.append(firing)

        logger.warning(
            "SkillWatchdog fired for %r (type=%r timeout=%.1fs action=%s)",
            skill_id, skill_type, rule.timeout_seconds, rule.action.value,
        )

        try:
            emit_event(
                "skill.watchdog_fired",
                skill_id=skill_id,
                skill_type=skill_type,
                timeout_seconds=rule.timeout_seconds,
                action=rule.action.value,
            )
        except Exception as exc:
            logger.debug("emit_event error in watchdog: %s", exc)

        self._execute_action(rule.action)

    def _execute_action(self, action: WatchdogAction) -> None:
        """Execute the recovery action on the robot. Never raises."""
        try:
            if action == WatchdogAction.STOP:
                self._robot.stop()
            elif action == WatchdogAction.OPEN_GRIPPER:
                self._robot.gripper_open()
            elif action == WatchdogAction.HOME:
                if self._home_fn is not None:
                    self._home_fn()
                else:
                    self._robot.stop()
            elif action in (WatchdogAction.LOG_ONLY, WatchdogAction.NONE):
                pass
        except Exception as exc:
            logger.error("SkillWatchdog recovery action failed: %s", exc)

    @property
    def firings(self) -> list[WatchdogFiring]:
        """All watchdog firings recorded so far."""
        with self._lock:
            return list(self._firings)

    @property
    def fire_count(self) -> int:
        with self._lock:
            return len(self._firings)

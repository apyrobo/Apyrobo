"""
Policy-backed skills — run a learned policy as a skill-graph node.

The VLA deployment literature is unambiguous: neural policies (π0,
OpenVLA, anything LeRobot-shaped) cannot be formally certified, so
**runtime safety monitors are mandatory**. This module makes a learned
policy just another node in an APYROBO skill graph, with the same safety
envelope, timeout, and recovery semantics every other skill gets:

    from apyrobo import Robot, SafetyEnforcer
    from apyrobo.skills.policy import PolicyRunner

    robot = Robot.discover("ros2://arm1")
    enforcer = SafetyEnforcer(robot)          # hard constraints, audit log

    runner = PolicyRunner(
        policy=my_policy,                     # anything Policy-shaped
        robot=enforcer,                       # enforcer is robot-shaped
        hz=5.0,
        max_duration_sec=60.0,
        success=lambda obs: obs["distance_to_goal"] < 0.1,
    )
    result = runner.run()

Or inside a plan, via the built-in ``run_policy`` skill handler:

    graph.add_skill(BUILTIN_SKILLS["run_policy"],
                    parameters={"policy": my_policy,
                                "max_duration_sec": 60.0})

A *policy* is any object with ``select_action(observation) -> action``
(and optionally ``reset()``). Observations are plain dicts built from the
robot each tick; actions are dicts of **position deltas**
(``{"dx": ..., "dy": ...}`` — the common VLA base-action shape) or
absolute targets (``{"x": ..., "y": ...}``). Every action passes through:

1. the runner's own hard bound (``max_step_m`` per tick — a runaway
   policy physically cannot ask for more than one bounded step), then
2. whatever the ``robot`` object enforces — pass a ``SafetyEnforcer``
   and speed clamps, collision zones, and the audit trail all apply.

Wrapping a LeRobot policy is a few lines (torch stays *your*
dependency; APYROBO core never imports it)::

    class LeRobotBasePolicy:
        def __init__(self, lerobot_policy, to_tensor, from_tensor):
            self._p = lerobot_policy
            self._to, self._from = to_tensor, from_tensor
        def reset(self):
            self._p.reset()
        def select_action(self, obs):
            action = self._p.select_action(self._to(obs))
            return self._from(action)   # → {"dx": ..., "dy": ...}

Status: verified with deterministic mock policies (including an
adversarial runaway policy that the bounds must contain). Not yet run
against a real VLA checkpoint — that integration example is tracked on
the roadmap wedge.
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Policy(Protocol):
    """Anything that maps an observation dict to an action dict."""

    def select_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Return the next action for *observation*."""
        ...  # pragma: no cover


@dataclass
class PolicyResult:
    """Outcome of a PolicyRunner.run() episode."""

    success: bool
    steps: int
    duration_sec: float
    stop_reason: str
    final_position: tuple[float, float]
    clamped_steps: int = 0
    error: str | None = None
    trajectory: list[tuple[float, float]] = field(default_factory=list)


class PolicyRunner:
    """Execute a policy against a robot-shaped object, safely.

    ``robot`` is anything robot-shaped (``move``/``stop``/``get_position``);
    pass a :class:`~apyrobo.safety.enforcer.SafetyEnforcer` to get hard
    constraints and the audit trail — the runner adds its own per-tick
    step bound on top, so even with a bare robot a runaway policy is
    contained.
    """

    def __init__(
        self,
        policy: Policy,
        robot: Any,
        hz: float = 10.0,
        max_duration_sec: float = 120.0,
        max_steps: int | None = None,
        max_step_m: float = 0.5,
        speed: float | None = None,
        success: Callable[[dict[str, Any]], bool] | None = None,
        observe: Callable[[Any], dict[str, Any]] | None = None,
        real_time: bool = True,
    ) -> None:
        if not isinstance(policy, Policy):
            raise TypeError(
                f"policy must have select_action(observation); got {type(policy).__name__}"
            )
        self.policy = policy
        self.robot = robot
        self.hz = float(hz)
        self.max_duration_sec = float(max_duration_sec)
        self.max_steps = max_steps
        self.max_step_m = float(max_step_m)
        self.speed = speed
        self.success = success
        self.observe = observe or self._default_observe
        self.real_time = real_time

    @staticmethod
    def _default_observe(robot: Any) -> dict[str, Any]:
        x, y = robot.get_position()
        obs: dict[str, Any] = {"x": x, "y": y, "position": (x, y)}
        get_theta = getattr(robot, "get_orientation", None)
        if callable(get_theta):
            obs["theta"] = get_theta()
        return obs

    def _bounded_target(
        self, action: dict[str, Any], pos: tuple[float, float]
    ) -> tuple[float, float, bool]:
        """Resolve the action to an absolute target, bounded to max_step_m.

        Returns (x, y, clamped).
        """
        px, py = pos
        if "dx" in action or "dy" in action:
            dx = float(action.get("dx", 0.0))
            dy = float(action.get("dy", 0.0))
        elif "x" in action or "y" in action:
            dx = float(action.get("x", px)) - px
            dy = float(action.get("y", py)) - py
        else:
            raise ValueError(
                f"policy action must contain dx/dy or x/y, got {sorted(action)}"
            )
        dist = math.hypot(dx, dy)
        if dist > self.max_step_m and dist > 0.0:
            scale = self.max_step_m / dist
            return px + dx * scale, py + dy * scale, True
        return px + dx, py + dy, False

    def run(self) -> PolicyResult:
        """Run the policy loop until success, a bound, or an error.

        The robot is always stopped on the way out, whatever happens.
        """
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

        t0 = time.monotonic()
        steps = 0
        clamped = 0
        trajectory: list[tuple[float, float]] = []
        stop_reason, err, ok = "max_duration", None, False
        try:
            while (time.monotonic() - t0) < self.max_duration_sec:
                if self.max_steps is not None and steps >= self.max_steps:
                    stop_reason = "max_steps"
                    break

                obs = self.observe(self.robot)
                trajectory.append(tuple(obs.get("position", (0.0, 0.0))))
                if self.success is not None and self.success(obs):
                    stop_reason, ok = "success", True
                    break

                action = self.policy.select_action(obs)
                if action is None:
                    stop_reason, ok = "policy_done", self.success is None
                    break

                tx, ty, was_clamped = self._bounded_target(
                    action, obs["position"]
                )
                clamped += int(was_clamped)
                self.robot.move(x=tx, y=ty, speed=self.speed)
                steps += 1

                if self.real_time and self.hz > 0:
                    time.sleep(1.0 / self.hz)
        except Exception as exc:  # policy or robot failure — fail closed
            stop_reason, err, ok = "error", f"{type(exc).__name__}: {exc}", False
            logger.warning("PolicyRunner: aborting episode — %s", err)
        finally:
            try:
                self.robot.stop()
            except Exception:  # never mask the episode outcome
                logger.exception("PolicyRunner: stop() failed during teardown")

        x, y = self.robot.get_position()
        result = PolicyResult(
            success=ok,
            steps=steps,
            duration_sec=time.monotonic() - t0,
            stop_reason=stop_reason,
            final_position=(x, y),
            clamped_steps=clamped,
            error=err,
            trajectory=trajectory,
        )
        logger.info(
            "PolicyRunner: %s after %d steps (%.1fs, %d clamped) — %s",
            "SUCCESS" if ok else "no-success", steps,
            result.duration_sec, clamped, stop_reason,
        )
        return result

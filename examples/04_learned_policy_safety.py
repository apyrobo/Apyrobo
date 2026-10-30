#!/usr/bin/env python3
"""
Learned policy, safety envelope — run a LeRobot-shaped policy on mock://.

A learned policy (VLA, diffusion, ACT, anything LeRobot-shaped) can't be
certified, so APYROBO wraps it at runtime instead. This example drives a
small stand-in policy through ``PolicyRunner`` + ``SafetyEnforcer`` in
three episodes:

  1. nominal    — the policy reaches its goal; nothing intervenes
  2. runaway    — the policy spikes once to a 25 m jump; the runner
                  turns it into one bounded 0.5 m step and the episode
                  still finishes
  3. no-go zone — the policy steers into a keep-out area; the enforcer
                  rejects the move, the episode fails closed, the robot
                  is stopped and the rejection is in the audit log

The stand-in policy speaks LeRobot's interface (``reset()`` and
``select_action(batch) -> action``, a batch of named features in, a
flat action vector out). Plain lists replace torch tensors so the example
runs anywhere. To use a real checkpoint, load it with LeRobot and pass it
to ``LeRobotBasePolicy`` with your own observation/action mappings; the
runner, bounds and enforcer stay exactly the same.

Run from the repo root (no ROS 2, GPU or torch needed):
    python examples/04_learned_policy_safety.py
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from apyrobo import Robot
from apyrobo.safety.enforcer import SafetyEnforcer, SafetyPolicy
from apyrobo.skills.policy import PolicyResult, PolicyRunner

GOAL = (3.0, 0.0)
# Keep-out rectangle beside the path (think: a workcell people stand in).
NO_GO_ZONE = {"x_min": 1.0, "x_max": 2.0, "y_min": 0.5, "y_max": 1.5}


# ── A LeRobot-shaped policy ────────────────────────────────────────────
class StandInPolicy:
    """Stand-in for a trained policy: same calls as a LeRobot policy.

    ``select_action`` takes a batch dict (``"observation.state"`` is the
    robot's [x, y]) and returns a flat [dx, dy] action. ``mode`` picks the
    failure we want to demonstrate.
    """

    def __init__(self, goal: tuple[float, float], mode: str = "nominal") -> None:
        self.goal = goal
        self.mode = mode
        self._t = 0

    def reset(self) -> None:
        self._t = 0

    def select_action(self, batch: dict[str, list[float]]) -> list[float]:
        x, y = batch["observation.state"]
        self._t += 1
        if self.mode == "runaway" and self._t == 2:
            return [25.0, 0.0]                     # one wild extrapolation
        if self.mode == "no_go" and self._t >= 2:
            return [0.25, 0.35]                    # drifts up into the zone
        dx, dy = self.goal[0] - x, self.goal[1] - y
        dist = math.hypot(dx, dy)
        step = min(0.4, dist)
        return [dx / dist * step, dy / dist * step] if dist else [0.0, 0.0]


# ── The adapter: LeRobot interface → APYROBO Policy ────────────────────
class LeRobotBasePolicy:
    """Wrap any object with LeRobot's ``select_action``/``reset``.

    ``to_batch`` maps APYROBO's observation dict to the policy's input;
    ``from_action`` maps its output to ``{"dx": ..., "dy": ...}``. With a
    real checkpoint these are where tensors and camera frames come in.
    """

    def __init__(
        self,
        lerobot_policy: Any,
        to_batch: Callable[[dict[str, Any]], Any],
        from_action: Callable[[Any], dict[str, float]],
    ) -> None:
        self._p = lerobot_policy
        self._to, self._from = to_batch, from_action

    def reset(self) -> None:
        self._p.reset()

    def select_action(self, obs: dict[str, Any]) -> dict[str, float]:
        return self._from(self._p.select_action(self._to(obs)))


def to_batch(obs: dict[str, Any]) -> dict[str, list[float]]:
    return {"observation.state": [obs["x"], obs["y"]]}


def from_action(action: list[float]) -> dict[str, float]:
    return {"dx": float(action[0]), "dy": float(action[1])}


def near_goal(obs: dict[str, Any]) -> bool:
    return math.hypot(obs["x"] - GOAL[0], obs["y"] - GOAL[1]) < 0.05


# ── Episodes ───────────────────────────────────────────────────────────
def run_episode(mode: str) -> tuple[PolicyResult, SafetyEnforcer]:
    robot = Robot.discover(f"mock://policy-{mode}")
    enforcer = SafetyEnforcer(
        robot,
        policy=SafetyPolicy(name="workcell", collision_zones=[NO_GO_ZONE]),
    )
    policy = LeRobotBasePolicy(StandInPolicy(GOAL, mode), to_batch, from_action)
    runner = PolicyRunner(
        policy=policy,
        robot=enforcer,          # every action goes through the enforcer
        hz=10.0,
        max_steps=30,
        max_step_m=0.5,          # the runner's own per-tick bound
        success=near_goal,
        real_time=False,         # no sleeping between ticks in a demo
    )
    return runner.run(), enforcer


def main(verbose: bool = True) -> dict[str, tuple[PolicyResult, SafetyEnforcer]]:
    results = {mode: run_episode(mode) for mode in ("nominal", "runaway", "no_go")}
    if verbose:
        for mode, (r, enf) in results.items():
            x, y = r.final_position
            print(f"{mode:>8}: success={r.success} stop={r.stop_reason} "
                  f"steps={r.steps} clamped={r.clamped_steps} "
                  f"final=({x:.2f}, {y:.2f})")
            for v in enf.violations:
                vx, vy = v["requested_position"]
                print(f"          rejected: {v['type']} at ({vx:.2f}, {vy:.2f})")
            if r.error:
                print(f"          error: {r.error}")
    return results


if __name__ == "__main__":
    main()

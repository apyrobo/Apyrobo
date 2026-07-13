"""
Policy-backed skills: PolicyRunner semantics, safety bounds (including an
adversarial runaway policy), the run_policy builtin, and integration with
SafetyEnforcer and the skill executor. All with deterministic mock
policies — no torch, no LeRobot.
"""
from __future__ import annotations

import math
from typing import Any

import pytest

from apyrobo.core.robot import Robot
from apyrobo.skills.policy import PolicyRunner
from apyrobo.skills.skill import BUILTIN_SKILLS


class GoalSeekingPolicy:
    """Deterministic mock: steps 0.4 m toward a goal each tick."""

    def __init__(self, goal: tuple[float, float], step: float = 0.4) -> None:
        self.goal = goal
        self.step = step
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def select_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        px, py = obs["position"]
        dx, dy = self.goal[0] - px, self.goal[1] - py
        dist = math.hypot(dx, dy)
        if dist <= self.step:
            return {"dx": dx, "dy": dy}
        return {"dx": dx / dist * self.step, "dy": dy / dist * self.step}


class RunawayPolicy:
    """Adversarial: demands a 100 m jump every tick."""

    def select_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {"dx": 100.0, "dy": 0.0}


def near(goal, tol=0.15):
    return lambda obs: math.hypot(
        obs["x"] - goal[0], obs["y"] - goal[1]
    ) < tol


@pytest.fixture()
def robot():
    return Robot.discover("mock://policybot")


def make_runner(policy, robot, **kw):
    kw.setdefault("real_time", False)
    kw.setdefault("max_duration_sec", 10.0)
    return PolicyRunner(policy, robot, **kw)


# ---------------------------------------------------------------------------
# Core loop semantics
# ---------------------------------------------------------------------------

class TestPolicyRunner:
    def test_reaches_goal_and_reports_success(self, robot):
        goal = (2.0, 1.0)
        policy = GoalSeekingPolicy(goal)
        result = make_runner(policy, robot, success=near(goal)).run()
        assert result.success and result.stop_reason == "success"
        assert math.hypot(result.final_position[0] - 2.0,
                          result.final_position[1] - 1.0) < 0.15
        assert policy.resets == 1
        assert result.steps > 1, "goal must be reached stepwise, not teleported"

    def test_max_steps_bound(self, robot):
        result = make_runner(
            GoalSeekingPolicy((50.0, 0.0)), robot, max_steps=3
        ).run()
        assert result.stop_reason == "max_steps"
        assert result.steps == 3 and not result.success

    def test_policy_none_action_ends_episode(self, robot):
        class OneShot:
            def select_action(self, obs):
                return None
        result = make_runner(OneShot(), robot).run()
        assert result.stop_reason == "policy_done"
        # No success predicate given → a clean policy-done counts as success.
        assert result.success

    def test_policy_exception_fails_closed_and_stops_robot(self, robot):
        class Exploding:
            def select_action(self, obs):
                raise RuntimeError("NaN in action head")
        stops: list[bool] = []
        original_stop = robot.stop
        robot.stop = lambda: (stops.append(True), original_stop())[1]  # type: ignore[method-assign]
        result = make_runner(Exploding(), robot).run()
        assert result.stop_reason == "error" and not result.success
        assert "NaN in action head" in (result.error or "")
        assert stops, "robot.stop() must be called on the way out"

    def test_rejects_non_policy_objects(self, robot):
        with pytest.raises(TypeError, match="select_action"):
            PolicyRunner(object(), robot)

    def test_action_must_contain_deltas_or_targets(self, robot):
        class Junk:
            def select_action(self, obs):
                return {"thrust": 1.0}
        result = make_runner(Junk(), robot).run()
        assert result.stop_reason == "error"
        assert "dx/dy or x/y" in (result.error or "")

    def test_absolute_target_actions_work(self, robot):
        class Absolute:
            def __init__(self):
                self.sent = False
            def select_action(self, obs):
                if self.sent:
                    return None
                self.sent = True
                return {"x": 0.3, "y": 0.2}
        result = make_runner(Absolute(), robot).run()
        assert result.final_position == pytest.approx((0.3, 0.2))


# ---------------------------------------------------------------------------
# Safety bounds
# ---------------------------------------------------------------------------

class TestSafetyBounds:
    def test_runaway_policy_is_contained_per_tick(self, robot):
        moves: list[tuple[float, float]] = []
        original_move = robot.move
        def spy(x, y, speed=None):
            moves.append((x, y))
            original_move(x=x, y=y, speed=speed)
        robot.move = spy  # type: ignore[method-assign]

        result = make_runner(
            RunawayPolicy(), robot, max_steps=5, max_step_m=0.5
        ).run()
        assert result.clamped_steps == 5
        prev = (0.0, 0.0)
        for target in moves:
            step = math.hypot(target[0] - prev[0], target[1] - prev[1])
            assert step <= 0.5 + 1e-9, f"unbounded step: {step}"
            prev = target

    def test_runner_composes_with_safety_enforcer(self, robot):
        from apyrobo.safety.enforcer import SafetyEnforcer

        enforcer = SafetyEnforcer(robot)
        goal = (1.0, 0.5)
        # Demand an illegal speed: the enforcer must clamp it (and audit the
        # violation) while the episode still succeeds.
        result = make_runner(
            GoalSeekingPolicy(goal), enforcer, success=near(goal), speed=99.0
        ).run()
        assert result.success
        clamps = [
            e for e in enforcer.audit_log
            if e.event_type == "intervention"
            and e.details.get("type") == "speed_clamped"
        ]
        assert clamps, "over-speed policy actions must be clamped and audited"


# ---------------------------------------------------------------------------
# Builtin skill + executor integration
# ---------------------------------------------------------------------------

class TestRunPolicySkill:
    def test_builtin_registered(self):
        assert "run_policy" in BUILTIN_SKILLS
        assert BUILTIN_SKILLS["run_policy"].required_capability.value == "navigate"

    def test_executes_inside_a_skill_graph(self, robot):
        from apyrobo.skills.executor import SkillExecutor, SkillGraph

        goal = (1.5, 0.0)
        graph = SkillGraph()
        graph.add_skill(
            BUILTIN_SKILLS["run_policy"],
            parameters={
                "policy": GoalSeekingPolicy(goal),
                "success": near(goal),
                "real_time": False,
            },
        )
        from apyrobo.core.schemas import TaskStatus

        executor = SkillExecutor(robot)
        result = executor.execute_graph(graph)
        assert result.status == TaskStatus.COMPLETED
        assert result.steps_completed == 1
        assert math.hypot(robot.get_position()[0] - 1.5,
                          robot.get_position()[1]) < 0.15

    def test_missing_policy_fails_the_skill(self, robot):
        from apyrobo.skills.handlers import dispatch

        ok = dispatch("run_policy", robot, {})
        assert ok is False

"""Sim-to-real runs and deploys through the SafetyEnforcer.

docs/comparison.md and docs/architecture.md say every command passes through
the SafetyEnforcer before reaching the robot. SimToRealTransfer (used by
``apyrobo plan --simulate --auto-deploy``) used to discover the sim and the
real robot itself and drive both raw, so "go to (3, 4) at speed 5" was sent
to the real robot at 5 m/s and no no-go zone was checked.
"""

from __future__ import annotations

import pytest

from apyrobo.core.robot import Robot
from apyrobo.safety.enforcer import POLICY_REGISTRY, SafetyPolicy
from apyrobo.skills.agent import Agent
from apyrobo.skills.simtoreal import SimToRealTransfer

SIM = "mock://sim_safety"
REAL = "mock://real_safety"


@pytest.fixture
def moves(monkeypatch):
    """Record every move() each discovered robot is sent, keyed by URI."""
    sent: dict[str, list[dict]] = {SIM: [], REAL: []}
    real_discover = Robot.discover

    def discover(uri, **kwargs):
        robot = real_discover(uri, **kwargs)
        real_move = robot.move

        def spy(*args, **kw):
            sent.setdefault(uri, []).append(dict(kw))
            return real_move(*args, **kw)

        robot.move = spy  # type: ignore[method-assign]
        return robot

    monkeypatch.setattr(Robot, "discover", staticmethod(discover))
    return sent


def _graph(task: str):
    return Agent(provider="rule").plan(task, Robot.discover("mock://planner"))


# --- agent path (task string) ---------------------------------------------

def test_deploy_clamps_real_robot_speed_to_default_policy(moves):
    t = SimToRealTransfer(SIM, REAL)
    t.deploy("go to (3, 4) at speed 5", agent=Agent(provider="rule"))
    assert len(moves[REAL]) == 1
    assert moves[REAL][0]["speed"] == pytest.approx(POLICY_REGISTRY["default"].max_speed)


def test_validate_clamps_sim_robot_speed_to_default_policy(moves):
    t = SimToRealTransfer(SIM, REAL)
    t.validate("go to (3, 4) at speed 5", agent=Agent(provider="rule"))
    assert len(moves[SIM]) == 1
    assert moves[SIM][0]["speed"] == pytest.approx(POLICY_REGISTRY["default"].max_speed)


def test_run_auto_deploy_clamps_both_robots_to_given_policy(moves):
    t = SimToRealTransfer(SIM, REAL, safety_policy=SafetyPolicy(name="slow", max_speed=0.2))
    sim_result, deployed = t.run(
        "go to (3, 4) at speed 5", agent=Agent(provider="rule"), auto_deploy=True,
    )
    assert sim_result.success and deployed
    assert moves[SIM][0]["speed"] == pytest.approx(0.2)
    assert moves[REAL][0]["speed"] == pytest.approx(0.2)


def test_registered_policy_name_is_accepted(moves):
    t = SimToRealTransfer(SIM, REAL, safety_policy="strict")
    t.deploy("go to (3, 4) at speed 5", agent=Agent(provider="rule"))
    assert moves[REAL][0]["speed"] == pytest.approx(POLICY_REGISTRY["strict"].max_speed)


def test_speed_within_limit_is_unchanged(moves):
    t = SimToRealTransfer(SIM, REAL)
    t.deploy("go to (3, 4) at speed 0.3", agent=Agent(provider="rule"))
    assert moves[REAL][0]["speed"] == pytest.approx(0.3)


def test_no_go_zone_fails_sim_and_nothing_is_deployed(moves):
    zone = SafetyPolicy(
        name="zone",
        collision_zones=[{"x_min": 2, "x_max": 4, "y_min": 2, "y_max": 4}],
    )
    t = SimToRealTransfer(SIM, REAL, safety_policy=zone)
    sim_result, deployed = t.run(
        "go to (3, 3)", agent=Agent(provider="rule"), auto_deploy=True,
    )
    assert not sim_result.success
    assert not deployed
    assert moves[SIM] == []
    assert moves[REAL] == []


def test_no_go_zone_refuses_real_deploy(moves):
    zone = SafetyPolicy(
        name="zone",
        collision_zones=[{"x_min": 2, "x_max": 4, "y_min": 2, "y_max": 4}],
    )
    t = SimToRealTransfer(SIM, REAL, safety_policy=zone)
    assert t.deploy("go to (3, 3)", agent=Agent(provider="rule")) is False
    assert moves[REAL] == []


# --- graph path (apyrobo plan --simulate --auto-deploy) --------------------

def test_graph_deploy_without_agent_clamps_real_robot_speed(moves):
    graph = _graph("go to (3, 4) at speed 5")
    t = SimToRealTransfer(SIM, REAL)
    sim_result, deployed = t.run(graph, auto_deploy=True)
    assert sim_result.success and deployed
    assert moves[SIM][0]["speed"] == pytest.approx(POLICY_REGISTRY["default"].max_speed)
    assert moves[REAL][0]["speed"] == pytest.approx(POLICY_REGISTRY["default"].max_speed)


def test_graph_deploy_with_agent_clamps_real_robot_speed(moves):
    graph = _graph("go to (3, 4) at speed 5")
    t = SimToRealTransfer(SIM, REAL, safety_policy=SafetyPolicy(name="slow", max_speed=0.25))
    assert t.deploy(graph, agent=Agent(provider="rule")) is True
    assert moves[REAL][0]["speed"] == pytest.approx(0.25)


def test_graph_in_no_go_zone_is_not_deployed(moves):
    graph = _graph("go to (3, 3)")
    zone = SafetyPolicy(
        name="zone",
        collision_zones=[{"x_min": 2, "x_max": 4, "y_min": 2, "y_max": 4}],
    )
    t = SimToRealTransfer(SIM, REAL, safety_policy=zone)
    sim_result, deployed = t.run(graph, auto_deploy=True)
    assert not sim_result.success
    assert not deployed
    assert moves[REAL] == []

"""
examples/04_learned_policy_safety.py runs end to end and shows what it
claims: a nominal episode succeeds, a one-off runaway action is clamped to
one bounded step, and a policy steering into a keep-out zone is rejected by
the SafetyEnforcer with the robot stopped and the rejection audited.
"""
from __future__ import annotations

import importlib.util
import math
from itertools import pairwise
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "04_learned_policy_safety.py"


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("learned_policy_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def results(example):
    return example.main(verbose=False)


def test_nominal_episode_reaches_goal_untouched(example, results):
    r, enforcer = results["nominal"]
    assert r.success and r.stop_reason == "success"
    assert r.clamped_steps == 0
    assert enforcer.violations == []
    assert math.dist(r.final_position, example.GOAL) < 0.05


def test_runaway_spike_is_one_bounded_step(example, results):
    r, enforcer = results["runaway"]
    assert r.success and r.stop_reason == "success"
    assert r.clamped_steps == 1
    # no tick moved further than the runner's 0.5 m bound
    traj = list(r.trajectory) + [r.final_position]
    assert max(math.dist(a, b) for a, b in pairwise(traj)) <= 0.5 + 1e-9
    assert enforcer.violations == []


def test_no_go_zone_fails_closed(example, results):
    r, enforcer = results["no_go"]
    assert not r.success and r.stop_reason == "error"
    assert "SafetyViolation" in (r.error or "")
    assert [v["type"] for v in enforcer.violations] == ["collision_zone"]
    assert any(e.event_type == "violation" for e in enforcer.audit_log)
    # the robot never entered the zone and was stopped on the way out
    x, y = r.final_position
    zone = example.NO_GO_ZONE
    assert not (zone["x_min"] <= x <= zone["x_max"] and zone["y_min"] <= y <= zone["y_max"])
    assert not enforcer.robot._adapter.is_moving


def test_adapter_wraps_lerobot_interface(example):
    inner = example.StandInPolicy(example.GOAL)
    policy = example.LeRobotBasePolicy(inner, example.to_batch, example.from_action)
    policy.reset()
    action = policy.select_action({"x": 0.0, "y": 0.0})
    assert set(action) == {"dx", "dy"}
    assert action["dx"] == pytest.approx(0.4)

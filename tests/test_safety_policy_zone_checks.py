"""A zone the enforcer cannot use is refused, not stored as one that blocks nothing."""

from __future__ import annotations

import math

import pytest

from apyrobo.config import ApyroboConfig
from apyrobo.core.robot import Robot
from apyrobo.safety.enforcer import SafetyEnforcer, SafetyPolicy, SafetyViolation

ZONE = {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2}
NUM = "must be a number"

BAD_ZONES = [
    pytest.param({"x_min": 2, "x_max": 0, "y_min": 0, "y_max": 2}, "min > max", id="x-inverted"),
    pytest.param({"x_min": 0, "x_max": 2, "y_min": 2, "y_max": 0}, "min > max", id="y-inverted"),
    pytest.param({"x_min": math.nan, "x_max": 2, "y_min": 0, "y_max": 2}, NUM, id="nan"),
    pytest.param({"x_min": "0", "x_max": 2, "y_min": 0, "y_max": 2}, NUM, id="string"),
    pytest.param({"x_min": True, "x_max": 2, "y_min": 0, "y_max": 2}, NUM, id="bool"),
    pytest.param([0, 0, 2, 2], "must be a dict", id="list"),
]


def _robot() -> Robot:
    return Robot.discover("mock://zone_checks")


@pytest.mark.parametrize("zone, message", BAD_ZONES)
def test_policy_refuses_bad_collision_zone(zone, message):
    with pytest.raises(ValueError, match=message):
        SafetyPolicy(name="t", collision_zones=[zone])


@pytest.mark.parametrize("zone, message", BAD_ZONES)
def test_policy_refuses_bad_allowed_area(zone, message):
    with pytest.raises(ValueError, match=message):
        SafetyPolicy(name="t", allowed_areas=[zone])


@pytest.mark.parametrize("zone, message", BAD_ZONES)
def test_add_collision_zone_refuses_bad_zone(zone, message):
    enforcer = SafetyEnforcer(_robot(), policy=SafetyPolicy(name="t"))
    with pytest.raises(ValueError, match=message):
        enforcer.add_collision_zone(zone)
    assert enforcer.policy.collision_zones == []


def test_inverted_zone_used_to_let_a_move_in():
    """The case that was broken: an inverted zone must not be accepted silently."""
    enforcer = SafetyEnforcer(_robot(), policy=SafetyPolicy(name="t"))
    with pytest.raises(ValueError):
        enforcer.add_collision_zone({"x_min": 2, "x_max": 0, "y_min": 0, "y_max": 2})
    enforcer.add_collision_zone(ZONE)
    with pytest.raises(SafetyViolation):
        enforcer.move(x=1, y=1, speed=0.2)


@pytest.mark.parametrize(
    "zone",
    [
        ZONE,
        {"x_min": 1, "x_max": 1, "y_min": 1, "y_max": 1},
        {"x_min": 0.5, "x_max": 2.5, "y_min": -1, "y_max": 1, "name": "dock"},
        {"x_min": 0},
        {"x_min": -math.inf, "x_max": math.inf, "y_min": 0, "y_max": 1},
    ],
)
def test_valid_zones_are_kept(zone):
    policy = SafetyPolicy(name="t", collision_zones=[zone], allowed_areas=[zone])
    assert policy.collision_zones == [zone]
    assert policy.allowed_areas == [zone]
    enforcer = SafetyEnforcer(_robot(), policy=SafetyPolicy(name="t"))
    enforcer.add_collision_zone(zone)
    assert enforcer.policy.collision_zones == [zone]


def test_open_sided_zone_still_blocks():
    policy = SafetyPolicy(name="t", collision_zones=[{"x_min": 0.5}])
    enforcer = SafetyEnforcer(_robot(), policy=policy)
    with pytest.raises(SafetyViolation):
        enforcer.move(x=1, y=0, speed=0.2)


def test_config_file_with_inverted_zone_is_refused(tmp_path):
    path = tmp_path / "apyrobo.yaml"
    path.write_text(
        "safety:\n"
        "  collision_zones:\n"
        "    - {x_min: 2, x_max: 0, y_min: 0, y_max: 2}\n"
    )
    config = ApyroboConfig.from_file(path)
    with pytest.raises(ValueError, match="min > max"):
        config.safety_policy()

"""A speed limit written in another unit or other words is still checked.

``apyrobo policy add`` read a speed limit only in m/s and only after a few
lead phrases, so "never exceed 5 km/h", "speed limit 50 cm/s" or "keep speed
below 0.5 m/s" became ``custom`` policies that ``apyrobo policy check`` never
reports as violated.
"""

import pytest

from apyrobo.safety.nl_policy import NLPolicyParser


@pytest.mark.parametrize(
    "rule, limit, as_written",
    [
        ("never exceed 5 km/h", 1.388888, "5 km/h"),
        ("never exceed 5 km/hour", 1.388888, "5 km/hour"),
        ("max speed 2 mph", 0.89408, "2 mph"),
        ("never go faster than 3 miles per hour", 1.34112, "3 miles per hour"),
        ("speed limit: 50 cm/s", 0.5, "50 cm/s"),
        ("never exceed 300 mm/s", 0.3, "300 mm/s"),
        ("top speed 3 ft/s", 0.9144, "3 ft/s"),
        ("never exceed 30 m/min", 0.5, "30 m/min"),
    ],
)
def test_other_units_are_stored_in_metres_per_second(rule, limit, as_written):
    policy = NLPolicyParser().parse(rule)
    assert policy.constraint_type == "speed_limit"
    assert policy.parameters == {
        "max_speed_ms": limit,
        "context": "always",
        "as_written": as_written,
    }


def test_converted_limit_is_never_looser_than_the_rule():
    # 5 km/h is 1.3888... m/s; the stored limit is rounded down, not up.
    limit = NLPolicyParser().parse("never exceed 5 km/h").parameters["max_speed_ms"]
    assert limit <= 5 / 3.6


@pytest.mark.parametrize(
    "rule, limit, context",
    [
        ("keep speed below 0.5 m/s", 0.5, "always"),
        ("keep the speed under 0.5 m/s in the lab", 0.5, "in the lab"),
        ("limit speed to 0.5 m/s", 0.5, "always"),
        ("limit the robot to 0.4 m/s", 0.4, "always"),
        ("cap speed at 0.5 m/s near humans", 0.5, "near humans"),
        ("never go above 0.5 m/s", 0.5, "always"),
        ("do not drive over 0.6 m/s", 0.6, "always"),
        ("maximum velocity 0.5 m/s", 0.5, "always"),
        ("speed <= 0.5 m/s", 0.5, "always"),
        ("slow down to 0.2 m/s near people", 0.2, "near people"),
    ],
)
def test_more_upper_bound_wordings_become_speed_limits(rule, limit, context):
    policy = NLPolicyParser().parse(rule)
    assert policy.constraint_type == "speed_limit"
    assert policy.parameters == {"max_speed_ms": limit, "context": context}


@pytest.mark.parametrize(
    "rule",
    [
        "never go below 2 km/h",  # a minimum, not a maximum
        "keep speed above 0.2 m/s",
        "keep at least 10 cm/s",
        "max speed 2 m/s²",  # an acceleration, not a speed
        "never exceed 5 m",  # a distance
    ],
)
def test_lower_bounds_and_non_speeds_stay_custom(rule):
    assert NLPolicyParser().parse(rule).constraint_type == "custom"


def test_a_km_per_hour_limit_is_checked():
    parser = NLPolicyParser()
    policy = parser.parse("never exceed 5 km/h")
    assert parser.check_compliance("navigate at 2 m/s to the dock", [policy])
    assert parser.check_compliance("navigate at 1 m/s to the dock", [policy]) == []

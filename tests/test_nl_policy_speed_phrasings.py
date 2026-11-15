"""A speed limit written in common words is read as a speed limit.

``apyrobo policy add`` used to read a speed limit only after "never exceed",
"max speed" or "speed limit" and only in ``m/s``; "Speed must stay below 1 m/s"
or "do not go faster than 0.3 meters per second" became ``custom`` policies
that ``apyrobo policy check`` never reports as violated.
"""

import pytest

from apyrobo.safety.nl_policy import NLPolicyParser


@pytest.mark.parametrize(
    "rule, limit, context",
    [
        ("never exceed 0.5 m/s near humans", 0.5, "near humans"),
        ("max speed 0.5 m/s near humans", 0.5, "near humans"),
        ("speed limit 1 m/s", 1.0, "always"),
        ("Speed must stay below 1 m/s", 1.0, "always"),
        ("keep under 0.7 m/s in the warehouse", 0.7, "in the warehouse"),
        ("Robots must not go faster than 0.3 meters per second", 0.3, "always"),
        ("do not exceed 0.8 mps in the lab", 0.8, "in the lab"),
        ("maximum speed of .5 m/s", 0.5, "always"),
        ("top speed: 1.2 m/s", 1.2, "always"),
        ("no more than 0.4 m/s", 0.4, "always"),
        ("never drive faster than 2 metres per second", 2.0, "always"),
    ],
)
def test_upper_bound_rules_become_speed_limits(rule, limit, context):
    policy = NLPolicyParser().parse(rule)
    assert policy.constraint_type == "speed_limit"
    assert policy.parameters == {"max_speed_ms": limit, "context": context}


@pytest.mark.parametrize(
    "rule",
    [
        "never go below 0.2 m/s",  # a minimum, not a maximum
        "keep at least 0.2 m/s",
        "never exceed 30 cm/s",  # not m/s: stays custom, as documented
        "never exceed 2 km/h",
    ],
)
def test_lower_bounds_and_other_units_stay_custom(rule):
    assert NLPolicyParser().parse(rule).constraint_type == "custom"


def test_a_reworded_limit_is_checked():
    parser = NLPolicyParser()
    policy = parser.parse("Speed must stay below 1 m/s")
    assert parser.check_compliance("navigate at 2 m/s to the dock", [policy])
    assert parser.check_compliance("navigate at 0.5 m/s to the dock", [policy]) == []


def test_proximity_rule_is_not_read_as_a_speed():
    policy = NLPolicyParser().parse("keep 1.5 meters away from humans")
    assert policy.constraint_type == "proximity_limit"

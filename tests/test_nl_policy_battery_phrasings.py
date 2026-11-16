"""A battery reserve written in common words is read as a battery reserve.

``apyrobo policy add`` used to read a battery reserve only after "always
keep", "maintain" or "reserve"; "keep 20% battery", "never go below 20%
battery" or "battery must stay above 20%" became ``custom`` policies that
``apyrobo policy check`` never reports as violated. The check also read a
battery level only when written as "15% battery".
"""

import pytest

from apyrobo.safety.nl_policy import CHECKED_CONSTRAINT_TYPES, NLPolicyParser


@pytest.mark.parametrize(
    "rule, pct",
    [
        ("always keep 20% battery", 20.0),
        ("reserve at least 15% battery", 15.0),
        ("keep 30% battery", 30.0),
        ("keep at least 20% battery", 20.0),
        ("keep 20 % of the battery", 20.0),
        ("maintain a 25% battery reserve", 25.0),
        ("never go below 20% battery", 20.0),
        ("Do not drop under 10% battery.", 10.0),
        ("never let the battery fall below 20%", 20.0),
        ("Never let battery level drop under 15%", 15.0),
        ("battery must stay above 20%", 20.0),
        ("Battery should always remain above 30%", 30.0),
        ("keep the battery above 25%", 25.0),
        ("minimum battery 20%", 20.0),
        ("min battery level: 12.5%", 12.5),
    ],
)
def test_battery_phrasings_become_battery_reserves(rule, pct):
    policy = NLPolicyParser().parse(rule)
    assert policy.constraint_type == "battery_reserve"
    assert policy.constraint_type in CHECKED_CONSTRAINT_TYPES
    assert policy.parameters == {"min_battery_pct": pct}


@pytest.mark.parametrize(
    "rule",
    [
        "warn when battery drops below 20%",  # a warning, not a reserve
        "keep 30% charge",  # no battery word
        "never go below 0.2 m/s",  # a lower speed bound
        "never exceed 0.5 m/s with less than 20% battery",  # a speed limit
    ],
)
def test_other_rules_are_not_battery_reserves(rule):
    assert NLPolicyParser().parse(rule).constraint_type != "battery_reserve"


@pytest.mark.parametrize(
    "action",
    [
        "go to (1, 1) with 15% battery",
        "go to (1, 1), battery 15%",
        "go to (1, 1) at battery 15%",
        "go to (1, 1) with battery at 15%",
        "battery level: 15%, go to (1, 1)",
        "battery 50%, then 15% battery",
    ],
)
def test_check_reads_battery_levels(action):
    parser = NLPolicyParser()
    policy = parser.parse("never go below 20% battery")
    violations = parser.check_compliance(action, [policy])
    assert len(violations) == 1
    assert "battery 15.0% below reserve 20.0%" in violations[0]


@pytest.mark.parametrize(
    "action", ["go to (1, 1) with 50% battery", "go to (1, 1), battery at 20%", "go to (1, 1)"]
)
def test_check_passes_enough_or_no_battery(action):
    parser = NLPolicyParser()
    policy = parser.parse("keep the battery above 20%")
    assert parser.check_compliance(action, [policy]) == []

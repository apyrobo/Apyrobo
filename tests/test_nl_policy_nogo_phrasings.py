"""A no-go zone written in common words is read as a no-go zone.

``apyrobo policy add`` used to read a no-go zone only after "never enter",
"do not enter" or "avoid"; "keep out of the lab", "stay out of the loading
dock" or "don't go into the kitchen" became ``custom`` policies that
``apyrobo policy check`` never reports as violated.
"""

import pytest

from apyrobo.safety.nl_policy import CHECKED_CONSTRAINT_TYPES, NLPolicyParser


@pytest.mark.parametrize(
    "rule, zone",
    [
        ("never enter the server room", "server room"),
        ("do not enter zone B", "zone B"),
        ("avoid the kitchen", "kitchen"),
        ("keep out of the lab", "lab"),
        ("Stay out of the loading dock.", "loading dock"),
        ("Robots must keep out of the paint shop", "paint shop"),
        ("don't enter the cold store", "cold store"),
        ("must not enter the clean room", "clean room"),
        ("never go into the kitchen", "kitchen"),
        ("Do not go into the server room", "server room"),
    ],
)
def test_no_go_phrasings_become_no_go_zones(rule, zone):
    policy = NLPolicyParser().parse(rule)
    assert policy.constraint_type == "no_go_zone"
    assert policy.constraint_type in CHECKED_CONSTRAINT_TYPES
    assert policy.parameters == {"zone_name": zone}


@pytest.mark.parametrize(
    "rule, action",
    [
        ("keep out of the lab", "navigate to the lab"),
        ("stay out of the loading dock", "drive through the Loading Dock"),
        ("don't go into the kitchen", "go to the kitchen"),
    ],
)
def test_policy_check_reports_the_zone(rule, action):
    parser = NLPolicyParser()
    violations = parser.check_compliance(action, [parser.parse(rule)])
    assert len(violations) == 1
    assert "no-go zone" in violations[0]


def test_other_actions_pass():
    parser = NLPolicyParser()
    policy = parser.parse("keep out of the lab")
    assert parser.check_compliance("navigate to the dock", [policy]) == []


@pytest.mark.parametrize(
    "rule, expected",
    [
        # Rules with a number keep their own type.
        ("keep 1.5 meters away from humans", "proximity_limit"),
        ("stay below 0.5 m/s", "speed_limit"),
        ("always keep 20% battery", "battery_reserve"),
        # Still not a zone: no way in is named.
        ("no robots in the lab", "custom"),
    ],
)
def test_other_rules_keep_their_type(rule, expected):
    assert NLPolicyParser().parse(rule).constraint_type == expected

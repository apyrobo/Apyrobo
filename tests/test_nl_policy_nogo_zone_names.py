"""A no-go rule's zone name stops where the zone does.

``apyrobo policy add "never enter the kitchen during lunch"`` stored the zone
as "kitchen during lunch", so ``apyrobo policy check "go to the kitchen"``
passed. The same happened with "the kitchen or the lab" (one zone named
"kitchen or the lab") and "stay out of the server room!" (a zone ending in
"!"). The zone is now the name alone, a condition is kept as ``context`` and
not checked (the zone is refused at all times), and several zones in one rule
are each checked.
"""

import argparse

import pytest

from apyrobo.cli import cmd_policy
from apyrobo.safety.nl_policy import NLPolicyParser, NLSafetyPolicy


@pytest.mark.parametrize(
    "rule, zone, context",
    [
        ("never enter the kitchen during lunch", "kitchen", "during lunch"),
        ("do not enter the lab without an escort", "lab", "without an escort"),
        ("avoid the loading dock at night", "loading dock", "at night"),
        ("keep out of the lab, unless escorted.", "lab", "unless escorted"),
        (
            "never enter the warehouse floor when humans are present",
            "warehouse floor",
            "when humans are present",
        ),
        ("stay out of the paint shop while it is drying", "paint shop", "while it is drying"),
        ("never enter the cold store on weekends", "cold store", "on weekends"),
    ],
)
def test_condition_is_kept_apart_from_the_zone(rule, zone, context):
    parser = NLPolicyParser()
    policy = parser.parse(rule)
    assert policy.constraint_type == "no_go_zone"
    assert policy.parameters == {"zone_name": zone, "context": context}
    violations = parser.check_compliance(f"navigate to the {zone}", [policy])
    assert len(violations) == 1
    assert repr(zone) in violations[0]


@pytest.mark.parametrize("rule", ["stay out of the server room!", "avoid the server room?"])
def test_end_punctuation_is_not_part_of_the_zone(rule):
    parser = NLPolicyParser()
    policy = parser.parse(rule)
    assert policy.parameters == {"zone_name": "server room"}
    assert parser.check_compliance("go to the server room", [policy])


@pytest.mark.parametrize(
    "rule, zones",
    [
        ("never enter the kitchen or the lab", ["kitchen", "lab"]),
        ("do not enter the kitchen and the lab", ["kitchen", "lab"]),
        ("never enter the kitchen, the lab and the dock", ["kitchen", "lab", "dock"]),
        ("avoid the kitchen, the lab, or the dock", ["kitchen", "lab", "dock"]),
        ("keep out of zone A or zone B at night", ["zone A", "zone B"]),
    ],
)
def test_every_zone_in_one_rule_is_checked(rule, zones):
    parser = NLPolicyParser()
    policy = parser.parse(rule)
    assert policy.parameters["zone_name"] == zones[0]
    assert policy.parameters["zone_names"] == zones
    for zone in zones:
        violations = parser.check_compliance(f"go to the {zone}", [policy])
        assert len(violations) == 1, zone
    assert parser.check_compliance("go to the hallway", [policy]) == []


def test_plain_rules_keep_their_parameters():
    policy = NLPolicyParser().parse("never enter the server room")
    assert policy.parameters == {"zone_name": "server room"}


def test_stored_single_zone_policies_still_check():
    # A policy stored before zone_names existed has only zone_name.
    policy = NLSafetyPolicy(constraint_type="no_go_zone", parameters={"zone_name": "lab"})
    parser = NLPolicyParser()
    assert parser.check_compliance("go to the lab", [policy])
    assert parser.check_compliance("go to the dock", [policy]) == []


def _args(db, description):
    return argparse.Namespace(
        policy_command="add", db=db, json=False, description=description, severity="hard"
    )


def test_policy_add_says_the_condition_is_not_checked(tmp_path, capsys):
    cmd_policy(_args(str(tmp_path / "p.db"), "never enter the kitchen during lunch"))
    out = capsys.readouterr()
    assert "no_go_zone" in out.out
    assert "'during lunch' is not checked" in out.err
    assert "at all times" in out.err


def test_policy_add_plain_zone_prints_no_note(tmp_path, capsys):
    cmd_policy(_args(str(tmp_path / "p.db"), "never enter the kitchen"))
    assert capsys.readouterr().err == ""

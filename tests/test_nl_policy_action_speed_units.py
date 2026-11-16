"""`apyrobo policy check` reads an action's speed in the units a rule may use.

Before this fix the check read an action's speed in m/s only: 'drive at 30
cm/s' against a 0.5 m/s limit was reported as unreadable, and 'drive at 7
km/h' could not be compared with the limit at all. A speed in km/h, mph,
cm/s, mm/s, ft/s or m/min is now converted to m/s, rounded up so it is never
understated. Units the check does not know still fail closed.
"""
from __future__ import annotations

import argparse
import json

import pytest

from apyrobo.safety.nl_policy import NLPolicyParser, NLSafetyPolicy


def _check(action: str, max_speed: float = 0.5) -> list[str]:
    policy = NLSafetyPolicy(
        description=f"never exceed {max_speed} m/s",
        constraint_type="speed_limit",
        parameters={"max_speed_ms": max_speed},
    )
    return NLPolicyParser().check_compliance(action, [policy])


@pytest.mark.parametrize(
    "action, speed, written",
    [
        ("drive at 7 km/h", "1.944445", "7 km/h"),
        ("drive at 2 mph", "0.89408", "2 mph"),
        ("drive at 51 cm/s", "0.51", "51 cm/s"),
        ("drive at 501 mm/s", "0.501", "501 mm/s"),
        ("drive at 1.7 ft/s", "0.51816", "1.7 ft/s"),
        ("drive at 31 m/min", "0.516667", "31 m/min"),
        ("drive at 2 kilometres per hour", "0.555556", "2 kilometres per hour"),
        ("drive at 0.4 m/s, then 3 km/h", "0.833334", "3 km/h"),
        ("at a speed of 3 kph", "0.833334", "3 kph"),
    ],
)
def test_speed_in_another_unit_over_the_limit_is_a_violation(
    action: str, speed: str, written: str
) -> None:
    violations = _check(action)
    assert len(violations) == 1
    assert f"speed {speed} m/s ({written}) exceeds limit 0.5 m/s" in violations[0]


@pytest.mark.parametrize(
    "action",
    [
        "drive at 30 cm/s",
        "drive at 500 mm/s",
        "drive at 1.8 km/h",
        "drive at 1 mph",
        "drive at 1.6 ft/s",
        "drive at 30 m/min",
        "drive at 30 meters per minute",
        "at speed 1 km/h",
    ],
)
def test_speed_in_another_unit_within_the_limit_passes(action: str) -> None:
    assert _check(action) == []


def test_rounding_up_never_understates_the_speed() -> None:
    # 1.8001 km/h is 0.50002777... m/s: just over a 0.5 m/s limit.
    assert _check("drive at 1.8001 km/h")


def test_largest_speed_counts_whatever_the_unit() -> None:
    violations = _check("drive at 0.6 m/s then 1 km/h")
    assert len(violations) == 1
    assert "speed 0.6 m/s exceeds limit" in violations[0]


@pytest.mark.parametrize(
    "action",
    [
        "drive at 2 knots",
        "drive at 2 in/s",
        "drive at 2 m/h",
        "drive at 3 km/h and 2 knots",
        "drive at 30 cm/s at speed 2",
    ],
)
def test_unknown_unit_still_fails_closed(action: str) -> None:
    violations = _check(action)
    assert len(violations) == 1
    assert "cannot be checked" in violations[0]


def test_km_per_hour_rule_against_km_per_hour_action() -> None:
    parser = NLPolicyParser()
    policy = parser.parse("never exceed 5 km/h")
    assert parser.check_compliance("drive at 4 km/h", [policy]) == []
    violations = parser.check_compliance("drive at 6 km/h", [policy])
    assert len(violations) == 1
    assert "(6 km/h) exceeds limit" in violations[0]


def test_cli_check_reads_cm_per_second(tmp_path, capsys) -> None:
    from apyrobo.cli import cmd_policy

    db = str(tmp_path / "p.db")
    base = {"db": db, "json": True, "description": "", "severity": "hard",
            "policy_id": "", "action": "", "all": False}
    cmd_policy(argparse.Namespace(**{**base, "policy_command": "add",
                                     "description": "never exceed 0.5 m/s"}))
    capsys.readouterr()
    cmd_policy(argparse.Namespace(**{**base, "policy_command": "check",
                                     "action": "navigate at 30 cm/s"}))
    assert json.loads(capsys.readouterr().out)["violations"] == []
    with pytest.raises(SystemExit) as exc:
        cmd_policy(argparse.Namespace(**{**base, "policy_command": "check",
                                         "action": "navigate at 7 km/h"}))
    assert exc.value.code == 1

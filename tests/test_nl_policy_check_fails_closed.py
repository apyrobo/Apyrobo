"""`apyrobo policy check` must not pass an action whose speed it cannot read.

docs/concepts/nl_safety_policies.md says the check "exits 1 with violation
message — safe to use in CI or pre-flight scripts". Before this fix a 0.5 m/s
policy passed "at speed 2", "2 meters per second", "7 km/h", the second speed
in "0.4 m/s then 2 m/s", and every violation when --json was given.
"""
from __future__ import annotations

import argparse
import json

import pytest

from apyrobo.safety.nl_policy import NLPolicyParser, NLPolicyStore, NLSafetyPolicy


def _limit(max_speed: float = 0.5) -> list[NLSafetyPolicy]:
    return [
        NLSafetyPolicy(
            description=f"never exceed {max_speed} m/s",
            constraint_type="speed_limit",
            parameters={"max_speed_ms": max_speed},
        )
    ]


def _check(action: str) -> list[str]:
    return NLPolicyParser().check_compliance(action, _limit())


@pytest.mark.parametrize(
    "action",
    [
        "go to (3, 4) at speed 2",
        "at speed: 2",
        "drive at 7 km/h",
        "drive at 2 mph",
        "drive at 30 cm/s",
        "drive at 2 m/min",
        "drive at 2 ft per second",
        "drive at 3 per second",
        "drive at 0,3 m/s",
        "drive at 1e3 m/s",
        "drive at 5.0.9 m/s",
    ],
)
def test_unreadable_speed_is_reported(action: str) -> None:
    violations = _check(action)
    assert len(violations) == 1
    assert "cannot be checked" in violations[0]


@pytest.mark.parametrize(
    "action, speed",
    [
        ("drive at 2 meters per second", "2.0"),
        ("drive at 2 metres/s", "2.0"),
        ("at 0.4 m/s then 2 m/s", "2.0"),
        ("drive at .9 m/s", "0.9"),
        ("drive at 2.0m/s", "2.0"),
    ],
)
def test_speed_over_limit_is_read(action: str, speed: str) -> None:
    violations = _check(action)
    assert len(violations) == 1
    assert f"speed {speed} m/s exceeds limit 0.5 m/s" in violations[0]


@pytest.mark.parametrize(
    "action",
    [
        "navigate at 0.3 m/s",
        "drive at 0.3 meters per second",
        "at speed of 0.3 m/s",
        "go to (1.5, 2) at 0.3 m/s",
        "navigate to the dock",
        "go 5 meters forward",
        "wait 3 seconds",
    ],
)
def test_compliant_or_speedless_action_passes(action: str) -> None:
    assert _check(action) == []


def test_no_speed_policy_means_no_speed_violation() -> None:
    assert NLPolicyParser().check_compliance("drive at 7 km/h", []) == []


def _args(sub: str, db: str, **kw) -> argparse.Namespace:
    base = {"policy_command": sub, "db": db, "json": False, "description": "",
            "severity": "hard", "policy_id": "", "action": "", "all": False}
    base.update(kw)
    return argparse.Namespace(**base)


def test_cli_json_check_exits_1_on_violation(tmp_path, capsys) -> None:
    from apyrobo.cli import cmd_policy

    db = str(tmp_path / "p.db")
    cmd_policy(_args("add", db, description="never exceed 0.5 m/s"))
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        cmd_policy(_args("check", db, action="navigate at 2 m/s", json=True))
    assert exc.value.code == 1
    assert len(json.loads(capsys.readouterr().out)["violations"]) == 1


def test_cli_json_check_exits_0_when_compliant(tmp_path, capsys) -> None:
    from apyrobo.cli import cmd_policy

    db = str(tmp_path / "p.db")
    cmd_policy(_args("add", db, description="never exceed 0.5 m/s"))
    capsys.readouterr()
    cmd_policy(_args("check", db, action="navigate at 0.3 m/s", json=True))
    assert json.loads(capsys.readouterr().out)["violations"] == []


def test_cli_check_exits_1_on_unreadable_speed(tmp_path) -> None:
    from apyrobo.cli import cmd_policy

    db = str(tmp_path / "p.db")
    cmd_policy(_args("add", db, description="never exceed 0.5 m/s"))
    with pytest.raises(SystemExit) as exc:
        cmd_policy(_args("check", db, action="go to (3, 4) at speed 2"))
    assert exc.value.code == 1


def test_store_expands_home_and_creates_folder(tmp_path, monkeypatch) -> None:
    # The docs open NLPolicyStore("~/.apyrobo/policies.db"); that raised
    # "unable to open database file" before.
    monkeypatch.setenv("HOME", str(tmp_path))
    store = NLPolicyStore("~/.apyrobo/policies.db")
    store.add(NLPolicyParser().parse("never exceed 0.5 m/s"))
    store.close()
    assert (tmp_path / ".apyrobo" / "policies.db").is_file()

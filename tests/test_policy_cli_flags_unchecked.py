"""`apyrobo policy` must say when a stored rule will never be checked.

`apyrobo policy check` reads speed limits in m/s, battery reserves and no-go
zones only (docs/concepts/nl_safety_policies.md). Before this fix,
`apyrobo policy add "never exceed 30 cm/s"` or "keep out of the lab" printed
"✓ Added policy" as a custom rule, and every later check printed
"✓ No policy violations" with nothing to say that rule was skipped.
"""
from __future__ import annotations

import argparse
import json

import pytest

from apyrobo.cli import cmd_policy
from apyrobo.safety.nl_policy import CHECKED_CONSTRAINT_TYPES, NLPolicyParser


def _args(sub: str, db: str, **kw) -> argparse.Namespace:
    base = {"policy_command": sub, "db": db, "json": False, "description": "",
            "severity": "hard", "policy_id": "", "action": "", "all": False}
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.mark.parametrize(
    "rule", ["never exceed 30 cm/s", "keep out of the lab", "keep 30% battery"]
)
def test_add_warns_when_rule_is_not_checked(tmp_path, capsys, rule) -> None:
    cmd_policy(_args("add", str(tmp_path / "p.db"), description=rule))
    out = capsys.readouterr()
    assert "✓ Added policy" in out.out
    assert "Not checked" in out.err
    assert "apyrobo policy remove" in out.err


def test_add_is_quiet_for_a_checked_rule(tmp_path, capsys) -> None:
    cmd_policy(_args("add", str(tmp_path / "p.db"), description="never exceed 0.5 m/s"))
    assert capsys.readouterr().err == ""


def test_add_json_says_whether_checked(tmp_path, capsys) -> None:
    db = str(tmp_path / "p.db")
    cmd_policy(_args("add", db, description="never exceed 0.5 m/s", json=True))
    assert json.loads(capsys.readouterr().out)["checked"] is True
    cmd_policy(_args("add", db, description="keep out of the lab", json=True))
    out = capsys.readouterr()
    data = json.loads(out.out)
    assert data["checked"] is False
    assert data["constraint_type"] == "custom"
    assert "Not checked" in out.err


def test_list_marks_unchecked(tmp_path, capsys) -> None:
    db = str(tmp_path / "p.db")
    cmd_policy(_args("add", db, description="never exceed 0.5 m/s"))
    cmd_policy(_args("add", db, description="keep out of the lab"))
    capsys.readouterr()
    cmd_policy(_args("list", db))
    lines = capsys.readouterr().out.splitlines()
    speed = [ln for ln in lines if "speed_limit" in ln]
    custom = [ln for ln in lines if "keep out of the lab" in ln]
    assert speed and "[not checked]" not in speed[0]
    assert custom and "[not checked]" in custom[0]


def test_check_names_skipped_policies_when_passing(tmp_path, capsys) -> None:
    db = str(tmp_path / "p.db")
    cmd_policy(_args("add", db, description="keep out of the lab"))
    capsys.readouterr()
    cmd_policy(_args("check", db, action="go to the lab"))
    out = capsys.readouterr().out
    assert "✓ No policy violations" in out
    assert "1 active policy was not checked" in out
    assert "keep out of the lab" in out


def test_check_names_skipped_policies_on_violation(tmp_path, capsys) -> None:
    db = str(tmp_path / "p.db")
    cmd_policy(_args("add", db, description="never exceed 0.5 m/s"))
    cmd_policy(_args("add", db, description="keep out of the lab"))
    cmd_policy(_args("add", db, description="keep 30% battery"))
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        cmd_policy(_args("check", db, action="go to the lab at 2 m/s"))
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "1 violation(s)" in out
    assert "2 active policies were not checked" in out


def test_check_json_lists_not_checked_ids(tmp_path, capsys) -> None:
    db = str(tmp_path / "p.db")
    cmd_policy(_args("add", db, description="keep out of the lab", json=True))
    pid = json.loads(capsys.readouterr().out)["policy_id"]
    cmd_policy(_args("check", db, action="go to the lab", json=True))
    data = json.loads(capsys.readouterr().out)
    assert data["violations"] == []
    assert data["not_checked"] == [pid]


def test_check_without_unchecked_policies_prints_no_note(tmp_path, capsys) -> None:
    db = str(tmp_path / "p.db")
    cmd_policy(_args("add", db, description="never exceed 0.5 m/s"))
    capsys.readouterr()
    cmd_policy(_args("check", db, action="navigate at 0.3 m/s"))
    assert "not checked" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "rule",
    ["never exceed 0.5 m/s", "always keep 20% battery", "never enter the server room"],
)
def test_checked_types_match_what_check_compliance_reads(rule) -> None:
    # Each documented checked form parses to a type in the set, and a
    # matching action is reported, so the set and the checker agree.
    parser = NLPolicyParser()
    policy = parser.parse(rule)
    assert policy.constraint_type in CHECKED_CONSTRAINT_TYPES
    action = {
        "speed_limit": "navigate at 2 m/s",
        "battery_reserve": "go at 5% battery",
        "no_go_zone": "go to the server room",
    }[policy.constraint_type]
    assert parser.check_compliance(action, [policy])

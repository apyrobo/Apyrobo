"""Tests for natural language safety policies (v7.0.0)."""
from __future__ import annotations

import argparse
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from apyrobo.safety.nl_policy import (
    NLSafetyPolicy,
    NLPolicyParser,
    NLPolicyStore,
)


# ---------------------------------------------------------------------------
# NLSafetyPolicy
# ---------------------------------------------------------------------------

class TestNLSafetyPolicy:
    def _make(self, **kwargs):
        defaults = dict(
            description="never exceed 0.5 m/s",
            constraint_type="speed_limit",
            parameters={"max_speed_ms": 0.5},
        )
        defaults.update(kwargs)
        return NLSafetyPolicy(**defaults)

    def test_to_dict_roundtrip(self):
        p = self._make()
        d = p.to_dict()
        p2 = NLSafetyPolicy.from_dict(d)
        assert p2.policy_id == p.policy_id
        assert p2.constraint_type == p.constraint_type
        assert p2.parameters == p.parameters

    def test_summary_contains_type(self):
        p = self._make()
        s = p.summary()
        assert "speed_limit" in s
        assert p.policy_id in s

    def test_auto_policy_id(self):
        p1 = NLSafetyPolicy(description="a")
        p2 = NLSafetyPolicy(description="b")
        assert p1.policy_id != p2.policy_id

    def test_default_active_true(self):
        p = NLSafetyPolicy(description="x")
        assert p.active is True

    def test_default_severity_hard(self):
        p = NLSafetyPolicy(description="x")
        assert p.severity == "hard"


# ---------------------------------------------------------------------------
# NLPolicyParser — regex patterns
# ---------------------------------------------------------------------------

class TestNLPolicyParserRegex:
    def setup_method(self):
        self.parser = NLPolicyParser()

    def test_speed_limit_basic(self):
        p = self.parser.parse("never exceed 0.5 m/s near humans")
        assert p.constraint_type == "speed_limit"
        assert p.parameters["max_speed_ms"] == 0.5

    def test_speed_limit_integer(self):
        p = self.parser.parse("maximum speed 1 m/s")
        assert p.constraint_type == "speed_limit"
        assert p.parameters["max_speed_ms"] == 1.0

    def test_speed_limit_context_captured(self):
        p = self.parser.parse("never exceed 0.5 m/s near humans")
        assert "near humans" in p.parameters.get("context", "")

    def test_proximity_limit(self):
        p = self.parser.parse("stay 2.0 meters away from humans")
        assert p.constraint_type == "proximity_limit"
        assert p.parameters["min_distance_m"] == 2.0
        assert "human" in p.parameters["target"].lower()

    def test_proximity_limit_keep(self):
        p = self.parser.parse("keep at least 1.5m away from obstacles")
        assert p.constraint_type == "proximity_limit"
        assert p.parameters["min_distance_m"] == 1.5

    def test_battery_reserve(self):
        p = self.parser.parse("always keep 20% battery")
        assert p.constraint_type == "battery_reserve"
        assert p.parameters["min_battery_pct"] == 20.0

    def test_battery_reserve_reserve_keyword(self):
        p = self.parser.parse("reserve at least 15% battery")
        assert p.constraint_type == "battery_reserve"
        assert p.parameters["min_battery_pct"] == 15.0

    def test_no_go_zone(self):
        p = self.parser.parse("never enter the server room")
        assert p.constraint_type == "no_go_zone"
        assert "server room" in p.parameters["zone_name"].lower()

    def test_no_go_zone_avoid(self):
        p = self.parser.parse("avoid the warehouse loading dock")
        assert p.constraint_type == "no_go_zone"
        assert "warehouse" in p.parameters["zone_name"].lower()

    def test_unknown_rule_fallback_custom(self):
        p = self.parser.parse("only operate when the lights are on")
        assert p.constraint_type == "custom"
        assert p.parameters.get("raw") is not None

    def test_severity_passed_through(self):
        parser = NLPolicyParser(severity="soft")
        p = parser.parse("never exceed 1.0 m/s")
        assert p.severity == "soft"

    def test_source_is_regex(self):
        p = self.parser.parse("never exceed 0.5 m/s")
        assert p.source == "regex"


# ---------------------------------------------------------------------------
# NLPolicyParser — LLM fallback
# ---------------------------------------------------------------------------

class TestNLPolicyParserLLM:
    def test_llm_not_called_when_regex_matches(self):
        agent = MagicMock()
        parser = NLPolicyParser(agent=agent)
        parser.parse("never exceed 0.5 m/s")
        agent.complete.assert_not_called()

    def test_llm_called_for_unknown_rule(self):
        agent = MagicMock()
        agent.complete.return_value = (
            '{"constraint_type": "time_window", "parameters": {"start": "09:00", "end": "18:00"}}'
        )
        parser = NLPolicyParser(agent=agent)
        p = parser.parse("only operate between 9am and 6pm")
        assert p.constraint_type == "time_window"
        assert p.source == "llm"

    def test_llm_error_falls_back_to_custom(self):
        agent = MagicMock()
        agent.complete.side_effect = RuntimeError("LLM unavailable")
        parser = NLPolicyParser(agent=agent)
        p = parser.parse("only operate when safe")
        assert p.constraint_type == "custom"

    def test_llm_bad_json_falls_back_to_custom(self):
        agent = MagicMock()
        agent.complete.return_value = "not valid json at all"
        parser = NLPolicyParser(agent=agent)
        p = parser.parse("some weird rule")
        assert p.constraint_type == "custom"


# ---------------------------------------------------------------------------
# NLPolicyParser.check_compliance
# ---------------------------------------------------------------------------

class TestCheckCompliance:
    def setup_method(self):
        self.parser = NLPolicyParser()

    def _speed_policy(self, max_speed: float) -> NLSafetyPolicy:
        return NLSafetyPolicy(
            description=f"never exceed {max_speed} m/s",
            constraint_type="speed_limit",
            parameters={"max_speed_ms": max_speed},
            severity="hard",
        )

    def _battery_policy(self, min_pct: float) -> NLSafetyPolicy:
        return NLSafetyPolicy(
            description=f"keep {min_pct}% battery",
            constraint_type="battery_reserve",
            parameters={"min_battery_pct": min_pct},
        )

    def _nogo_policy(self, zone: str) -> NLSafetyPolicy:
        return NLSafetyPolicy(
            description=f"never enter {zone}",
            constraint_type="no_go_zone",
            parameters={"zone_name": zone},
        )

    def test_no_violations(self):
        violations = self.parser.check_compliance(
            "navigate at 0.3 m/s",
            [self._speed_policy(0.5)],
        )
        assert violations == []

    def test_speed_violation(self):
        violations = self.parser.check_compliance(
            "navigate at 2.0 m/s",
            [self._speed_policy(0.5)],
        )
        assert len(violations) == 1
        assert "2.0" in violations[0]

    def test_multiple_policies_one_violation(self):
        policies = [self._speed_policy(0.5), self._battery_policy(20.0)]
        violations = self.parser.check_compliance("navigate at 1.5 m/s", policies)
        assert len(violations) == 1

    def test_no_go_zone_violation(self):
        violations = self.parser.check_compliance(
            "navigate to the server room",
            [self._nogo_policy("server room")],
        )
        assert len(violations) == 1
        assert "server room" in violations[0]

    def test_inactive_policy_not_checked(self):
        policy = self._speed_policy(0.5)
        policy.active = False
        violations = self.parser.check_compliance("navigate at 2.0 m/s", [policy])
        assert violations == []

    def test_no_speed_in_action_skips_speed_check(self):
        violations = self.parser.check_compliance(
            "navigate to the dock",
            [self._speed_policy(0.5)],
        )
        assert violations == []


# ---------------------------------------------------------------------------
# NLPolicyStore
# ---------------------------------------------------------------------------

class TestNLPolicyStore:
    def _store(self):
        return NLPolicyStore(":memory:")

    def _policy(self, desc="never exceed 0.5 m/s"):
        return NLSafetyPolicy(
            description=desc,
            constraint_type="speed_limit",
            parameters={"max_speed_ms": 0.5},
        )

    def test_add_and_retrieve(self):
        store = self._store()
        p = self._policy()
        store.add(p)
        retrieved = store.get(p.policy_id)
        assert retrieved is not None
        assert retrieved.description == p.description
        assert retrieved.parameters == p.parameters

    def test_get_active_policies(self):
        store = self._store()
        store.add(self._policy("rule 1"))
        store.add(self._policy("rule 2"))
        policies = store.get_active_policies()
        assert len(policies) == 2

    def test_count(self):
        store = self._store()
        store.add(self._policy())
        assert store.count() == 1

    def test_remove(self):
        store = self._store()
        p = self._policy()
        store.add(p)
        result = store.remove(p.policy_id)
        assert result is True
        assert store.get(p.policy_id) is None
        assert store.count() == 0

    def test_remove_nonexistent_returns_false(self):
        store = self._store()
        assert store.remove("nonexistent-id") is False

    def test_deactivate(self):
        store = self._store()
        p = self._policy()
        store.add(p)
        store.deactivate(p.policy_id)
        active = store.get_active_policies()
        assert len(active) == 0
        all_policies = store.get_all_policies()
        assert len(all_policies) == 1
        assert not all_policies[0].active

    def test_add_replaces_on_duplicate_id(self):
        store = self._store()
        p = self._policy()
        store.add(p)
        p.parameters = {"max_speed_ms": 1.0}
        store.add(p)  # same policy_id
        retrieved = store.get(p.policy_id)
        assert retrieved.parameters["max_speed_ms"] == 1.0

    def test_get_returns_none_when_not_found(self):
        store = self._store()
        assert store.get("missing-id") is None

    def test_parameters_roundtrip(self):
        store = self._store()
        p = NLSafetyPolicy(
            description="test",
            constraint_type="custom",
            parameters={"key": "value", "num": 42, "nested": {"a": 1}},
        )
        store.add(p)
        retrieved = store.get(p.policy_id)
        assert retrieved.parameters == p.parameters


# ---------------------------------------------------------------------------
# CLI — apyrobo policy add/list/remove/check
# ---------------------------------------------------------------------------

def _policy_args(sub, **kwargs) -> argparse.Namespace:
    defaults = {
        "policy_command": sub,
        "db": ":memory:",
        "json": False,
        "description": "",
        "severity": "hard",
        "policy_id": "",
        "action": "",
        "all": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestCmdPolicyAdd:
    def test_add_speed_limit(self, capsys):
        from apyrobo.cli import cmd_policy
        args = _policy_args("add", description="never exceed 0.5 m/s near humans")
        with patch("apyrobo.safety.nl_policy.NLPolicyStore") as MockStore:
            mock_store = MagicMock()
            MockStore.return_value = mock_store
            mock_store.add = MagicMock()
            from apyrobo.safety.nl_policy import NLSafetyPolicy
            mock_policy = NLSafetyPolicy(
                description="never exceed 0.5 m/s near humans",
                constraint_type="speed_limit",
                parameters={"max_speed_ms": 0.5},
            )
            with patch("apyrobo.safety.nl_policy.NLPolicyParser.parse", return_value=mock_policy):
                cmd_policy(args)
        out = capsys.readouterr().out
        assert "Added" in out or "speed_limit" in out

    def test_add_json_output(self, capsys):
        from apyrobo.cli import cmd_policy
        args = _policy_args("add", description="never exceed 0.5 m/s", json=True)
        # Use a real NLPolicyStore in-memory
        cmd_policy(args)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["constraint_type"] == "speed_limit"
        assert data["parameters"]["max_speed_ms"] == 0.5


class TestCmdPolicyList:
    def test_list_empty(self, capsys):
        from apyrobo.cli import cmd_policy
        args = _policy_args("list")
        cmd_policy(args)
        out = capsys.readouterr().out
        assert "No safety policies" in out

    def test_list_shows_added_policy(self, capsys, tmp_path):
        from apyrobo.cli import cmd_policy
        db = str(tmp_path / "policies.db")
        # Add a policy first
        add_args = _policy_args("add", description="never exceed 0.5 m/s", db=db)
        cmd_policy(add_args)
        capsys.readouterr()  # clear
        # Now list
        list_args = _policy_args("list", db=db)
        cmd_policy(list_args)
        out = capsys.readouterr().out
        assert "speed_limit" in out

    def test_list_json(self, capsys, tmp_path):
        from apyrobo.cli import cmd_policy
        db = str(tmp_path / "policies.db")
        add_args = _policy_args("add", description="never exceed 0.5 m/s", db=db)
        cmd_policy(add_args)
        capsys.readouterr()
        list_args = _policy_args("list", db=db, json=True)
        cmd_policy(list_args)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, list)
        assert data[0]["constraint_type"] == "speed_limit"


class TestCmdPolicyRemove:
    def test_remove_existing(self, capsys, tmp_path):
        from apyrobo.cli import cmd_policy
        db = str(tmp_path / "policies.db")
        add_args = _policy_args("add", description="never exceed 0.5 m/s", db=db)
        cmd_policy(add_args)
        capsys.readouterr()
        # Get the policy_id from list
        store = NLPolicyStore(db)
        policies = store.get_active_policies()
        store.close()
        pid = policies[0].policy_id

        remove_args = _policy_args("remove", policy_id=pid, db=db)
        cmd_policy(remove_args)
        out = capsys.readouterr().out
        assert "Removed" in out

    def test_remove_nonexistent_exits(self, tmp_path):
        from apyrobo.cli import cmd_policy
        args = _policy_args("remove", policy_id="no-such-id",
                            db=str(tmp_path / "policies.db"))
        with pytest.raises(SystemExit) as exc:
            cmd_policy(args)
        assert exc.value.code == 1


class TestCmdPolicyCheck:
    def test_check_no_violations(self, capsys, tmp_path):
        from apyrobo.cli import cmd_policy
        db = str(tmp_path / "p.db")
        cmd_policy(_policy_args("add", description="never exceed 0.5 m/s", db=db))
        capsys.readouterr()
        cmd_policy(_policy_args("check", action="navigate at 0.3 m/s", db=db))
        out = capsys.readouterr().out
        assert "No policy violations" in out

    def test_check_violation_exits(self, tmp_path):
        from apyrobo.cli import cmd_policy
        db = str(tmp_path / "p.db")
        cmd_policy(_policy_args("add", description="never exceed 0.5 m/s", db=db))
        with pytest.raises(SystemExit) as exc:
            cmd_policy(_policy_args("check", action="navigate at 2.0 m/s", db=db))
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestSafetyModuleExports:
    def test_importable(self):
        from apyrobo.safety import NLSafetyPolicy, NLPolicyParser, NLPolicyStore
        assert NLSafetyPolicy is not None
        assert NLPolicyParser is not None
        assert NLPolicyStore is not None

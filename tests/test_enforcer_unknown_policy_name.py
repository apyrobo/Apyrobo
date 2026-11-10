"""A misspelt policy name must not quietly run the looser default policy."""

from __future__ import annotations

import pytest

from apyrobo.core.robot import Robot
from apyrobo.safety.enforcer import (
    POLICY_REGISTRY,
    STRICT_POLICY,
    SafetyEnforcer,
    SafetyPolicy,
)


def _robot() -> Robot:
    return Robot.discover("mock://policy-name")


@pytest.mark.parametrize("name", ["Strict", "STRICT", "stirct", "strict ", "", "safe"])
def test_unknown_name_is_refused_at_construction(name: str) -> None:
    with pytest.raises(ValueError) as err:
        SafetyEnforcer(_robot(), policy=name)
    message = str(err.value)
    assert repr(name) in message
    assert "default" in message and "strict" in message


def test_registered_names_still_work() -> None:
    assert SafetyEnforcer(_robot(), policy="strict").policy is STRICT_POLICY
    assert SafetyEnforcer(_robot(), policy="default").policy.name == "default"
    assert SafetyEnforcer(_robot()).policy.name == "default"


def test_strict_by_name_clamps_to_its_limit() -> None:
    # "Strict" (capital S) used to run the default policy and send 1.4 m/s.
    enforcer = SafetyEnforcer(_robot(), policy="strict")
    enforcer.move(2.0, 0.0, speed=1.4)
    assert enforcer.last_speed == pytest.approx(STRICT_POLICY.max_speed)


def test_unknown_swap_keeps_the_active_policy() -> None:
    enforcer = SafetyEnforcer(_robot(), policy="strict")
    with pytest.raises(ValueError, match="stirct"):
        enforcer.swap_policy("stirct")
    assert enforcer.policy is STRICT_POLICY
    assert not any(e.event_type == "policy_swap" for e in enforcer.audit_log)
    enforcer.move(2.0, 0.0, speed=1.4)
    assert enforcer.last_speed == pytest.approx(STRICT_POLICY.max_speed)


def test_a_user_registered_name_is_found() -> None:
    custom = SafetyPolicy(name="lab", max_speed=0.3)
    POLICY_REGISTRY["lab"] = custom
    try:
        assert SafetyEnforcer(_robot(), policy="lab").policy is custom
    finally:
        POLICY_REGISTRY.pop("lab", None)


def test_policy_objects_are_unchanged() -> None:
    custom = SafetyPolicy(name="not-registered", max_speed=0.2)
    assert SafetyEnforcer(_robot(), policy=custom).policy is custom

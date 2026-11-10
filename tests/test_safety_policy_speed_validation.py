"""SafetyPolicy refuses speed caps that cap nothing or drive the robot wrongly.

A max_speed of NaN or inf let any speed through, and 0 or a negative cap was
sent to the robot as the speed. The CLIs already refused these flags; the
Python API and config files did not.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from apyrobo.config import ApyroboConfig
from apyrobo.core.schemas import SafetyPolicyRef
from apyrobo.safety.enforcer import (
    DEFAULT_POLICY,
    POLICY_REGISTRY,
    STRICT_POLICY,
    SafetyPolicy,
    check_speed_cap,
)

BAD_CAPS = [math.nan, math.inf, -math.inf, 0, 0.0, -1, -0.5, True, False, "1.5", None]


@pytest.mark.parametrize("bad", BAD_CAPS, ids=repr)
def test_max_speed_must_be_a_positive_finite_number(bad: object) -> None:
    with pytest.raises(ValueError, match="max_speed must be a positive number"):
        SafetyPolicy(max_speed=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", BAD_CAPS, ids=repr)
def test_max_angular_speed_must_be_a_positive_finite_number(bad: object) -> None:
    with pytest.raises(ValueError, match="max_angular_speed must be a positive number"):
        SafetyPolicy(max_angular_speed=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("good", [0.1, 1, 1.5, 3.0, Fraction(1, 2)], ids=repr)
def test_valid_caps_are_kept_unchanged(good: float) -> None:
    policy = SafetyPolicy(max_speed=good, max_angular_speed=good)
    assert policy.max_speed == good
    assert policy.max_angular_speed == good


def test_named_policies_still_build() -> None:
    assert DEFAULT_POLICY.max_speed == 1.5
    assert STRICT_POLICY.max_angular_speed == 1.0
    for policy in POLICY_REGISTRY.values():
        assert policy.max_speed > 0 and math.isfinite(policy.max_speed)


def test_check_speed_cap_returns_value() -> None:
    assert check_speed_cap(0.25, "max_speed") == 0.25


@pytest.mark.parametrize("text", [".nan", ".inf", "-1", "0"])
def test_config_file_with_bad_max_speed_is_refused(tmp_path, text: str) -> None:
    path = tmp_path / "apyrobo.yaml"
    path.write_text(f"safety:\n  max_speed: {text}\n")
    config = ApyroboConfig.from_file(path)
    with pytest.raises(ValueError, match="max_speed must be a positive number"):
        config.safety_policy()


def test_config_file_with_good_max_speed_builds(tmp_path) -> None:
    path = tmp_path / "apyrobo.yaml"
    path.write_text("safety:\n  max_speed: 0.4\n")
    assert ApyroboConfig.from_file(path).safety_policy().max_speed == 0.4


def test_from_ref_zero_is_refused_not_turned_into_default() -> None:
    with pytest.raises(ValueError, match="max_speed must be a positive number"):
        SafetyPolicy.from_ref(SafetyPolicyRef(policy_name="p", max_speed=0))


def test_from_ref_without_speed_uses_default() -> None:
    assert SafetyPolicy.from_ref(SafetyPolicyRef(policy_name="p")).max_speed == 1.5

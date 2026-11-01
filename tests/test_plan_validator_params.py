"""PlanValidator handles a step whose ``params`` is null or not an object."""

from __future__ import annotations

import pytest

from apyrobo.skills.discovery import DiscoveryRegistry
from apyrobo.skills.plan_validator import PlanValidator


def _validator() -> PlanValidator:
    return PlanValidator(discovery_registry=DiscoveryRegistry())


def test_null_params_are_treated_as_empty():
    result = _validator().validate([{"skill": "pick_object", "params": None}])
    assert not any("params for skill" in i.message for i in result.issues)


@pytest.mark.parametrize("params", [[], "x", 3, True])
def test_non_object_params_are_an_error(params):
    result = _validator().validate([{"skill": "pick_object", "params": params}])
    assert not result.valid
    assert any("must be an object" in i.message for i in result.errors())

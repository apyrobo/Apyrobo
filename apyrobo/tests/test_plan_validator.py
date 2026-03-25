"""Tests for apyrobo.skills.plan_validator"""
from __future__ import annotations

import pytest

from apyrobo.skills.plan_validator import (
    ValidationIssue,
    ValidationResult,
    PlanValidator,
)
from apyrobo.skills.discovery import DiscoveryRegistry, SkillDiscovery, SkillManifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(extra: list[SkillManifest] | None = None) -> DiscoveryRegistry:
    """Return a DiscoveryRegistry with all built-in skills + optional extras."""
    discovery = SkillDiscovery(extra_manifests=extra)
    registry = DiscoveryRegistry(discovery)
    registry.refresh(available_capabilities=[])
    return registry


def _step(skill: str, params: dict | None = None, depends_on: list | None = None) -> dict:
    return {
        "skill": skill,
        "params": params or {},
        "depends_on": depends_on or [],
    }


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

class TestValidationResult:
    def test_errors_filtered(self):
        issues = [
            ValidationIssue("error", "s", "e"),
            ValidationIssue("warning", "s", "w"),
            ValidationIssue("info", "s", "i"),
        ]
        vr = ValidationResult(valid=False, issues=issues)
        assert len(vr.errors()) == 1
        assert vr.errors()[0].severity == "error"

    def test_warnings_filtered(self):
        issues = [
            ValidationIssue("error", "s", "e"),
            ValidationIssue("warning", "s", "w"),
        ]
        vr = ValidationResult(valid=False, issues=issues)
        assert len(vr.warnings()) == 1

    def test_valid_with_no_errors(self):
        vr = ValidationResult(valid=True, issues=[ValidationIssue("warning", "s", "w")])
        assert vr.valid is True
        assert len(vr.errors()) == 0


# ---------------------------------------------------------------------------
# PlanValidator — no registry (bare mode)
# ---------------------------------------------------------------------------

class TestPlanValidatorNoRegistry:
    def test_empty_plan_is_valid(self):
        validator = PlanValidator()
        result = validator.validate([])
        assert result.valid is True
        assert result.issues == []

    def test_missing_skill_key_produces_error(self):
        validator = PlanValidator()
        result = validator.validate([{"params": {}, "depends_on": []}])
        assert not result.valid
        assert any("missing a 'skill' name" in i.message for i in result.errors())

    def test_no_registry_skips_existence_check(self):
        validator = PlanValidator()
        result = validator.validate([_step("some_unknown_skill")])
        # Without a registry, unknown skill names are NOT flagged
        assert result.valid is True


# ---------------------------------------------------------------------------
# PlanValidator — with DiscoveryRegistry
# ---------------------------------------------------------------------------

class TestPlanValidatorWithRegistry:
    def setup_method(self):
        self.registry = _make_registry()
        self.validator = PlanValidator(discovery_registry=self.registry)

    # -- Skill existence --

    def test_valid_plan_all_known_skills(self):
        plan = [_step("navigate_to", {"x": 1.0, "y": 2.0}), _step("stop")]
        result = self.validator.validate(plan)
        assert result.valid is True, result.issues

    def test_missing_skill_raises_error(self):
        plan = [_step("fly_to_moon")]
        result = self.validator.validate(plan)
        assert not result.valid
        errors = result.errors()
        assert any("fly_to_moon" in e.message for e in errors)

    # -- Capability checks --

    def test_missing_capability_raises_error(self):
        plan = [_step("navigate_to", {"x": 0.0, "y": 0.0})]
        result = self.validator.validate(plan, available_capabilities=[])
        # navigate_to requires "move"
        assert not result.valid
        assert any("move" in e.message for e in result.errors())

    def test_sufficient_capabilities_no_error(self):
        plan = [_step("navigate_to", {"x": 0.0, "y": 0.0})]
        result = self.validator.validate(plan, available_capabilities=["move"])
        assert result.valid, result.issues

    def test_skill_with_no_requirements_always_passes_cap_check(self):
        plan = [_step("stop"), _step("report_status")]
        result = self.validator.validate(plan, available_capabilities=[])
        assert result.valid, result.issues

    def test_multiple_missing_caps_reported_per_skill(self):
        # Create a skill that needs two caps
        manifest = SkillManifest(
            name="dual_cap",
            version="1.0",
            description="needs two caps",
            parameters={},
            requirements=["arm", "camera"],
        )
        registry = _make_registry(extra=[manifest])
        validator = PlanValidator(discovery_registry=registry)
        plan = [_step("dual_cap")]
        result = validator.validate(plan, available_capabilities=[])
        assert not result.valid
        errors = result.errors()
        assert any("arm" in e.message or "camera" in e.message for e in errors)

    # -- Circular dependency --

    def test_no_circular_deps_valid(self):
        plan = [
            _step("stop", depends_on=[]),
            _step("navigate_to", {"x": 1.0, "y": 1.0}, depends_on=["stop"]),
        ]
        result = self.validator.validate(plan, available_capabilities=["move"])
        assert result.valid, result.issues

    def test_direct_circular_dep_detected(self):
        plan = [
            _step("navigate_to", {"x": 1.0, "y": 1.0}, depends_on=["stop"]),
            _step("stop", depends_on=["navigate_to"]),
        ]
        result = self.validator.validate(plan, available_capabilities=["move"])
        assert not result.valid
        assert any("Circular dependency" in e.message for e in result.errors())

    def test_self_dependency_detected(self):
        plan = [_step("stop", depends_on=["stop"])]
        result = self.validator.validate(plan, available_capabilities=[])
        assert not result.valid
        assert any("Circular" in e.message for e in result.errors())

    # -- Resource conflicts --

    def test_no_resource_conflict_single_gripper(self):
        plan = [_step("pick_object")]
        result = self.validator.validate(plan, available_capabilities=["gripper"])
        assert result.valid, result.issues

    def test_gripper_resource_conflict(self):
        plan = [_step("pick_object"), _step("place_object")]
        result = self.validator.validate(plan, available_capabilities=["gripper"])
        assert not result.valid
        assert any("gripper" in e.message for e in result.errors())

    def test_camera_resource_conflict(self):
        # capture_image and stream_video both need exclusive camera
        plan = [
            _step("capture_image"),
            _step("stream_video"),
        ]
        # No registry for these (unknown skills), but conflict check still runs
        validator = PlanValidator()
        result = validator.validate(plan)
        assert not result.valid
        assert any("camera" in e.message for e in result.errors())

    def test_different_resources_no_conflict(self):
        plan = [_step("pick_object"), _step("capture_image")]
        validator = PlanValidator()
        result = validator.validate(plan)
        # pick_object → gripper, capture_image → camera: no conflict
        assert result.valid, result.issues

    # -- Combined plan --

    def test_mixed_issues_all_reported(self):
        plan = [
            _step("navigate_to", {"x": 1.0, "y": 2.0}, depends_on=[]),
            _step("ghost_skill"),
        ]
        result = self.validator.validate(plan, available_capabilities=[])
        # navigate_to: missing "move" cap + ghost_skill: unknown
        assert not result.valid
        assert len(result.errors()) >= 2

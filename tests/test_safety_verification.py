"""Tests for formal safety verification."""
import json
import pytest
from apyrobo.safety.verification import (
    SafetyProperty,
    SafetyVerifier,
    VerificationResult,
    CertificationReport,
    generate_certification_report,
    BUILTIN_PROPERTIES,
)


@pytest.fixture
def verifier():
    return SafetyVerifier()


def test_builtin_properties_count():
    assert len(BUILTIN_PROPERTIES) == 3


def test_builtin_property_names():
    names = {p.name for p in BUILTIN_PROPERTIES}
    assert "speed_limit_invariant" in names
    assert "no_collision_zone_invariant" in names
    assert "gripper_safe_release" in names


def test_custom_property():
    p = SafetyProperty("battery", "Battery >= 10%", "invariant", "battery >= 0.1", "warning")
    assert p.severity == "warning"


def test_verify_empty_plan(verifier):
    assert all(r.satisfied for r in verifier.verify_plan([]))


def test_verify_plan_speed_ok(verifier):
    results = verifier.verify_plan(
        [{"skill": "move_to", "params": {"speed": 0.5, "max_speed": 1.0}}]
    )
    r = next(r for r in results if r.property_name == "speed_limit_invariant")
    assert r.satisfied is True


def test_verify_plan_speed_violation(verifier):
    results = verifier.verify_plan(
        [{"skill": "move_to", "params": {"speed": 2.0, "max_speed": 1.0}}]
    )
    r = next(r for r in results if r.property_name == "speed_limit_invariant")
    assert r.satisfied is False
    assert r.counterexample is not None


def test_verify_plan_collision_violation(verifier):
    results = verifier.verify_plan(
        [{"skill": "move_to", "params": {"target": "zone_A", "collision_zones": ["zone_A"]}}]
    )
    r = next(r for r in results if r.property_name == "no_collision_zone_invariant")
    assert r.satisfied is False


def test_verify_plan_no_collision(verifier):
    results = verifier.verify_plan(
        [{"skill": "move_to", "params": {"target": "zone_C", "collision_zones": ["zone_A"]}}]
    )
    r = next(r for r in results if r.property_name == "no_collision_zone_invariant")
    assert r.satisfied is True


def test_verify_state_sequence_ok(verifier):
    results = verifier.verify_state_sequence([{"speed": 0.3, "max_speed": 1.0}])
    r = next(r for r in results if r.property_name == "speed_limit_invariant")
    assert r.satisfied is True


def test_verify_state_sequence_violation(verifier):
    results = verifier.verify_state_sequence([{"speed": 1.5, "max_speed": 1.0}])
    r = next(r for r in results if r.property_name == "speed_limit_invariant")
    assert r.satisfied is False


def test_export_proof_json(verifier):
    results = verifier.verify_plan([])
    data = json.loads(verifier.export_proof(results, "json"))
    assert isinstance(data, list)
    assert data[0]["property"] == results[0].property_name


def test_export_proof_markdown(verifier):
    output = verifier.export_proof(verifier.verify_plan([]), "markdown")
    assert "# Safety Verification Report" in output


def test_export_proof_invalid_format(verifier):
    with pytest.raises(ValueError):
        verifier.export_proof([], format="xml")


def test_certification_report_fields(verifier):
    results = verifier.verify_plan([])
    report = generate_certification_report(verifier, results, "robot-001")
    assert report.robot_id == "robot-001"
    assert len(report.hash) == 64  # SHA-256 hex


def test_certification_all_satisfied_false(verifier):
    results = verifier.verify_plan(
        [{"skill": "move_to", "params": {"speed": 5.0, "max_speed": 1.0}}]
    )
    report = generate_certification_report(verifier, results, "robot-001")
    assert report.all_satisfied is False


def test_certification_properties_verified_count(verifier):
    results = verifier.verify_plan([])
    report = generate_certification_report(verifier, results, "robot-x")
    assert report.properties_verified == len(BUILTIN_PROPERTIES)

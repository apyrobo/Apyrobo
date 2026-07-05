"""Spec/implementation drift guard.

Validates the serialized output of the reference implementation against the
normative JSON Schemas in spec/schemas/. If one of these tests fails, either
the code broke the protocol or the spec needs an RFC — never ship the failure.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import (
    SafetyPolicyRef,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from apyrobo.orchestration.adapter import (
    MockOrchestrationAdapter,
    OrchestrationMessage,
    OrchestrationServer,
)
from apyrobo.skills.agent import Agent
from apyrobo.skills.discovery import SkillDiscovery

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "spec" / "schemas"


def load_schema(name: str) -> dict:
    with open(SCHEMA_DIR / name) as f:
        return json.load(f)


def validate(instance: dict, schema_name: str) -> None:
    jsonschema.validate(
        instance=instance,
        schema=load_schema(schema_name),
        cls=jsonschema.Draft202012Validator,
    )


def model_to_json_dict(model) -> dict:
    """Serialize a pydantic model (or the no-pydantic fallback) to plain JSON types."""
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    # Fallback BaseModel: round-trip through json with enum-aware default
    return json.loads(
        json.dumps(vars(model), default=lambda o: getattr(o, "value", vars(o)))
    )


# ---------------------------------------------------------------------------
# Wire protocol — orchestration-message.schema.json
# ---------------------------------------------------------------------------

class TestOrchestrationMessageSchema:
    def test_request_message(self):
        msg = OrchestrationMessage(task="navigate to dock")
        validate(msg.to_dict(), "orchestration-message.schema.json")

    def test_planned_response_from_server(self):
        adapter = MockOrchestrationAdapter(tasks=["navigate to the dock"])
        server = OrchestrationServer(
            adapter,
            Agent(provider="rule"),
            default_robot=Robot.discover("mock://spec-test"),
        )
        server.run()
        assert len(adapter.sent) == 1
        response = adapter.sent[0].to_dict()
        validate(response, "orchestration-message.schema.json")
        assert response["metadata"]["status"] == "planned"

    def test_error_response_from_server(self):
        adapter = MockOrchestrationAdapter(
            tasks=[OrchestrationMessage(task="do a thing", robot_uri="bogus://nowhere")]
        )
        server = OrchestrationServer(adapter, Agent(provider="rule"))
        server.run()
        assert len(adapter.sent) == 1
        response = adapter.sent[0].to_dict()
        validate(response, "orchestration-message.schema.json")
        assert response["metadata"]["status"] == "error"
        assert response["metadata"]["error"]

    def test_task_is_required(self):
        with pytest.raises(jsonschema.ValidationError):
            validate({"robot_uri": "mock://x"}, "orchestration-message.schema.json")

    def test_robot_uri_must_have_scheme(self):
        with pytest.raises(jsonschema.ValidationError):
            validate(
                {"task": "go", "robot_uri": "no-scheme-here"},
                "orchestration-message.schema.json",
            )


# ---------------------------------------------------------------------------
# Capability model — robot-capability.schema.json
# ---------------------------------------------------------------------------

class TestRobotCapabilitySchema:
    def test_mock_adapter_capabilities(self):
        caps = Robot.discover("mock://spec-test").capabilities()
        validate(model_to_json_dict(caps), "robot-capability.schema.json")

    def test_robot_id_required(self):
        with pytest.raises(jsonschema.ValidationError):
            validate({"name": "anon"}, "robot-capability.schema.json")

    def test_unknown_capability_type_rejected(self):
        with pytest.raises(jsonschema.ValidationError):
            validate(
                {
                    "robot_id": "r1",
                    "name": "r1",
                    "capabilities": [{"capability_type": "levitate", "name": "levitate"}],
                },
                "robot-capability.schema.json",
            )


# ---------------------------------------------------------------------------
# Skill manifests — skill-manifest.schema.json
# ---------------------------------------------------------------------------

class TestSkillManifestSchema:
    def test_all_builtin_manifests(self):
        for manifest in SkillDiscovery().scan_library():
            validate(manifest.to_dict(), "skill-manifest.schema.json")

    def test_version_must_be_semver(self):
        with pytest.raises(jsonschema.ValidationError):
            validate(
                {"name": "x", "version": "one", "description": "d", "parameters": {}},
                "skill-manifest.schema.json",
            )


# ---------------------------------------------------------------------------
# Task request / result — task-request.schema.json, task-result.schema.json
# ---------------------------------------------------------------------------

class TestTaskSchemas:
    def test_task_request(self):
        req = TaskRequest(
            task_name="deliver_package",
            parameters={"destination": "dock"},
            priority=5,
            safety_policy=SafetyPolicyRef(max_speed=0.5, human_proximity_limit=1.0),
        )
        validate(model_to_json_dict(req), "task-request.schema.json")

    def test_task_result(self):
        res = TaskResult(
            task_name="deliver_package",
            status=TaskStatus.COMPLETED,
            confidence=0.9,
            steps_completed=3,
            steps_total=3,
        )
        validate(model_to_json_dict(res), "task-result.schema.json")

    def test_priority_out_of_range_rejected(self):
        with pytest.raises(jsonschema.ValidationError):
            validate(
                {"task_name": "t", "priority": 11},
                "task-request.schema.json",
            )


# ---------------------------------------------------------------------------
# Packaged schema copies — apyrobo/conformance/schemas/
# ---------------------------------------------------------------------------

class TestPackagedSchemaCopies:
    """The conformance suite ships copies of spec/schemas/ inside the wheel.

    They must stay byte-identical to the normative spec files — if this
    fails, re-copy: cp spec/schemas/*.json apyrobo/conformance/schemas/
    """

    PKG_DIR = (
        Path(__file__).resolve().parent.parent
        / "apyrobo" / "conformance" / "schemas"
    )

    def test_same_file_set(self):
        spec_names = {p.name for p in SCHEMA_DIR.glob("*.json")}
        pkg_names = {p.name for p in self.PKG_DIR.glob("*.json")}
        assert spec_names == pkg_names

    def test_copies_are_byte_identical(self):
        for spec_file in SCHEMA_DIR.glob("*.json"):
            packaged = self.PKG_DIR / spec_file.name
            assert packaged.read_bytes() == spec_file.read_bytes(), (
                f"{packaged} has drifted from {spec_file} — "
                "re-copy spec/schemas/ into apyrobo/conformance/schemas/"
            )

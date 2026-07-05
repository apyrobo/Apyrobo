"""Protocol fuzzing (v8 Phase 2): property-based tests over the parsers.

Pass criterion per the roadmap: hostile input — malformed JSON, wrong
types, capability spoofing, oversized payloads — always produces safe
rejection (a controlled error or an ``error`` response), never a crash or
undefined behavior.

Covers the three spec surfaces:
- wire protocol (spec/wire-protocol.md): message parsing, the stdio
  framing, and the server loop's planned|error contract
- skill manifests (spec/skill-manifest.md): Skill/manifest deserialization
  and version constraint parsing
- capability model (spec/capability-model.md): the "capability list is
  exhaustive" safety invariant — spoofed plans must be rejected
"""
from __future__ import annotations

import io
import json
from typing import Any

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import CapabilityType
from apyrobo.orchestration.adapter import (
    OrchestrationMessage,
    OrchestrationServer,
    StdioOrchestrationAdapter,
)
from apyrobo.skills.agent import Agent
from apyrobo.skills.discovery import (
    DiscoveryRegistry,
    SkillDiscovery,
    SkillManifest,
)
from apyrobo.skills.package import (
    check_version_constraint,
    validate_package_name,
    validate_version,
)
from apyrobo.skills.plan_validator import PlanValidator
from apyrobo.skills.skill import Skill

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

# Bounded runtime: the whole file should stay in tens of seconds.
FUZZ = settings(
    max_examples=75,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def json_values(max_leaves: int = 20) -> Any:
    """Arbitrary JSON-serializable values (spec forbids NaN/Infinity)."""
    return st.recursive(
        st.none()
        | st.booleans()
        | st.integers(min_value=-(2**53), max_value=2**53)
        | st.floats(allow_nan=False, allow_infinity=False)
        | st.text(max_size=40),
        lambda children: st.lists(children, max_size=4)
        | st.dictionaries(st.text(max_size=15), children, max_size=4),
        max_leaves=max_leaves,
    )


def hostile_message_dicts() -> Any:
    """Dicts that may or may not resemble a wire message."""
    return st.dictionaries(
        st.sampled_from(
            ["task", "robot_uri", "metadata", "source", "x-junk", "count"]
        )
        | st.text(max_size=10),
        json_values(max_leaves=8),
        max_size=6,
    )


def valid_messages() -> Any:
    """Well-formed OrchestrationMessages, per the schema."""
    return st.builds(
        OrchestrationMessage,
        task=st.text(min_size=1, max_size=60),
        robot_uri=st.just("") | st.just("mock://fuzz-bot"),
        metadata=st.dictionaries(
            st.text(max_size=10), json_values(max_leaves=5), max_size=3
        ),
        source=st.text(max_size=15),
    )


CONTROLLED = (ValueError, KeyError, TypeError, json.JSONDecodeError)


# ---------------------------------------------------------------------------
# Wire protocol — message parsing
# ---------------------------------------------------------------------------

class TestWireMessageFuzz:
    @FUZZ
    @given(data=hostile_message_dicts())
    def test_from_dict_never_raises_and_normalizes_types(self, data):
        msg = OrchestrationMessage.from_dict(data)
        # Whatever came in, the parsed message has spec-shaped fields …
        assert isinstance(msg.task, str)
        assert isinstance(msg.robot_uri, str)
        assert isinstance(msg.metadata, dict)
        assert isinstance(msg.source, str)
        # … and serializes to a schema-valid shape (robot_uri omitted if empty).
        wire = msg.to_dict()
        assert isinstance(wire["task"], str)
        assert wire.get("robot_uri", "x://y") != ""

    @FUZZ
    @given(msg=valid_messages())
    def test_valid_messages_round_trip(self, msg):
        restored = OrchestrationMessage.from_dict(msg.to_dict())
        assert restored == msg

    @FUZZ
    @given(line=st.text(max_size=200).filter(lambda s: s.strip()))
    def test_stdio_receive_never_raises_on_any_line(self, line):
        """§2.1/§2.3: one hostile line must parse to a message, not crash.

        Includes text that is valid JSON but not an object (e.g. "[1, 2]"),
        which must be treated like malformed input.
        """
        adapter = StdioOrchestrationAdapter(infile=io.StringIO(line + "\n"))
        msg = adapter.receive()
        assert msg is None or isinstance(msg, OrchestrationMessage)

    @FUZZ
    @given(value=json_values(max_leaves=10))
    def test_stdio_receive_handles_any_json_document(self, value):
        """Valid JSON of every type — object, array, scalar — never crashes."""
        line = json.dumps(value)
        if not line.strip():
            return
        adapter = StdioOrchestrationAdapter(infile=io.StringIO(line + "\n"))
        msg = adapter.receive()
        assert msg is None or isinstance(msg, OrchestrationMessage)


# ---------------------------------------------------------------------------
# Wire protocol — server loop
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fuzz_server() -> OrchestrationServer:
    from apyrobo.orchestration.adapter import MockOrchestrationAdapter

    return OrchestrationServer(
        MockOrchestrationAdapter(),
        Agent(provider="rule"),
        default_robot=Robot.discover("mock://fuzz-default"),
        default_robot_uri="mock://fuzz-default",
    )


class TestServerLoopFuzz:
    @FUZZ
    @given(data=hostile_message_dicts())
    def test_handle_always_answers_planned_or_error(self, fuzz_server, data):
        """§3: whatever arrives, the response reports planned or error."""
        response = fuzz_server._handle(OrchestrationMessage.from_dict(data))
        assert response.metadata["status"] in ("planned", "error")
        assert response.source == "orchestration_server"

    @FUZZ
    @given(data=hostile_message_dicts())
    def test_responses_stay_schema_valid_under_hostile_input(
        self, fuzz_server, data
    ):
        jsonschema = pytest.importorskip("jsonschema")
        from apyrobo.conformance.spec_schemas import load_schema

        response = fuzz_server._handle(OrchestrationMessage.from_dict(data))
        jsonschema.validate(
            instance=response.to_dict(),
            schema=load_schema("orchestration-message"),
            cls=jsonschema.Draft202012Validator,
        )

    @FUZZ
    @given(tasks=st.lists(st.text(min_size=1, max_size=30), max_size=6))
    def test_n_messages_in_yield_n_responses_in_order(self, tasks):
        """§4: strictly sequential — one response per message, in order."""
        from apyrobo.orchestration.adapter import MockOrchestrationAdapter

        adapter = MockOrchestrationAdapter(tasks=list(tasks))
        server = OrchestrationServer(
            adapter,
            Agent(provider="rule"),
            default_robot=Robot.discover("mock://fuzz-seq"),
            default_robot_uri="mock://fuzz-seq",
        )
        server.run()
        assert [m.task for m in adapter.sent] == list(tasks)

    def test_oversized_payload_is_survived(self, fuzz_server):
        """§5: servers MAY reject oversized messages but must not crash."""
        huge = OrchestrationMessage(
            task="x" * 300_000, metadata={"blob": "y" * 300_000}
        )
        response = fuzz_server._handle(huge)
        assert response.metadata["status"] in ("planned", "error")


# ---------------------------------------------------------------------------
# Robot URIs (adapter-contract.md §1)
# ---------------------------------------------------------------------------

class TestRobotUriFuzz:
    @FUZZ
    @given(uri=st.text(max_size=80))
    def test_discover_returns_robot_or_valueerror(self, uri):
        try:
            robot = Robot.discover(uri)
        except ValueError:
            return  # safe rejection
        assert robot is not None
        assert uri.partition("://")[0] in ("mock", "gazebo", "mqtt", "http")


# ---------------------------------------------------------------------------
# Skill manifests (skill-manifest.md)
# ---------------------------------------------------------------------------

class TestManifestFuzz:
    @FUZZ
    @given(
        requirements=st.lists(st.text(max_size=15), max_size=6),
        available=st.lists(st.text(max_size=15), max_size=6),
    )
    def test_matches_capabilities_is_exactly_subset(self, requirements, available):
        """§2: ALL requirements must be satisfied — no more, no less."""
        manifest = SkillManifest(
            name="fuzz", version="1.0.0", description="d",
            parameters={}, requirements=requirements,
        )
        assert manifest.matches_capabilities(available) == set(
            requirements
        ).issubset(set(available))

    @FUZZ
    @given(text=st.text(max_size=200))
    def test_skill_from_json_rejects_garbage_safely(self, text):
        try:
            skill = Skill.from_json(text)
        except CONTROLLED:
            return  # safe rejection
        assert isinstance(skill, Skill)

    @FUZZ
    @given(data=st.dictionaries(st.text(max_size=15), json_values(max_leaves=6), max_size=6))
    def test_skill_from_dict_rejects_hostile_dicts_safely(self, data):
        try:
            skill = Skill.from_dict(data)
        except CONTROLLED:
            return
        assert isinstance(skill, Skill)

    @FUZZ
    @given(capability=st.text(min_size=1, max_size=20))
    def test_unknown_capability_types_map_to_custom(self, capability):
        """capability-model.md §1: unknown enum values must not fail —
        consumers map unknown CapabilityType to custom semantics."""
        skill = Skill.from_dict(
            {"skill_id": "s1", "name": "n", "required_capability": capability}
        )
        known = {c.value for c in CapabilityType}
        if capability in known:
            assert skill.required_capability.value == capability
        else:
            assert skill.required_capability is CapabilityType.CUSTOM


class TestVersionFuzz:
    @FUZZ
    @given(text=st.text(max_size=30))
    def test_validators_return_bool_never_raise(self, text):
        assert isinstance(validate_version(text), bool)
        assert isinstance(validate_package_name(text), bool)

    @FUZZ
    @given(
        major=st.integers(0, 99), minor=st.integers(0, 99), patch=st.integers(0, 99)
    )
    def test_version_satisfies_its_own_bounds(self, major, minor, patch):
        version = f"{major}.{minor}.{patch}"
        assert check_version_constraint(version, f"=={version}")
        assert check_version_constraint(version, f">={version}")
        assert check_version_constraint(version, f"<={version}")
        assert not check_version_constraint(version, f"<{version}")
        assert not check_version_constraint(version, f">{version}")

    @FUZZ
    @given(version=st.text(max_size=20), constraint=st.text(max_size=20))
    def test_constraint_checks_reject_garbage_safely(self, version, constraint):
        try:
            result = check_version_constraint(version, constraint)
        except CONTROLLED:
            return  # safe rejection of malformed inputs
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Capability spoofing (capability-model.md §2 safety invariant)
# ---------------------------------------------------------------------------

class TestCapabilitySpoofingFuzz:
    """A planner MUST treat the capability list as exhaustive: plans that
    use capabilities the robot did not declare are invalid and MUST be
    rejected before execution."""

    @FUZZ
    @given(data=st.data())
    def test_plans_requiring_undeclared_capabilities_are_rejected(self, data):
        discovery = SkillDiscovery()
        manifests = [m for m in discovery.scan_library() if m.requirements]
        manifest = data.draw(st.sampled_from(manifests))

        # Declare a capability set that is missing at least one requirement.
        declared = data.draw(
            st.lists(
                st.sampled_from(sorted({r for m in manifests for r in m.requirements})),
                max_size=5,
            )
        )
        missing = [r for r in manifest.requirements if r not in declared]
        if not missing:
            declared = [c for c in declared if c != manifest.requirements[0]]

        validator = PlanValidator(
            discovery_registry=DiscoveryRegistry(discovery=discovery)
        )
        result = validator.validate(
            [{"skill": manifest.name, "params": {}}],
            available_capabilities=declared,
        )
        assert not result.valid
        assert any(
            "requires capabilities" in issue.message for issue in result.errors()
        )

    @FUZZ
    @given(
        plan=st.lists(
            st.dictionaries(
                st.sampled_from(["skill", "params", "depends_on", "junk"]),
                json_values(max_leaves=5),
                max_size=4,
            ),
            max_size=4,
        )
    )
    def test_validator_never_crashes_on_hostile_plans(self, plan):
        validator = PlanValidator(discovery_registry=DiscoveryRegistry())
        try:
            result = validator.validate(plan, available_capabilities=["navigate"])
        except CONTROLLED:
            return
        assert isinstance(result.valid, bool)

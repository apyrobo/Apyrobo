# APYROBO Protocol Specification

**Spec version: 1.0-draft** · Status: draft, open for RFC · License: Apache-2.0

This directory is the language-agnostic specification of the APYROBO protocol —
the contract that lets any client, adapter, or alternative implementation
interoperate with an APYROBO orchestrator without depending on the Python
reference implementation.

The specification is the durable artifact. The Python package in
[`apyrobo/`](../apyrobo/) is the *reference implementation* of this spec, not
its definition. Where the two disagree, that is a bug in one of them and should
be reported.

## Documents

| Document | Defines |
|----------|---------|
| [capability-model.md](capability-model.md) | The semantic data model: what a "robot" is to an agent — capabilities, sensors, joints, task requests and results |
| [adapter-contract.md](adapter-contract.md) | The behavioral contract every capability adapter must satisfy, including the robot URI scheme |
| [wire-protocol.md](wire-protocol.md) | The orchestration wire protocol: JSON messages over stdio or WebSocket between clients and an orchestration server |
| [skill-manifest.md](skill-manifest.md) | The skill manifest format used for discovery and capability matching |

## JSON Schemas

Machine-readable schemas live in [`schemas/`](schemas/) (JSON Schema
draft 2020-12). They are normative for message *shape*; the prose documents are
normative for *behavior*.

| Schema | Validates |
|--------|-----------|
| [orchestration-message.schema.json](schemas/orchestration-message.schema.json) | Every message on the wire protocol, both directions |
| [robot-capability.schema.json](schemas/robot-capability.schema.json) | The capability profile returned by `get_capabilities` |
| [skill-manifest.schema.json](schemas/skill-manifest.schema.json) | Skill manifests |
| [task-request.schema.json](schemas/task-request.schema.json) | Task submissions |
| [task-result.schema.json](schemas/task-result.schema.json) | Task outcomes |

The test suite validates the reference implementation's serialized output
against these schemas (`tests/test_spec_schemas.py`), so the spec and the code
cannot silently drift apart.

## Testing an implementation

`apyrobo conformance <target>` runs any adapter or wire-protocol server —
in any language — against this spec and emits a machine-readable report.
See [docs/conformance.md](../docs/conformance.md) for the check catalog and
the "APYROBO Conformant" badge program.

## Conformance language

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are to be
interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

## Versioning

The spec is versioned independently of the `apyrobo` Python package.

- **Patch** revisions clarify prose without changing behavior.
- **Minor** revisions add optional fields or messages. Implementations MUST
  ignore unknown fields, so minor revisions are backwards compatible.
- **Major** revisions may change or remove behavior and require a new
  negotiation story.

Version 1.0 has **no wire-level version negotiation** — it describes the
protocol as deployed today. Clients MAY declare the spec version they
implement in `metadata.spec` of any message (e.g. `"spec": "1.0"`); servers
MUST tolerate its absence. A handshake message is reserved for spec 2.0.

## Proposing changes

Changes to anything in `spec/` go through an RFC: open a GitHub issue titled
`RFC: <summary>` describing the motivation, the exact schema/prose diff, and
the compatibility impact. See the roadmap's v8.0.0 Phase 3 for the process
being formalized.

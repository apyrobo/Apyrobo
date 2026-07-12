# Skill Manifest

**Spec version 1.0** · Reference implementation:
[`apyrobo/skills/discovery.py`](../apyrobo/skills/discovery.py)

A skill manifest describes one skill so that agents can discover it at
runtime and match it against a robot's capabilities without importing or
executing the skill's code. Manifests are what the skill registry serves and
what `apyrobo skill search` returns.

## 1. Format

One JSON object per skill ([schema](schemas/skill-manifest.schema.json)):

```json
{
  "name": "navigate_to",
  "version": "1.0.0",
  "description": "Move the robot to target (x, y) coordinates.",
  "parameters": {
    "type": "object",
    "properties": {
      "x": {"type": "number"},
      "y": {"type": "number"},
      "speed": {"type": "number", "default": 0.5}
    },
    "required": ["x", "y"]
  },
  "requirements": ["move"],
  "ros_topics": ["/cmd_vel", "/odom"]
}
```

| Field | Type | Required | Semantics |
|-------|------|----------|-----------|
| `name` | string | yes | Unique within a registry; the verb agents plan with |
| `version` | string | yes | SemVer |
| `description` | string | yes | Shown to LLM planners — write it for a model, one clear sentence |
| `parameters` | object | yes | A JSON Schema (draft 2020-12 subset) for the skill's arguments; also usable directly as an LLM tool/function definition |
| `requirements` | string[] | no (default `[]`) | Capability requirements that must **all** be satisfied for the skill to be available (§2) |
| `ros_topics` | string[] | no | Informational: topics the skill touches. Not used for matching |

Implementations MUST ignore unknown fields (forward compatibility).

## 2. Capability matching

A skill is *available* on a robot iff every entry in `requirements` appears
in the robot's available-capability list. Matching is exact string equality —
there is no wildcard or hierarchy in spec 1.0.

Requirement strings are drawn from a small controlled vocabulary rather than
`CapabilityType` values directly: `move`, `gripper`, `camera`, `arm`,
`voice`. A registry MAY define additional requirement strings; skills using
them simply won't match robots that don't declare them, which fails safe
(skill hidden, never wrongly offered).

An empty `requirements` list means the skill is available on every robot
(e.g. `stop`, `report_status`).

## 3. Versioning and identity

`(name, version)` identifies a manifest. Publishing different content under
an existing `(name, version)` pair is non-conformant; registries MUST reject
it. Consumers SHOULD treat manifests as immutable and cache freely.

# Capability Model

**Spec version 1.0** · Reference implementation:
[`apyrobo/core/schemas.py`](../apyrobo/core/schemas.py)

The capability model is the semantic contract between AI agents and hardware.
Agents plan against *capabilities*, never against ROS 2 topics, vendor SDKs,
or transport details. This indirection is what makes APYROBO robot-agnostic:
a planner that works against `RobotCapability` works against every conformant
adapter.

## 1. Enumerations

All enum values are lowercase strings on the wire.

| Enum | Values | Notes |
|------|--------|-------|
| `CapabilityType` | `navigate`, `rotate`, `pick`, `place`, `scan`, `speak`, `manipulate`, `dock`, `custom` | `custom` carries vendor-specific capabilities; its semantics live in the capability's `description` and `parameters` |
| `SensorType` | `camera`, `lidar`, `imu`, `depth`, `force_torque`, `gps` | |
| `AdapterState` | `disconnected`, `connecting`, `connected`, `error` | |
| `TaskStatus` | `pending`, `started`, `in_progress`, `completed`, `failed`, `aborted` | |
| `RecoveryAction` | `retry`, `reroute`, `escalate`, `abort` | |

New enum values are a **minor** spec revision; consumers MUST treat unknown
values as opaque rather than fail (map unknown `CapabilityType` to `custom`
semantics).

## 2. RobotCapability

The complete profile of one robot, returned by an adapter's
`get_capabilities` ([schema](schemas/robot-capability.schema.json)):

| Field | Type | Required | Semantics |
|-------|------|----------|-----------|
| `robot_id` | string | yes | Unique within the deployment |
| `name` | string | yes | Human-friendly, e.g. `"TurtleBot4-alpha"` |
| `capabilities` | Capability[] | no (default `[]`) | What the robot can do |
| `joints` | JointInfo[] | no | Actuated joints, for arms |
| `sensors` | SensorInfo[] | no | Available sensor streams |
| `max_speed` | number ≥ 0 \| null | no | m/s; planners MUST NOT plan speeds above it |
| `workspace` | object | no | Bounding box / polygon of reachable space |
| `metadata` | object | no | Vendor, firmware, etc. — ignored by planners |

`Capability` entries pair a `capability_type` (enum above) with a `name`
(the skill-level verb, e.g. `navigate_to`), optional `parameters` (a JSON
Schema fragment describing accepted arguments), and a `description`.

**Safety-relevant invariant:** a planner MUST treat the capability list as
exhaustive. If a robot does not declare `pick`, plans containing pick steps
for that robot are invalid and MUST be rejected before execution — this is
what prevents an LLM from hallucinating hardware.

## 3. TaskRequest and TaskResult

`TaskRequest` ([schema](schemas/task-request.schema.json)) is the structured
form of "do this":

- `task_name` (string, required), `parameters` (object)
- `priority` — integer 1–10, higher is more urgent
- `safety_policy` — a `SafetyPolicyRef`: `policy_name`, optional `max_speed`,
  `collision_zones`, `human_proximity_limit` (metres). When both the policy
  and the robot declare `max_speed`, the **lower** value governs.
- `target_robot_id` — a specific robot, or `null` for fleet auto-assignment.

`TaskResult` ([schema](schemas/task-result.schema.json)) reports the outcome:
`task_name`, `status` (`TaskStatus`), `confidence` (0.0–1.0),
`steps_completed` / `steps_total`, optional `error`, and
`recovery_actions_taken` (list of `RecoveryAction`). A result with
`status: "completed"` and `steps_completed < steps_total` is malformed;
implementations MUST NOT emit it.

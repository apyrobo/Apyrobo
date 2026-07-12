# API Stability Guarantees

apyrobo follows [Semantic Versioning](https://semver.org/). This document is
re-baselined against the **v4.x** release series (the current stable line on
PyPI) and defines both the frozen surface and the deprecation policy that
governs how it can ever change.

> **Note on version history:** roadmap milestone numbers (v5.0.0 "Five-Minute
> Success", v6.0.0 "Ecosystem Integrations", …) are planning labels and are
> **not** package versions. The published package version advances by SemVer
> rules only. As of this document, PyPI = `4.x`.

---

## The protocol is stabler than the API

The [`spec/`](spec/) directory defines the APYROBO Protocol — the wire
protocol, capability model, adapter contract, and skill manifest format. The
spec is **frozen at 1.0** (July 2026), versioned independently of the Python
package, and changes only through an accepted RFC. Anything the
spec covers carries a stronger guarantee than the Python API: message shapes
and behavioral contracts survive even refactors that move Python symbols.
When in doubt, depend on the protocol, not the import path.

---

## Stable — no breaking changes within v4.x

No breaking changes to the following public APIs without a major version
bump, and never without a deprecation cycle (see policy below).

### Core (spec-backed)

| Module | Public API |
|--------|-----------|
| `apyrobo.core.robot` | `Robot`, `Robot.discover()` |
| `apyrobo.core.adapters` | `CapabilityAdapter`, `register_adapter()`, `get_adapter()` |
| `apyrobo.core.schemas` | `RobotCapability`, `Capability`, `CapabilityType`, `SensorInfo`, `SensorType`, `JointInfo`, `TaskRequest`, `TaskResult`, `TaskStatus`, `RecoveryAction`, `SafetyPolicyRef`, `AdapterState` |
| `apyrobo.orchestration` | `OrchestrationAdapter`, `OrchestrationServer`, `OrchestrationMessage`, `StdioOrchestrationAdapter`, `WebSocketOrchestrationAdapter` |
| `apyrobo.skills.discovery` | `SkillManifest`, `SkillDiscovery`, `DiscoveryRegistry` |

### Skills & agents

| Module | Public API |
|--------|-----------|
| `apyrobo.skills.executor` | `SkillExecutor`, `execute()`, `@skill_handler` |
| `apyrobo.skills.library` | All built-in skill functions |
| `apyrobo.skills.registry` | `SkillRegistry`, `register()`, `lookup()` |
| `apyrobo.skills.retry` | `RetryStrategy`, `RetryPolicy`, `CircuitBreaker`, `RetryExecutor` |
| `apyrobo.skills.checkpoint` | `CheckpointEntry`, `CheckpointStore`, `CheckpointedExecutor` |
| `apyrobo.skills.failover` | `FailoverPolicy`, `FailoverExecutor`, `SafeStateAction` |
| `apyrobo.skills.rollback` | `RollbackRegistry`, `RollbackExecutor` |
| `apyrobo.skills.feedback` | `FeedbackCollector`, `AdaptiveExecutor`, `ExecutionResult` |
| `apyrobo.skills.plan_validator` | `PlanValidator`, `ValidationResult`, `ValidationIssue` |
| `apyrobo.agents.multiturn` | `MultiTurnAgent`, `ConversationHistory`, `ConversationMessage` |
| `apyrobo.agents.tool_agent` | `ToolCallingAgent`, `SkillTool` |

### Inference & memory

| Module | Public API |
|--------|-----------|
| `apyrobo.inference.router` | `InferenceRouter`, `route()`, `route_vision()` |
| `apyrobo.inference.vlm` | `VLMAdapter`, `MockVLMAdapter`, `LiteLLMVLMAdapter` |
| `apyrobo.memory.episodic` | `EpisodicMemory` |
| `apyrobo.memory.semantic` | `SemanticMemory` |
| `apyrobo.memory.plan_cache` | `PlanCache` |

### Platform

| Module | Public API |
|--------|-----------|
| `apyrobo.config` | `load_config()`, `ApyroboConfig` |
| `apyrobo.auth` | `authenticate()`, `RBACManager`, `RBACRole` |
| `apyrobo.audit` | `AuditTrail`, `AuditEvent` |
| `apyrobo.fleet.manager` | `FleetManager`, `RobotInfo` |
| `apyrobo.api.app` | FastAPI app endpoints (contract) |
| `apyrobo.plugins` | `ApyroboPlugin`, `PluginLoader`, `PluginRegistry` |
| `apyrobo.registry.models` | `SkillPackage`, `SkillVersion` |
| `apyrobo.registry.client` | `SkillRegistryClient` |
| `apyrobo.safety` | `SafetyLayer`, `SafetyRule` |
| `apyrobo.swarm` | `SwarmCoordinator` |
| `apyrobo.observability` | `ObservabilityManager`, metrics/tracing hooks |
| `apyrobo.lts` | `LTSPolicy`, `LTSRelease`, `VersionChecker` |
| `apyrobo.versioning` | `ChangelogParser`, `MigrationGuide`, `APICompatibilityChecker` |

The `apyrobo` CLI's documented commands and exit codes are part of the stable
surface; flag additions are minor, flag removals are major.

---

## Experimental — may change in minor versions

Functional, but the API may change based on community feedback. Pin to a
specific minor version if you depend on them.

| Module | Notes |
|--------|-------|
| `apyrobo.sim` | Gazebo / MuJoCo / Isaac **in-memory stand-ins** (no live simulator connection; live sim = `ros2://`) — APIs evolve |
| `apyrobo.core.unitree_adapter`, `apyrobo.core.isaac_adapter` | Vendor SDK surfaces still settling |
| `apyrobo.voice` | STT/TTS model APIs change with model updates |
| `apyrobo.sensors`, `apyrobo.vision` | Sensor/vision adapter interfaces still being refined |
| `apyrobo.registry.server` | Hosted registry server — endpoint paths may change |
| `apyrobo.dashboard` | Web dashboard routes and markup |
| `apyrobo.coordination` | `TaskBus` / `MultiAgentCoordinator` — new in v4.0 |
| `apyrobo.orchestration.slack_adapter` | Tracks Slack API changes |

---

## Deprecation policy

1. **Announce:** a deprecated symbol emits `DeprecationWarning` on use and is
   listed in the table below and in the CHANGELOG, with its replacement.
2. **Grace period:** deprecated symbols keep working for **at least two minor
   releases and no less than 6 months**, whichever is longer.
3. **Remove:** removal happens only in a major release whose CHANGELOG lists
   it under Breaking Changes, with a migration note in `MIGRATION.md`.
4. Protocol-level changes (anything in `spec/`) additionally require an
   accepted RFC before step 1.

### Currently deprecated

| Symbol | Deprecated in | Earliest removal | Replacement |
|--------|--------------|------------------|-------------|
| *(none)* | | | |

---

## Internal — no stability guarantee

Anything in `apyrobo._internal`, `apyrobo.*.utils`, or modules prefixed with
an underscore (`_`) is private and may change at any time.

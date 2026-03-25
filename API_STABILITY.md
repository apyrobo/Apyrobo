# API Stability Guarantees

apyrobo follows [Semantic Versioning](https://semver.org/).  The stability
level of each module is documented below.

---

## Stable — no breaking changes within v1.x

The following public APIs are **frozen** for the v1.x release series.  No
breaking changes will be made without a major version bump.

| Module | Public API |
|--------|-----------|
| `apyrobo.skills.executor` | `SkillExecutor`, `execute()`, `@skill_handler` |
| `apyrobo.skills.library` | All built-in skill functions |
| `apyrobo.skills.registry` | `SkillRegistry`, `register()`, `lookup()` |
| `apyrobo.skills.feedback` | `FeedbackCollector`, `AdaptiveExecutor`, `ExecutionResult` |
| `apyrobo.skills.discovery` | `SkillManifest`, `DiscoveryRegistry` |
| `apyrobo.agents.multiturn` | `MultiTurnAgent`, `ConversationHistory`, `ConversationMessage` |
| `apyrobo.agents.tool_agent` | `ToolCallingAgent`, `SkillTool` |
| `apyrobo.inference.router` | `InferenceRouter`, `route()` |
| `apyrobo.inference.vlm` | `VLMAdapter`, `MockVLMAdapter`, `LiteLLMVLMAdapter` |
| `apyrobo.memory.episodic` | `EpisodicMemory` |
| `apyrobo.memory.semantic` | `SemanticMemory` |
| `apyrobo.memory.plan_cache` | `PlanCache` |
| `apyrobo.config` | `load_config()`, `ApyroboConfig` |
| `apyrobo.auth` | `authenticate()`, `RBACManager`, `RBACRole` |
| `apyrobo.audit` | `AuditTrail`, `AuditEvent` |
| `apyrobo.fleet.manager` | `FleetManager`, `RobotInfo` |
| `apyrobo.api.app` | FastAPI app endpoints (contract) |
| `apyrobo.plugins.base` | `ApyroboPlugin` |
| `apyrobo.plugins.loader` | `PluginLoader` |
| `apyrobo.plugins.registry` | `PluginRegistry` |
| `apyrobo.registry.models` | `SkillPackage`, `SkillVersion` |
| `apyrobo.registry.client` | `SkillRegistryClient` |
| `apyrobo.lts.policy` | `LTSPolicy`, `LTSRelease` |
| `apyrobo.lts.checker` | `VersionChecker` |
| `apyrobo.versioning` | `ChangelogParser`, `MigrationGuide`, `APICompatibilityChecker` |
| `apyrobo.swarm` | `SwarmCoordinator` |
| `apyrobo.safety` | `SafetyLayer`, `SafetyRule` |
| `apyrobo.observability` | `ObservabilityManager`, metrics/tracing hooks |

---

## Experimental — may change in minor versions

These modules are functional but their API may change based on community
feedback.  Pin to a specific minor version if you depend on them.

| Module | Notes |
|--------|-------|
| `apyrobo.sim` | Gazebo / MuJoCo adapters — hardware APIs evolve |
| `apyrobo.voice` | STT/TTS model APIs change with model updates |
| `apyrobo.sensors` | Sensor adapter interface still being refined |
| `apyrobo.registry.server` | Hosted registry server — endpoint paths may change |

---

## Deprecated — will be removed in v2.0

| Symbol | Replacement |
|--------|------------|
| `apyrobo.memory.MemoryStore` | `apyrobo.memory.semantic.SemanticMemory` |
| `apyrobo.core.adapters.BaseAdapter.connect_sync()` | `BaseAdapter.connect()` (async) |

---

## Internal — no stability guarantee

Anything in `apyrobo._internal`, `apyrobo.*.utils`, or modules prefixed with
an underscore (`_`) is considered private and may change at any time.

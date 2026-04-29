# APYROBO Roadmap

Public roadmap for the APYROBO project. Items are grouped by milestone and roughly ordered by priority within each group.

**Legend:** :white_check_mark: Done | :construction: In Progress | :clipboard: Planned | :bulb: Exploring

**Contribution labels:**
- ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) — Great for newcomers; well-scoped, mentored
- ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) — Community contributions welcome; may require domain expertise

---

## v0.1.0 — Foundation (Current Release)

Core framework with mock adapter support, offline planning, and safety enforcement.

| Status | Item | Description |
|--------|------|-------------|
| :white_check_mark: | Capability adapter pattern | `mock://`, `gazebo://`, `mqtt://`, `http://` URI schemes |
| :white_check_mark: | Skill graph engine | DAG-based skill composition with preconditions/postconditions |
| :white_check_mark: | Skill executor | Sequential + parallel execution with timeout and retry |
| :white_check_mark: | Rule-based agent | Offline planning with no API key required |
| :white_check_mark: | LLM agent | Model-agnostic planning via LiteLLM |
| :white_check_mark: | Safety enforcer | Speed clamping, collision zones, watchdog, escalation |
| :white_check_mark: | Swarm coordination | Task splitting, failure reassignment, deadlock detection |
| :white_check_mark: | Observability | Prometheus metrics, OTel export, alerting, replay |
| :white_check_mark: | State persistence | JSON, SQLite, Redis backends with crash recovery |
| :white_check_mark: | Test suite | 120+ tests including property-based and chaos tests |

---

## v0.2.0 — Handler Registry + Voice

Focus: dynamic skill handler dispatch, voice control, reliability, and real hardware support.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| ✅ | Handler registry | Dynamic `@skill_handler` registration and dispatch — `HandlerRegistry` class with decorator, `dispatch`, introspection ([#1][i1]) | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ | Voice adapter layer | `WhisperAdapter` (offline STT), `PiperAdapter` (offline TTS), `OpenAIVoiceAdapter` (cloud STT+TTS), `MockVoiceAdapter` (tests), `VoiceAgent` (STT→plan→execute→TTS in one call), `WhisperAdapter.transcribe(bytes\|str)` ([#2][i2]) | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ | Nav2 adapter | Real `NavigateToPose` action client with odom pose tracking (`/odom`, BEST_EFFORT QoS), `get_position()`, `cancel_navigation()` via goal handle; stub mode when rclpy unavailable. Matches `fake_turtlebot4.py` interface. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ | MoveIt adapter | `home_arm()`, `get_joint_states()` (live from `/joint_states`), `plan_motion()` / `execute_motion()` separation, `MockMoveItAdapter`; stub mode when rclpy unavailable. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ | Gazebo adapter improvements | Spawn/despawn models, reset world, joint states, apply forces, error handling | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ | Connection resilience | Auto-reconnect with exponential backoff, disconnect/reconnect hooks, observability events | |
| ✅ Done | Skill retry policies | Exponential backoff, jitter, circuit breaker per skill — `RetryStrategy`, `RetryPolicy`, `CircuitBreaker`, `RetryExecutor` in `apyrobo/skills/retry.py` | |
| ✅ Done | Execution checkpointing | Resume skill graphs from last successful step — `CheckpointEntry`, `CheckpointStore`, `CheckpointedExecutor` in `apyrobo/skills/checkpoint.py` | |
| ✅ | Increase test coverage to 90% | Add unit tests for voice, handler registry, and edge cases ([#3][i3]) — **achieved 92% (2076 tests)** | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ | Config file support | YAML/TOML config for policies, adapters, inference — auto-detect format, `from_toml_file()`, `to_toml()` | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |

---

## v0.3.0 — Memory + VLM

Focus: persistent agent memory, vision-language model integration, smarter planning.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| ✅ | Episodic memory | SQLite-backed task execution history — `EpisodicStore`, queryable by time/robot/outcome ([#4][i4]) | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ | Semantic memory | Key-value fact store with cosine similarity vector recall — `SemanticStore` with numpy embeddings | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ Done | VLM integration | Vision-language models for camera-informed planning — `VLMAdapter`, `LiteLLMVLMAdapter`, `MockVLMAdapter`, `VLMRouter`, `InferenceRouter.route_vision()` ([#5][i5]) | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ | Plan caching | TTL-based plan cache with hit/miss metrics — `PlanCache` (memory + SQLite) | |
| ✅ Done | Plan validation | LLM plans checked against capability model before execution — `ValidationIssue`, `ValidationResult`, `PlanValidator` in `apyrobo/skills/plan_validator.py` | |
| ✅ Done | Multi-turn agent | Clarification dialogue when task is ambiguous — `ConversationMessage`, `ConversationHistory`, `MultiTurnAgent` with token-aware context truncation | |
| ✅ Done | Tool-calling agent | Function-calling LLM directly invokes skills — `SkillTool`, `ToolCallingAgent` with mock fallback | |
| ✅ Done | Execution feedback loop | Feed execution results back to planner for re-planning — `ExecutionResult`, `FeedbackCollector`, `AdaptiveExecutor` | |
| ✅ Done | Skill discovery | Agents discover available skills at runtime — `SkillManifest`, `SkillDiscovery`, `DiscoveryRegistry` | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ Done | MuJoCo integration | Lightweight sim adapter for MuJoCo physics — merged PR #38 | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | Formal safety verification | Export safety proofs for regulatory compliance — merged PR #38 | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ | Learning from demonstrations | `DemonstrationRecorder`, `DemonstrationStore` (JSON), `DemonstrationReplayer`, `SkillLearner` (frequency analysis, next-step prediction) in `apyrobo/skills/demonstrations.py` | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |

---

## v0.4.0 — Fleet & Cloud

Focus: multi-robot fleet management, cloud deployment, enterprise features.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| ✅ Done | Fleet manager | Centralized fleet dashboard with task queue — `RobotInfo`, `FleetManager` with load-balanced assignment and offline detection | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | Cloud deployment | Docker Compose / Kubernetes deployment templates — multi-stage `docker/Dockerfile`, `docker/docker-compose.yml`, full `k8s/` manifests with HPA, `docs/deployment.md` | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ Done | REST API gateway | HTTP API for external systems to submit tasks — FastAPI app with task/robot endpoints and API-key auth | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | Role-based access | Per-robot and per-task permission model — `RBACRole`, `ROLE_PERMISSIONS`, `RBACManager` added to `auth.py` | |
| ✅ Done | Audit trail | Immutable log of all commands, decisions, and violations — `AuditEvent`, `AuditTrail` with SHA-256 hash chain integrity verification | |
| ✅ Done | Multi-site support | Coordinate robots across separate physical locations — merged PR #37 | |
| ✅ Done | Edge inference | Run small models on robot hardware for low-latency decisions — merged PR #37 | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | Digital twin sync | Sync physical robot state to simulation in real-time — merged PR #37 | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |

---

## v1.0.0 — Hosted Registry & Stable Release

Focus: API stability, hosted skill/adapter registry, backwards compatibility, comprehensive documentation.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| ✅ | Hosted skill registry | `apyrobo/registry/` — FastAPI server + `SkillRegistryClient` for discovering and publishing skills | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ | API freeze | `API_STABILITY.md` — public API surface frozen for v1.x | |
| ✅ | Migration guide | `MIGRATION.md` — v0.x to v1.0 upgrade documentation | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ | Certification support | `apyrobo/audit.py` — immutable audit trail with SHA-256 hash chain for regulatory compliance | |
| ✅ | Plugin system | `apyrobo/plugins/` — third-party skills and adapters as pip packages via entry-points | |
| ✅ | Long-term support | `apyrobo/lts/` — LTS policy, EOL tracking, `VersionChecker`, security advisories | |
| ✅ | Changelog | `CHANGELOG.md` — complete release history from v0.1.0 to v1.0.0 | |
| ✅ | Versioning tools | `apyrobo/versioning/` — changelog parser, migration guide generator, deprecated API scanner | |

---

## Where to Start Contributing

### Good First Issues

These items are well-scoped, have clear acceptance criteria, and include mentoring:

| Item | Milestone | Issue |
|------|-----------|-------|
| ✅ Increase test coverage to 90% | v0.2.0 | [#3][i3] |
| Document memory system APIs | v0.3.0 | [#4][i4] |
| ✅ Add Kubernetes deployment template | v0.4.0 | — |
| Write v0.x → v1.0 migration guide | v1.0.0 | — |
| Create new skill package (e.g. patrol, inspection) | v0.2.0 | [#7][i7] |

### Help Wanted

These items need domain expertise (ROS 2, simulation, speech, computer vision):

| Item | Milestone | Issue |
|------|-----------|-------|
| Voice adapter — Whisper STT + Piper TTS | v0.2.0 | [#2][i2] |
| VLM integration — camera-informed planning | v0.3.0 | [#5][i5] |
| MuJoCo simulation adapter | v0.3.0 | [#6][i6] |
| Formal safety verification proofs | v0.3.0 | — |
| Nav2 / MoveIt adapter | v0.2.0 | — |

### How to Pick Up an Item

1. Check the [issues list](https://github.com/apyrobo/apyrobo/issues) for the matching issue
2. Comment on the issue to claim it
3. Discuss your approach before writing code
4. Submit a PR against `main` with tests and docs
5. See [CONTRIBUTING.md](CONTRIBUTING.md) for code style and setup instructions

### Suggest New Items

Have an idea not on this roadmap? Open a **Feature Request** issue with:
- **Use case:** What problem does it solve?
- **Proposed approach:** How would you implement it?
- **Alternatives:** What else did you consider?

### Priority Requests

If a planned item is critical for your use case, comment on or :+1: the relevant issue. Community demand influences prioritization.

---

## Non-Goals

Things APYROBO intentionally does **not** aim to do:

- **Replace ROS 2** — APYROBO builds on ROS 2, it doesn't replace the middleware, drivers, or navigation stack
- **Be a simulator** — Use Gazebo, Isaac Sim, or Webots for simulation; APYROBO connects to them via adapters
- **Train models** — APYROBO orchestrates pre-trained LLMs; training happens elsewhere
- **Hardware drivers** — Write ROS 2 drivers for new hardware; APYROBO adapts the semantic layer above them

---

<!-- Issue references — run scripts/create_roadmap_issues.sh then update numbers -->
[i1]: https://github.com/apyrobo/apyrobo/issues
[i2]: https://github.com/apyrobo/apyrobo/issues
[i3]: https://github.com/apyrobo/apyrobo/issues
[i4]: https://github.com/apyrobo/apyrobo/issues
[i5]: https://github.com/apyrobo/apyrobo/issues
[i6]: https://github.com/apyrobo/apyrobo/issues
[i7]: https://github.com/apyrobo/apyrobo/issues

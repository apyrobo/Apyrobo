# Changelog

All notable changes to apyrobo are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and apyrobo adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-04-29

First stable release. Covers all work from PRs #32–#45.

### Added

**Real-hardware ROS 2 support**
- **ROS 2 bridge** (`apyrobo/core/ros2_bridge.py`) — production `ROS2Adapter` with Nav2 `NavigateToPose` action client, cmd_vel proportional-control fallback, BEST_EFFORT odometry subscription, configurable QoS, namespace support, feedback hooks, SLAM trigger, multi-floor map switching (PRs #32–#34)
- **Nav2 adapter** (`apyrobo/nav2.py`) — `Nav2Adapter` / `MockNav2Adapter` with `get_position()`, `cancel_navigation()`, `set_initial_pose()`, stub mode without rclpy (#35)
- **MoveIt 2 adapter** (`apyrobo/moveit.py`) — `MoveItAdapter` / `MockMoveItAdapter` with `plan_motion()` / `execute_motion()` separation, `home_arm()`, live `get_joint_states()` from `/joint_states`, stub mode (#35)

**Voice control**
- **Voice adapters** (`apyrobo/voice.py`) — `WhisperAdapter` (offline STT), `PiperAdapter` (offline TTS), `OpenAIVoiceAdapter` (cloud STT+TTS), `MockVoiceAdapter`, `VoiceAgent` (STT → plan → execute → TTS in one call), `WhisperAdapter.transcribe(bytes|str)` (#36)

**Agent and skill improvements**
- **`@skill` decorator** (`apyrobo/skills/decorators.py`) — annotate plain functions as skills; `Skill.simple()` factory; `SkillLibrary.from_decorated()` auto-registers decorated functions as runtime handlers (#43)
- **Skill discovery** (`apyrobo/skills/discovery.py`) — `SkillManifest`, `SkillDiscovery`, `DiscoveryRegistry` with robot capability matching (#40)
- **Skill retry policies** (`apyrobo/skills/retry.py`) — `RetryStrategy`, `RetryPolicy`, `CircuitBreaker`, `RetryExecutor` (#38)
- **Execution checkpointing** (`apyrobo/skills/checkpoint.py`) — `CheckpointedExecutor` resumes skill graphs from last successful step (#38)
- **Learning from demonstrations** (`apyrobo/skills/demonstrations.py`) — `DemonstrationRecorder`, `DemonstrationStore`, `DemonstrationReplayer`, `SkillLearner` (#41)

**Simulation**
- **MuJoCo adapter** (`apyrobo/sim/`) — lightweight physics sim adapter (#38)
- **Formal safety verification** — export safety proofs for regulatory compliance (#38)

**Fleet and cloud**
- **Fleet manager** (`apyrobo/fleet/`) — load-balanced task assignment, heartbeat monitoring, offline detection (#37)
- **REST API gateway** (`apyrobo/api/`) — FastAPI task/robot endpoints with API-key auth (#37)
- **Role-based access** (`apyrobo/auth.py`) — `RBACRole`, `ROLE_PERMISSIONS`, `RBACManager` (#37)
- **Audit trail** (`apyrobo/audit.py`) — immutable SQLite event log with SHA-256 hash-chain integrity (#37)
- **Multi-site and edge inference** support (#37)
- **Digital twin sync** — real-time state mirroring to simulation (#37)
- **Kubernetes and Docker Compose** deployment templates (`k8s/`, `docker/`) (#37)

**Developer experience**
- **Handler registry** — `@skill_handler` decorator for dynamic dispatch (#33)
- **YAML / TOML config** — file-based adapter and policy configuration (#34)
- **Connection resilience** — exponential-backoff auto-reconnect in base adapter (#34)
- **Integration test suite** — `tests/integration/` with `fake_turtlebot4.py` ROS 2 stub, Docker Compose `integration` profile, GitHub Actions workflow (#44)
- **Plugin system** (`apyrobo/plugins/`) — installable third-party bundles via entry-points (#45)
- **Skill registry server** (`apyrobo/registry/`) — FastAPI registry + `SkillRegistryClient` (#45)
- **Versioning tools** (`apyrobo/versioning/`) — changelog parser, migration guide generator, deprecated API scanner (#45)
- **LTS policy** (`apyrobo/lts/`) — EOL tracking, security advisory lookup (#45)
- Runnable example scripts (`examples/`) — hello robot, custom skill, fleet (#42)
- `MIGRATION.md`, `API_STABILITY.md`, `docs/TURTLEBOT4.md` reference documents (#42, #44)

### Changed
- Public API surface frozen for v1.x (see `API_STABILITY.md`)
- Test coverage: 92 % across 2 076 tests (#33–#36)
- `Development Status` classifier promoted from Alpha → Production/Stable

---

## [0.4.0] - 2026-03-23

### Added
- **REST API gateway** (`apyrobo/api/`) — FastAPI endpoints for task submission, robot listing, and skill execution with `X-API-Key` authentication
- **Fleet manager** (`apyrobo/fleet/`) — load-balanced task assignment, heartbeat monitoring, offline robot detection
- **Audit trail** (`apyrobo/audit.py`) — immutable SQLite event log with SHA-256 cryptographic hash chaining and `verify_chain()` integrity check
- **RBAC** — `RBACRole`, `ROLE_PERMISSIONS`, and `RBACManager` added to `apyrobo/auth.py` (admin / operator / viewer roles)

---

## [0.3.0] - 2026-03-23

### Added
- **VLM integration** (`apyrobo/inference/vlm.py`) — `VLMAdapter` base class, `LiteLLMVLMAdapter` (GPT-4V / Claude Vision), `MockVLMAdapter` for testing
- **Multi-turn agent** (`apyrobo/agents/multiturn.py`) — `ConversationHistory` with token-budget truncation, `MultiTurnAgent` with mock-LLM fallback
- **Tool-calling agent** (`apyrobo/agents/tool_agent.py`) — converts apyrobo skills into LLM tool definitions; handles parallel tool calls
- **Execution feedback loop** (`apyrobo/skills/feedback.py`) — `FeedbackCollector` with rolling success-rate tracking, `AdaptiveExecutor` with dynamic retry strategy
- **Skill discovery** (`apyrobo/skills/discovery.py`) — `SkillManifest`, `SkillDiscovery`, `DiscoveryRegistry` with robot capability matching
- **Memory system** (`apyrobo/memory/`) — episodic memory (SQLite), semantic memory (numpy cosine similarity), TTL plan cache
- **Handler registry** (`apyrobo/skills/registry.py`) — `@skill_handler` decorator for dynamic dispatch
- **YAML/TOML config** — file-based robot/skill/agent configuration support
- **Connection resilience** — exponential-backoff auto-reconnect in base adapter
- **Gazebo improvements** — spawn/despawn models, joint states, apply forces, world reset

---

## [0.2.0] - 2026-03-01

### Added
- Voice adapter (Whisper STT, OpenAI TTS, Piper TTS)
- Nav2 adapter for ROS 2 navigation
- MoveIt adapter for arm manipulation
- Skill execution checkpointing
- Retry and circuit-breaker patterns

---

## [0.1.0] - 2026-01-15

### Added
- Initial release — apyrobo MVP
- Skill graph, SkillExecutor, safety layer, swarm coordinator
- Observability (metrics, traces), state persistence
- REST API for operations (task submission, scheduling)
- Preflight checker, pilot quickstart guide
- Test coverage: 92% (2076 tests)

# Changelog

All notable changes to apyrobo are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and apyrobo adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-04-01

### Added
- **Plugin system** (`apyrobo/plugins/`) — installable third-party skill and adapter bundles via `apyrobo.plugins` setuptools entry-point group
- **Skill registry** (`apyrobo/registry/`) — FastAPI-based hosted registry for discovering and publishing skills; `SkillRegistryClient` for programmatic access
- **Versioning tools** (`apyrobo/versioning/`) — changelog parser, migration guide generator, and deprecated API compatibility checker
- **LTS policy** (`apyrobo/lts/`) — long-term support release definitions, EOL tracking, security advisory lookup
- `CHANGELOG.md`, `MIGRATION.md`, `API_STABILITY.md` reference documents

### Changed
- Public API surface frozen for v1.x (see `API_STABILITY.md`)

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

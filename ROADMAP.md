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
| :construction: | Voice adapter layer | STT/TTS integration — Whisper, Piper, OpenAI ([#2][i2]) | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :construction: | Nav2 adapter | Full ROS 2 Nav2 integration for real navigation stacks | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :construction: | MoveIt adapter | ROS 2 MoveIt 2 integration for manipulation | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
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
| :bulb: | Learning from demonstrations | Record human teleoperation as new skills | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |

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

## v1.1.0 — Ship & Discover

Focus: get apyrobo into developers' hands. Today the package version is still `0.1.0` and `pip install apyrobo` returns nothing. This milestone closes that gap and gives real-robot users a first-class diagnostic experience.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| :clipboard: | **PyPI publish** | Bump version to `1.0.0`, publish wheel to PyPI so `pip install apyrobo` works. Wire `python-package.yml` to fire on version tags. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| :clipboard: | **`apyrobo doctor`** | CLI command that checks the local environment: Python version, rclpy availability, ROS_DOMAIN_ID, reachable adapters, API keys. Prints a pass/fail checklist with fix suggestions. Single biggest DX win for real-robot onboarding. | |
| :clipboard: | **Grafana dashboard** | Add a pre-built Grafana dashboard (`docker/grafana/`) wired to the existing Prometheus metrics. Launch with `docker compose --profile observability up`. Covers task throughput, skill latency, fleet status, error rates. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :clipboard: | **Version alignment** | All version strings (`pyproject.toml`, `apyrobo/__version__.py`, `CHANGELOG.md`, Docker image tags) should reflect the same value. Create a `scripts/bump_version.sh` helper. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| :clipboard: | **README badges** | Add CI status, PyPI version, coverage, and license badges to README. Makes the project look alive to anyone landing on GitHub. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| :clipboard: | **Worktree cleanup script** | `scripts/clean_worktrees.sh` — prune the 50+ leftover `.claude/worktrees/` branches and merged remote refs that accumulate during development. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |

---

## v1.2.0 — Real Robot Hardening

Focus: make the `ros2://` path reliable enough that a developer can stake production work on it. The integration test proves it works once; this milestone proves it keeps working.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| :clipboard: | **Connection health monitor** | Detect `/odom` timeouts, auto-reconnect with exponential backoff, emit `robot.disconnected` / `robot.reconnected` events. Today a silent network drop leaves the adapter stuck. | |
| :clipboard: | **`apyrobo connect --verify`** | `apyrobo connect ros2://turtlebot4 --verify` — one command that connects, reads battery + position + velocity, prints a latency measurement, and exits 0/1. Replaces the current "write a script and guess" workflow. | |
| :clipboard: | **Nav2 costmap awareness** | Pass the Nav2 costmap to the planner so it can reject goals inside obstacles before sending them, rather than waiting for Nav2 to fail. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :clipboard: | **Multi-robot task handoff** | When a robot fails mid-task, automatically reassign the remaining steps to the next available robot in the fleet. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :clipboard: | **Diagnostics export** | `apyrobo diagnose --robot ros2://turtlebot4 --out diag.json` — capture adapter state, last N tasks, error history, and hardware readings into a portable file for sharing with maintainers. | |
| :clipboard: | **Hardware-in-the-loop CI** | GitHub Actions workflow that runs the integration suite against a real TurtleBot4 on a self-hosted runner. Triggered on release tags only. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |

---

## v1.3.0 — Skill Ecosystem

Focus: make it worth publishing skills. Right now the registry infrastructure exists but there are zero community skills and no way to test a skill in isolation.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| :clipboard: | **`apyrobo-skills-turtlebot4`** | A pip-installable skill package for common TurtleBot4 tasks: `patrol_area`, `dock`, `undock`, `follow_person`, `inspect_room`. The reference implementation for third-party skill packages. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :clipboard: | **Skill test harness** | `apyrobo test-skill my_skill.py --robot mock://` — runs a skill against a mock robot, checks preconditions/postconditions, prints a test report. Lowers the bar for contributing skills. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| :clipboard: | **Hosted skill registry** | Deploy the `apyrobo/registry/` FastAPI server publicly (e.g. `registry.apyrobo.dev`) so `apyrobo skill search patrol` actually returns results. | |
| :clipboard: | **Skill composition CLI** | `apyrobo skill compose` — interactive REPL for chaining skills into a graph and testing the result, without writing Python. | :bulb: |

---

## v2.0.0 — Adaptive Intelligence

Focus: the planner gets smarter. Today the LLM produces a static plan and retries on failure; this milestone makes it actually adapt.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| :clipboard: | **LLM replanning loop** | When a skill fails, send the failure reason back to the LLM and ask it to replan the remaining steps. Not just retry — actually reconsider the approach. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :clipboard: | **VLM task verification** | After a skill completes, use the VLM adapter to check camera feed and confirm the expected state ("is the cup actually picked up?"). Flag discrepancies for human review or replan. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :clipboard: | **Long-horizon planning** | Break a multi-step, multi-hour goal ("restock all shelves") into a plan that spans robots, shifts, and recharges. Requires persistent plan state and checkpoint recovery. | :bulb: |
| :clipboard: | **Correction learning** | When a user overrides a plan step, record the correction and use it to bias future planning for similar tasks. Builds on the existing demonstrations system. | :bulb: |
| :clipboard: | **Sim-to-real transfer** | `apyrobo plan --simulate` — run a full task plan in Gazebo first, report any failures, then optionally deploy to the real robot. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |

---

## Where to Start Contributing

### Good First Issues

These items are well-scoped, have clear acceptance criteria, and include mentoring:

| Item | Milestone | Issue |
|------|-----------|-------|
| ✅ Increase test coverage to 90% | v0.2.0 | [#3][i3] |
| ✅ Add Kubernetes deployment template | v0.4.0 | — |
| PyPI publish + version bump to 1.0.0 | v1.1.0 | — |
| README badges (CI, PyPI, coverage) | v1.1.0 | — |
| Worktree cleanup script | v1.1.0 | — |
| Skill test harness (`apyrobo test-skill`) | v1.3.0 | — |
| `apyrobo-skills-turtlebot4` package | v1.3.0 | [#7][i7] |

### Help Wanted

These items need domain expertise (ROS 2, simulation, speech, computer vision):

| Item | Milestone | Issue |
|------|-----------|-------|
| Grafana dashboard provisioning | v1.1.0 | — |
| Nav2 costmap awareness | v1.2.0 | — |
| Hardware-in-the-loop CI (self-hosted runner) | v1.2.0 | — |
| LLM replanning loop | v2.0.0 | — |
| VLM task verification | v2.0.0 | [#5][i5] |
| Sim-to-real transfer | v2.0.0 | [#6][i6] |

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

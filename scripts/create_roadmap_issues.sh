#!/usr/bin/env bash
# Create GitHub issues matching ROADMAP.md items tagged "good first issue".
# Run this script after authenticating with `gh auth login`.
#
# Usage:
#   gh auth login
#   bash scripts/create_roadmap_issues.sh

set -euo pipefail

REPO="apyrobo/apyrobo"

echo "Creating good-first-issue GitHub issues for ROADMAP.md..."

gh issue create --repo "$REPO" \
  --title "Increase test coverage to 90% (voice, handlers, edge cases)" \
  --label "good first issue" \
  --body "$(cat <<'BODY'
## Roadmap Reference
**Milestone:** v0.2.0 — Handler Registry + Voice
**ROADMAP.md item:** Increase test coverage to 90%

## Description
Current test coverage is good (120+ tests) but we want to reach 90% across the codebase. Key areas that need more coverage:

- `apyrobo/voice.py` — voice adapter layer (WhisperAdapter, PiperAdapter, OpenAIVoiceAdapter)
- `apyrobo/skills/handlers.py` — dynamic `@skill_handler` registration and dispatch
- `apyrobo/memory.py` — episodic and semantic memory edge cases (TTL expiry, search)
- `apyrobo/auth.py` — role-based access control edge cases

## Acceptance Criteria
- [ ] Add tests for `voice.py` covering all adapter types
- [ ] Add tests for `handlers.py` covering handler registration, lookup, and error cases
- [ ] Add tests for edge cases in `memory.py` (TTL expiry, empty search results)
- [ ] Overall coverage reaches 90% as measured by `pytest --cov`

## Getting Started
```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=apyrobo --cov-report=term-missing
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for code style guidelines.
BODY
)"

gh issue create --repo "$REPO" \
  --title "Create new skill package: patrol pattern" \
  --label "good first issue" \
  --body "$(cat <<'BODY'
## Roadmap Reference
**Milestone:** v0.2.0 — Handler Registry + Voice
**ROADMAP.md item:** New skill packages

## Description
Create a reusable skill package for patrol patterns. This involves writing JSON skill definitions and a simple handler that composes existing built-in skills (navigate_to, rotate) into patrol routes.

## Acceptance Criteria
- [ ] Create `skills/patrol/` directory with skill JSON definitions
- [ ] Implement at least 2 patrol patterns: waypoint loop, perimeter sweep
- [ ] Each skill has preconditions/postconditions defined
- [ ] Add tests in `tests/test_skills/test_patrol.py`
- [ ] Add brief usage example in the skill package README

## Getting Started
Look at existing skill packages in `skills/builtin-skills/` for the JSON format.
See [Skill Authoring Guide](docs/skill_authoring.md) for detailed instructions.
See [CONTRIBUTING.md](CONTRIBUTING.md) for code style guidelines.
BODY
)"

gh issue create --repo "$REPO" \
  --title "Document memory system APIs (episodic + semantic)" \
  --label "good first issue" \
  --body "$(cat <<'BODY'
## Roadmap Reference
**Milestone:** v0.3.0 — Memory + VLM
**ROADMAP.md item:** Episodic + semantic memory

## Description
The memory system (`apyrobo/memory.py`) implements episodic and semantic memory but lacks dedicated documentation. Write a guide covering usage, configuration, and integration patterns.

## Acceptance Criteria
- [ ] Add `docs/memory_guide.md` covering:
  - EpisodicMemory: recording tasks, searching episodes, persistence
  - SemanticMemory: storing facts, TTL-based expiry, querying
- [ ] Add code examples showing how to use memory with an Agent
- [ ] Link the new guide from `docs/api_reference.md`

## Getting Started
Read `apyrobo/memory.py` and `tests/test_memory.py` to understand the current API.
See [CONTRIBUTING.md](CONTRIBUTING.md) for code style guidelines.
BODY
)"

gh issue create --repo "$REPO" \
  --title "Add Gazebo adapter improvements: spawn/delete models, reset world" \
  --label "good first issue" \
  --body "$(cat <<'BODY'
## Roadmap Reference
**Milestone:** v0.2.0 — Handler Registry + Voice
**ROADMAP.md item:** Gazebo adapter improvements

## Description
The current `GazeboAdapter` and `GazeboNativeAdapter` support basic operations. Extend them with model lifecycle management (spawn, delete) and world reset functionality.

## Acceptance Criteria
- [ ] Add `spawn_model(name, sdf_path, pose)` method to GazeboNativeAdapter
- [ ] Add `delete_model(name)` method
- [ ] Add `reset_world()` method to restore initial state
- [ ] Add sensor stream subscription helpers
- [ ] Add unit tests for all new methods
- [ ] Update `docs/adapter_authoring.md` with Gazebo-specific notes

## Getting Started
See `apyrobo/sim/adapters.py` for the current Gazebo adapter implementation.
See `apyrobo/core/adapters.py` for the `CapabilityAdapter` ABC.
See [CONTRIBUTING.md](CONTRIBUTING.md) for code style guidelines.
BODY
)"

gh issue create --repo "$REPO" \
  --title "Add YAML/TOML config file support for policies and adapters" \
  --label "good first issue" \
  --body "$(cat <<'BODY'
## Roadmap Reference
**Milestone:** v0.2.0 — Handler Registry + Voice
**ROADMAP.md item:** Config file support

## Description
Currently configuration is done programmatically via `ApyroboConfig`. Add support for loading configuration from YAML and TOML files, covering safety policies, adapter settings, and inference routing.

## Acceptance Criteria
- [ ] Support loading config from `apyrobo.yaml` or `apyrobo.toml`
- [ ] Config file can specify: adapter URIs, safety policy overrides, inference tier preferences
- [ ] Add example config files in `config/` directory
- [ ] Add tests for config loading, validation, and error handling
- [ ] Document config file format in a new `docs/configuration.md`

## Getting Started
See `apyrobo/config.py` for the existing `ApyroboConfig` class.
See `config/` directory for existing templates.
See [CONTRIBUTING.md](CONTRIBUTING.md) for code style guidelines.
BODY
)"

echo ""
echo "Done! Created 5 issues tagged 'good first issue'."
echo "Run 'gh issue list --repo $REPO --label \"good first issue\"' to verify."

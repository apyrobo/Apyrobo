<!--
  Pre-filed good-first-issue and help-wanted issues for LN-03.
  Create these issues on GitHub after pushing:

  gh issue create --title "Add YAML/TOML config file support" \
    --label "good first issue" --body "..."
-->

# Good First Issues (to be created on GitHub)

Below are 5 issues to create manually or via `gh issue create`.
Run the script at the bottom to create them all at once.

---

## 1. Add YAML/TOML config file support for policies and adapters

**Labels:** `good first issue`
**Roadmap:** v0.2 — Production Hardening

Currently, policies, adapters, and inference settings are configured in Python code. Add a YAML/TOML config loader so users can configure APYROBO without modifying source.

**Acceptance criteria:**
- Load adapter config from `apyrobo.yml` or `apyrobo.toml`
- Support safety policy overrides (speed limits, collision zones)
- Support inference router settings (model, fallback, token budget)
- Fallback to defaults when no config file is present
- Tests for config loading and validation

---

## 2. Improve test coverage to 90%+

**Labels:** `good first issue`
**Roadmap:** v0.1 — Foundation

The test suite has 120+ tests but some modules have low coverage. Identify uncovered paths and add tests.

**Acceptance criteria:**
- Run `pytest --cov=apyrobo` and identify modules below 80%
- Add tests for uncovered branches in `apyrobo/core/`, `apyrobo/skills/`, `apyrobo/safety/`
- No decrease in existing coverage

---

## 3. Add runtime skill discovery for agents

**Labels:** `good first issue`
**Roadmap:** v0.3 — Intelligence

Agents currently use a hard-coded skill list. Add a discovery mechanism so agents can query available skills at runtime.

**Acceptance criteria:**
- Agent can list all registered skills and their schemas
- Agent can filter skills by capability requirements
- Works with both rule-based and LLM agents
- Unit tests for discovery logic

---

## 4. Add Docker Compose / Kubernetes deployment templates

**Labels:** `good first issue`
**Roadmap:** v0.4 — Fleet & Cloud

Create production-ready deployment templates for cloud hosting.

**Acceptance criteria:**
- Docker Compose file for production (non-dev) deployment
- Basic Kubernetes manifests (Deployment, Service, ConfigMap)
- Documentation in `docs/deployment.md`

---

## 5. Add docstrings to public APIs in core and skills modules

**Labels:** `good first issue`
**Roadmap:** v0.1 — Foundation

Several public classes and functions in `apyrobo/core/` and `apyrobo/skills/` are missing or have incomplete docstrings.

**Acceptance criteria:**
- All public classes in `apyrobo/core/` have docstrings
- All public classes in `apyrobo/skills/` have docstrings
- Docstrings follow Google style
- No code logic changes

---

## Create all issues at once

```bash
# Run from the repo root:
gh issue create --title "Add YAML/TOML config file support for policies and adapters" \
  --label "good first issue" \
  --body "**Roadmap:** v0.2 — Production Hardening

Currently policies, adapters, and inference settings are configured in Python code. Add a YAML/TOML config loader so users can configure APYROBO without modifying source.

**Acceptance criteria:**
- [ ] Load adapter config from \`apyrobo.yml\` or \`apyrobo.toml\`
- [ ] Support safety policy overrides (speed limits, collision zones)
- [ ] Support inference router settings (model, fallback, token budget)
- [ ] Fallback to defaults when no config file is present
- [ ] Tests for config loading and validation

See [ROADMAP.md](../blob/main/ROADMAP.md) for context."

gh issue create --title "Improve test coverage to 90%+" \
  --label "good first issue" \
  --body "**Roadmap:** v0.1 — Foundation

The test suite has 120+ tests but some modules have low coverage. Identify uncovered paths and add tests.

**Acceptance criteria:**
- [ ] Run \`pytest --cov=apyrobo\` and identify modules below 80%
- [ ] Add tests for uncovered branches
- [ ] No decrease in existing coverage

See [ROADMAP.md](../blob/main/ROADMAP.md) for context."

gh issue create --title "Add runtime skill discovery for agents" \
  --label "good first issue" \
  --body "**Roadmap:** v0.3 — Intelligence

Agents currently use a hard-coded skill list. Add a discovery mechanism so agents can query available skills at runtime.

**Acceptance criteria:**
- [ ] Agent can list all registered skills and their schemas
- [ ] Agent can filter skills by capability requirements
- [ ] Works with both rule-based and LLM agents
- [ ] Unit tests for discovery logic

See [ROADMAP.md](../blob/main/ROADMAP.md) for context."

gh issue create --title "Add Docker Compose / Kubernetes deployment templates" \
  --label "good first issue" \
  --body "**Roadmap:** v0.4 — Fleet & Cloud

Create production-ready deployment templates for cloud hosting.

**Acceptance criteria:**
- [ ] Docker Compose file for production (non-dev) deployment
- [ ] Basic Kubernetes manifests (Deployment, Service, ConfigMap)
- [ ] Documentation in \`docs/deployment.md\`

See [ROADMAP.md](../blob/main/ROADMAP.md) for context."

gh issue create --title "Add docstrings to public APIs in core and skills modules" \
  --label "good first issue" \
  --body "**Roadmap:** v0.1 — Foundation

Several public classes and functions in \`apyrobo/core/\` and \`apyrobo/skills/\` are missing or have incomplete docstrings.

**Acceptance criteria:**
- [ ] All public classes in \`apyrobo/core/\` have docstrings
- [ ] All public classes in \`apyrobo/skills/\` have docstrings
- [ ] Docstrings follow Google style
- [ ] No code logic changes

See [ROADMAP.md](../blob/main/ROADMAP.md) for context."
```

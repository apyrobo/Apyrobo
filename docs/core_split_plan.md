# apyrobo-core Split — Analysis & Plan

Status: analysis complete, implementation scheduled (v8.0.0 Phase 1).
This document records the measurements and the decision so the refactor PR
can be reviewed against explicit targets.

## Roadmap targets

From ROADMAP.md v8 Phase 1: install < 15 s, `import apyrobo` < 500 ms,
core hard dependencies ≤ 5.

## Measurements (2026-07-04, Python 3.12, arm64 macOS)

| Metric | Value | Verdict |
|--------|-------|---------|
| `import apyrobo` | **0.09 s** | ✅ already met — the top-level `__init__` is lazy; the heaviest module chain is `core.schemas` → pydantic at ~56 ms |
| Declared hard deps | `pydantic`, `litellm`, `networkx`, `pyyaml` | ❌ two of the four don't earn their place (below) |
| `litellm` install footprint | **77 MB** package alone, plus a large transitive tree (tokenizers, tiktoken, httpx, …) — dominates install time | it is the install-time problem |
| `networkx` usage | **zero imports anywhere in `apyrobo/`** | dead dependency; the skill graph engine is hand-rolled |
| `litellm` usage | function-level (lazy) imports only: `skills/agent.py`, `skills/longterm.py`, `agents/tool_agent.py`, `agents/multiturn.py`, `inference/vlm.py` | can move to an extra with **no architectural change** |
| `pyyaml` usage | `config.py`, `hardware/schema.py`, `core/ros2_bridge.py` | small, keep |
| `pydantic` usage | `core/schemas.py` (which already has a working no-pydantic fallback), `registry/models.py`, `api/app.py` | keep as the one substantial core dep |

## Decision: slim the package, don't split the distribution (yet)

Two options were considered:

**A. Single-package slimming (chosen).** Keep one `apyrobo` distribution;
reduce hard deps to `pydantic` + `pyyaml`; move LLM support behind an extra.

**B. Separate `apyrobo-core` distribution.** Two PyPI packages with
`apyrobo` depending on `apyrobo-core`.

Option A reaches every Phase 1 target with a fraction of the churn: no
namespace migration for users, no version-coupling between two dists, no
doubled release pipeline. Option B buys nothing extra until there are
third-party distributions that want to depend on the kernel *without* the
first-party extras — revisit when the conformance suite and TS client
(Phase 2) create that demand. The spec in `spec/` — not a package boundary —
is the real portability layer.

## Implementation plan (separate PR)

1. **Drop `networkx`** — unused. *(Done in the v8-phase1 branch, full suite green.)*
2. **Move `litellm` to a new `llm` extra.**
   - All imports are already lazy; wrap each in the existing guard pattern
     (see `registry/server.py`) so the error says
     `pip install 'apyrobo[llm]'`.
   - `Agent(provider="auto")` already degrades to rule-based planning when
     the LLM path fails (v4.0 degraded-operation mode); with litellm absent
     it must degrade identically at construction time, with one logged
     `agent.degraded` event.
   - Add `llm` to a new `full` meta-extra: `pip install 'apyrobo[full]'`.
3. **CI guard:** a workflow job that installs bare `apyrobo` (no extras) in a
   clean venv and runs an import + rule-based-planning smoke test, so a
   heavyweight import can never silently become load-bearing again.
4. **Docs:** README install matrix (bare / `[llm]` / `[full]`), MIGRATION.md
   note. Compute profiles that select LLM models must state they require
   `[llm]`.

### Compatibility

Moving a dependency to an extra changes the default install → **minor-version
release with loud CHANGELOG entry** (existing `pip install apyrobo` users who
call LLM planning get a clear install hint, not silent breakage, thanks to
the degraded-mode event + error message). Dropping never-imported `networkx`
is invisible to users.

### Acceptance criteria for the refactor PR

- [x] `pip install apyrobo` in a clean venv pulls ≤ 5 packages beyond
  pydantic's own tree and completes in < 15 s on CI
  *(measured: 2.5 s, 8 packages total, import 353 ms)*
- [x] Bare install: `import apyrobo`, `Robot.discover("mock://x")`, and
  rule-based `Agent.plan()` all work; LLM planning raises with the
  `apyrobo[llm]` hint *(scripts/bare_install_smoke.py, also a CI job)*
- [x] `apyrobo[llm]` behaves exactly as today *(full suite: 3556 passed)*
- [x] Full test suite green in both bare and `[llm]` environments
  *(bare: 3550 passed + 6 litellm-patching tests correctly skipped via
  importorskip; `[llm]`: 3556 passed)*

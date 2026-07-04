# Migration Guide

## Migrating to v4.1.0 from v4.0.x

### LLM support is now an extra: `apyrobo[llm]`

`litellm` (and its large dependency tree) is no longer installed by default.
The bare `pip install apyrobo` now contains only the kernel — adapters,
capability model, skill graph, rule-based planning, safety.

**If you use LLM or VLM planning** (`Agent(provider="llm")`,
`provider="routed"`, `ToolCallingAgent`, `MultiTurnAgent`,
`LiteLLMVLMAdapter`, long-horizon decomposition), install the extra:

```bash
pip install 'apyrobo[llm]'
```

**What happens if you don't:**

- `Agent(provider="auto")` degrades to rule-based planning and emits an
  `agent.degraded` observability event (`reason: litellm_not_installed`) —
  no crash, same behavior as the existing runtime degraded mode.
- `Agent(provider="llm")` raises `RuntimeError` with the install hint at
  planning time.
- `ToolCallingAgent` and `MultiTurnAgent` fall back to their mock providers,
  as they already did.

`pip install 'apyrobo[full]'` installs every optional feature (LLM, REST API,
registry, dashboard, WebSocket, Slack).

Also in this release: the unused `networkx` dependency was removed — it was
never imported.

---

## Migrating to v1.0.0 from v0.x

This document covers breaking changes and migration steps when upgrading to
apyrobo v1.0.0.

### No breaking changes from v0.4.0 → v1.0.0

The v1.0.0 release adds new modules but does **not** remove or rename any
existing public APIs. Upgrading from v0.4.0 is a drop-in update.

---

## Migrating to v0.4.0 from v0.3.x

### REST API `X-API-Key` header required

The new REST gateway (`apyrobo/api/`) requires an `X-API-Key` header on all
requests. If you call the API directly, add the header:

```python
# Before (no auth):
import requests
resp = requests.post("http://localhost:8080/tasks", json={...})

# After:
resp = requests.post(
    "http://localhost:8080/tasks",
    json={...},
    headers={"X-API-Key": "your-api-key"},
)
```

Configure the key in `apyrobo.yaml`:

```yaml
api:
  key: "your-api-key"
```

### Auth module — new RBAC classes

`apyrobo.auth` now exports `RBACRole`, `ROLE_PERMISSIONS`, and `RBACManager`.
Existing code using `apyrobo.auth` is unaffected; these are additive exports.

---

## Migrating to v0.3.0 from v0.2.x

### `apyrobo.memory` module restructured

**Before** — single `memory.py` file:
```python
from apyrobo.memory import MemoryStore
```

**After** — package with episodic / semantic / plan-cache modules:
```python
from apyrobo.memory.episodic import EpisodicMemory
from apyrobo.memory.semantic import SemanticMemory
from apyrobo.memory.plan_cache import PlanCache
```

The legacy `apyrobo.memory.MemoryStore` is still available as a compatibility
shim but will be removed in v2.0.

---

## Migrating to v0.2.0 from v0.1.x

No breaking changes. v0.2.0 is fully backward-compatible with v0.1.0.

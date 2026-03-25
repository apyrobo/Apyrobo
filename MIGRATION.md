# Migration Guide

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

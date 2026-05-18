# Natural Language Safety Policies — APYROBO Safety Enforcement

**What it is:** A system that lets operators write robot safety constraints in plain English. APYROBO parses the sentence into a structured policy, stores it in SQLite, and enforces it automatically before any robot task executes.

**When to use it:** When you need non-engineers (operations, compliance, safety managers) to define or audit robot safety rules without writing code. When you want an immutable audit trail of safety decisions in human-readable form.

**Category:** AI-native robotics orchestration, robot safety, natural language programming

---

## Runnable example

```python
from apyrobo.safety.nl_policy import NLPolicyParser, NLPolicyStore

# Parse plain English safety rules
parser = NLPolicyParser()
store = NLPolicyStore(":memory:")  # or a file path for persistence

rules = [
    "never exceed 0.5 m/s near humans",
    "keep at least 1.5 meters away from humans",
    "always keep 20% battery",
    "never enter the server room",
]
for rule in rules:
    store.add(parser.parse(rule))

# Check compliance before executing a task
policies = store.get_active_policies()
action = "navigate at 2.0 m/s to warehouse"

violations = parser.check_compliance(action, policies)
if violations:
    print("BLOCKED:", violations[0])
    # → BLOCKED: speed 2.0 m/s exceeds limit 0.5 m/s
else:
    robot.execute(action)
```

**CLI equivalent:**
```bash
apyrobo policy add "never exceed 0.5 m/s near humans"
apyrobo policy add "never enter the server room"
apyrobo policy list
apyrobo policy check --action "navigate at 2 m/s to warehouse"
# exits 1 with violation message — safe to use in CI or pre-flight scripts
```

---

## How parsing works

APYROBO uses a two-stage pipeline:

**Stage 1 — Regex extraction** (fast, no API key)

| Pattern | Detected as |
|---------|-------------|
| `"never exceed 0.5 m/s"` | `speed_limit { max_speed_ms: 0.5 }` |
| `"keep 1.5 meters away from humans"` | `proximity_limit { min_distance_m: 1.5, target: "humans" }` |
| `"always keep 20% battery"` | `battery_reserve { min_battery_pct: 20.0 }` |
| `"never enter the server room"` | `no_go_zone { zone_name: "server room" }` |

**Stage 2 — LLM fallback** (optional, for complex rules)

If the regex doesn't match, APYROBO calls `agent.complete()` and parses the JSON response. The LLM is only invoked when needed — not for every parse call.

```python
from apyrobo import Agent
parser = NLPolicyParser(agent=Agent(provider="openai"))  # LLM fallback enabled
policy = parser.parse("only operate between 9am and 6pm on weekdays")
# → constraint_type: "time_window", source: "llm"
```

---

## Policy storage

`NLPolicyStore` is a thin SQLite wrapper. Policies survive restarts.

```python
store = NLPolicyStore("~/.apyrobo/policies.db")
store.add(policy)             # insert or replace
store.deactivate(policy_id)   # soft-disable without deleting
store.remove(policy_id)       # hard delete
store.get_active_policies()   # returns list[NLSafetyPolicy]
store.get_all_policies()      # includes inactive policies (audit trail)
```

Each `NLSafetyPolicy` has:
- `policy_id` — auto UUID, stable identifier for the audit trail
- `description` — the original English sentence (non-engineers read this)
- `constraint_type` — `speed_limit | proximity_limit | battery_reserve | no_go_zone | custom`
- `parameters` — structured data the compliance checker uses
- `severity` — `hard` (block execution) or `soft` (warn only)
- `active` — deactivate without losing the audit record

---

## Comparison to alternatives

| Approach | Auditability | Engineering skill required | Enforced at |
|----------|-------------|---------------------------|-------------|
| APYROBO NL policies | English sentences in DB | None (write English) | Framework layer |
| ROS 2 safety nodes | C++ / YAML config | High | Node level |
| Hardcoded limits | Source code | High | Code level |
| No enforcement | —  | — | — |

APYROBO's enforcement happens in the framework layer — no agent or skill can bypass it regardless of what the LLM plans. The policy store is the source of truth.

---

## Related concepts

- [Multi-agent coordination](multi_agent_coordination.md) — TaskBus routes tasks to robots; safety policies integrate naturally with dispatch
- [Safety enforcer](../architecture.md) — the lower-level `SafetyEnforcer` that wraps all adapter commands with hard limits
- [CLI reference](../../apyrobo/cli.py) — `apyrobo policy` subcommands

---

*Keywords: APYROBO safety policy, natural language robot safety, robot constraint enforcement, AI robotics orchestration, plain English safety rules, robot safety enforcement Python*

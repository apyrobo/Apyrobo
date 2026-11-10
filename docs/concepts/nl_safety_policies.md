# Natural Language Safety Policies — APYROBO Safety Enforcement

**What it is:** A system that lets operators write robot safety constraints in plain English. APYROBO parses the sentence into a structured policy, stores it in SQLite, and checks a written action against the stored policies before you run it (`check_compliance`, or `apyrobo policy check` in a script).

**What it is not:** these policies are not applied to motion. `SafetyEnforcer`, `apyrobo execute` and `apyrobo mcp` do not read the policy store; a robot moves whatever the store says unless your code runs the check first and stops on a violation. For limits that are enforced on every motion command, use `SafetyEnforcer` (see [quickstart](../quickstart_5min.md), Step 4) or `apyrobo mcp --max-speed / --no-go / --policy-file` (see [docs/mcp.md](../mcp.md)).

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
apyrobo policy check "navigate at 2 m/s to warehouse"
# exits 1 with violation message (also with --json) — usable in CI or pre-flight scripts
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

If the regex doesn't match and you gave the parser an `agent`, APYROBO calls `agent.complete(prompt)` and parses the JSON it returns. The LLM is only invoked when needed — not for every parse call. If the call fails or the reply is not JSON, the rule is stored as `custom`.

`agent` is any object with a `complete(prompt) -> str` method. `apyrobo.Agent` has no such method, so passing one raises `TypeError`. A few lines around [litellm](https://docs.litellm.ai/) (`pip install apyrobo[llm]`) are enough:

```python
import litellm
from apyrobo.safety.nl_policy import NLPolicyParser

class LiteLLM:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def complete(self, prompt: str) -> str:
        reply = litellm.completion(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        return reply.choices[0].message.content

parser = NLPolicyParser(agent=LiteLLM())  # LLM fallback enabled
policy = parser.parse("only operate between 9am and 6pm on weekdays")
# → e.g. constraint_type: "time_window", source: "llm" (the model decides)
```

A `time_window` policy is stored and listed, not checked (see the table below).

---

## What the check reads

The check works on the text of the action, not on what a robot will do.

| Policy type | Checked how |
|-------------|-------------|
| `speed_limit` | Every speed written in m/s (`2 m/s`, `2.0m/s`, `2 meters per second`, `0.2 mps`) is compared with the limit; the highest counts. A speed written any other way (`2 km/h`, `30 cm/s`, `speed 2`, `0,3 m/s`) is reported as a violation, because the check cannot tell whether it is over the limit. A bare number with no unit and no "speed" word (`go to (2, 3) at 1.5`) is not read as a speed. |
| `battery_reserve` | A battery level written as `N% battery` is compared with the reserve. |
| `no_go_zone` | A violation when the zone's name appears in the action text. |
| `proximity_limit`, `time_window`, `load_limit`, `custom` | Stored and listed, not checked. |

An action with no speed in it passes a speed limit.

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
| APYROBO NL policies | English sentences in DB | None (write English) | Pre-flight check you run (not applied to motion) |
| ROS 2 safety nodes | C++ / YAML config | High | Node level |
| Hardcoded limits | Source code | High | Code level |
| No enforcement | —  | — | — |

The policy store is a readable record of the rules; the check is only as strong as the code that runs it and stops on a violation. Motion limits that no agent or skill can bypass come from `SafetyEnforcer`, which wraps the robot.

---

## Related concepts

- [Multi-agent coordination](multi_agent_coordination.md) — TaskBus routes tasks to robots; safety policies integrate naturally with dispatch
- [Safety enforcer](../architecture.md) — the lower-level `SafetyEnforcer` that wraps all adapter commands with hard limits
- [CLI reference](../../apyrobo/cli.py) — `apyrobo policy` subcommands

---

*Keywords: APYROBO safety policy, natural language robot safety, robot constraint enforcement, AI robotics orchestration, plain English safety rules, robot safety enforcement Python*

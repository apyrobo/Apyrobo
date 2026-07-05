# Demo: Humanoid Task Delegation with Natural Language Safety

<img src="demo.gif" alt="Demo: Humanoid Task Delegation with Natural Language Safety — terminal recording" width="720">

_Re-render: `./record.sh` (requires [vhs](https://github.com/charmbracelet/vhs))_

Operators write safety rules in plain English. APYROBO parses them, stores them
in SQLite, and enforces them before any task executes on the robot.

```bash
pip install apyrobo
python demo.py
```

**What you'll see:** 5 safety rules loaded from plain English strings. 6 tasks run
through the compliance checker — speed violations, no-go zones, and low battery are
all caught automatically. A new policy is added at runtime and immediately blocks
the task it targets.

**Key APYROBO APIs used:**
- `NLPolicyParser.parse("never exceed 0.3 m/s near humans")` — regex → policy
- `NLPolicyStore(":memory:")` — SQLite-backed policy persistence
- `parser.check_compliance(action, policies)` — returns list of violation strings
- Runtime updates: `store.add(policy)` takes effect on the next check

**Policy types supported:**
| Rule example | Detected as |
|---|---|
| `"never exceed 0.5 m/s near humans"` | `speed_limit` |
| `"keep 1.5 meters away from obstacles"` | `proximity_limit` |
| `"always keep 20% battery"` | `battery_reserve` |
| `"never enter the server room"` | `no_go_zone` |
| anything else | `custom` (LLM fallback) |

**CLI equivalent:**
```bash
apyrobo policy add "never exceed 0.3 m/s near humans"
apyrobo policy check --action "navigate at 2 m/s"
# → exits 1 with violation message
```

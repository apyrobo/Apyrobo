# Multi-Agent Coordination — APYROBO TaskBus

**What it is:** A thread-safe task bus that lets multiple AI agents coordinate over shared robots. One call to `bus.dispatch("pick cup", required_capability="PICK")` routes the task to the best available agent. `bus.broadcast("emergency_stop")` fans out to every agent simultaneously.

**When to use it:** When you have more tasks than one robot can handle, when different robots have different capabilities (pickers, packers, haulers), or when you want a fleet to collaborate on a complex goal without writing custom pub/sub code.

**Category:** AI-native robotics orchestration, multi-robot coordination, robot fleet management, AI agent coordination

---

## Runnable example

```python
from apyrobo import Agent, MockAdapter, Robot
from apyrobo.coordination.bus import TaskBus, MultiAgentCoordinator

# Create a shared task bus
bus = TaskBus(timeout=10.0)

# Add robots to the fleet
fleet = [
    ("picker_bot", ["PICK", "NAVIGATE"]),
    ("packer_bot", ["PLACE", "NAVIGATE"]),
    ("hauler_bot", ["NAVIGATE"]),
]
coordinators = []
for name, caps in fleet:
    robot = Robot(f"mock://{name}", MockAdapter(name))
    coord = MultiAgentCoordinator(
        Agent(provider="rule"), robot, bus,
        agent_id=name, capabilities=caps,
    )
    coord.start()
    coordinators.append(coord)

# Route tasks by capability — no routing code needed
result = bus.dispatch("pick_object", required_capability="PICK")
print(f"Handled by: {result.agent_id}")  # → picker_bot

# Fan out to all agents (e.g., emergency stop)
results = bus.broadcast("navigate_to")
print(f"Responded: {[r.agent_id for r in results]}")

for coord in coordinators:
    coord.stop()
```

---

## How routing works

```
bus.dispatch("pick cup", required_capability="PICK")
      │
      ▼
  filter agents with "PICK" capability
      │
      ├─ if one match → route to it
      ├─ if multiple matches → pick least-loaded (shortest queue)
      └─ if no capability match → route to any least-loaded agent
```

Routing is synchronous from the caller's perspective — `dispatch()` blocks until the task completes or the `timeout` elapses. Each `MultiAgentCoordinator` runs a background worker thread that dequeues tasks, calls `agent.plan()`, and returns a `TaskResult`.

---

## Scaling up

```python
# Add more pickers for higher throughput — the bus load-balances automatically
for i in range(5):
    robot = Robot(f"mock://picker_{i}", MockAdapter(f"picker_{i}"))
    coord = MultiAgentCoordinator(
        Agent(provider="rule"), robot, bus,
        agent_id=f"picker_{i}", capabilities=["PICK", "NAVIGATE"],
    )
    coord.start()

# Now dispatch routes across all 5 pickers, round-robin by load
```

**`broadcast()` for fleet-wide actions:**
```python
# All agents receive the task and execute in parallel
results = bus.broadcast("emergency_stop")
success = all(r.success for r in results)
```

---

## TaskResult

`bus.dispatch()` returns a `TaskResult`:
```python
result.success        # bool
result.agent_id       # which agent handled it
result.skills_planned # list of skill names in the plan
result.elapsed_ms     # total handling time
result.error          # error string if success=False
```

---

## Comparison to alternatives

| Approach | Code required | Load balancing | Capability routing |
|----------|--------------|----------------|-------------------|
| APYROBO TaskBus | ~5 lines per robot | Automatic | Built-in |
| ROS 2 action servers | Custom action server per robot | Manual | Custom |
| Redis + worker queue | Queue setup + serialization | Via Redis | Custom |
| Direct agent calls | Sequential in your code | None | None |

Unlike ROS 2 action servers, `TaskBus` requires no message definitions, no serialization, and no separate processes. All coordination happens in-process with Python threads.

---

## Integration with safety policies

Tasks dispatched through the bus can be checked against NL safety policies before execution:

```python
from apyrobo.safety.nl_policy import NLPolicyParser, NLPolicyStore

store = NLPolicyStore("~/.apyrobo/policies.db")
parser = NLPolicyParser()
policies = store.get_active_policies()

action = "navigate at 2.0 m/s to dock"
violations = parser.check_compliance(action, policies)
if not violations:
    result = bus.dispatch("navigate_to", required_capability="NAVIGATE")
```

---

## Related concepts

- [Natural language safety policies](nl_safety_policies.md) — enforce plain-English constraints before dispatch
- [Fleet coordination](../architecture.md) — lower-level swarm module for homogeneous robot fleets
- [Skill graph engine](../architecture.md) — DAG-based skill composition used by each agent's planner

---

*Keywords: APYROBO multi-agent coordination, robot task bus, multi-robot Python, fleet coordination AI, capability routing robots, AI robotics orchestration, robot agent coordination*

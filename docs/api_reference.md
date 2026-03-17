# API Reference

Auto-generated reference for the APYROBO public API. For guides and tutorials, see the [docs index](../README.md#documentation).

---

## Core

### `Robot`

```python
from apyrobo import Robot
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `discover()` | `(uri: str) -> Robot` | Discover a robot from a URI (`mock://`, `gazebo://`, `mqtt://`, `http://`) |
| `capabilities()` | `(refresh: bool = False) -> RobotCapability` | Get semantic capabilities (cached) |
| `move()` | `(x: float, y: float, speed: float = None) -> None` | Navigate to position |
| `stop()` | `() -> None` | Emergency stop |
| `rotate()` | `(angle_rad: float, speed: float = None) -> None` | Rotate in place |
| `cancel()` | `() -> None` | Cancel navigation goal |
| `gripper_open()` | `() -> bool` | Open gripper |
| `gripper_close()` | `() -> bool` | Close gripper |
| `get_position()` | `() -> tuple[float, float]` | Current (x, y) |
| `get_orientation()` | `() -> float` | Heading in radians |
| `get_health()` | `() -> dict[str, Any]` | Diagnostic info |
| `connect()` | `() -> None` | Establish connection |
| `disconnect()` | `() -> None` | Clean disconnect |

**Properties:** `robot_id: str`, `is_connected: bool`, `state: AdapterState`

---

### `RobotCapability`

```python
from apyrobo.core.schemas import RobotCapability
```

| Field | Type | Description |
|-------|------|-------------|
| `robot_id` | `str` | Unique identifier |
| `name` | `str` | Human-readable name |
| `capabilities` | `list[Capability]` | What the robot can do |
| `sensors` | `list[SensorInfo]` | Available sensors |
| `max_speed` | `float` | Maximum speed (m/s) |
| `workspace` | `dict` | Operating area bounds |

---

## Skills

### `Skill`

```python
from apyrobo import Skill, BUILTIN_SKILLS
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `skill_id` | `str` | required | Unique identifier |
| `name` | `str` | required | Human-readable name |
| `description` | `str` | `""` | What the skill does |
| `required_capability` | `CapabilityType` | `CUSTOM` | Robot capability needed |
| `preconditions` | `list[Condition]` | `[]` | Must be true before execution |
| `postconditions` | `list[Condition]` | `[]` | Become true after execution |
| `parameters` | `dict[str, Any]` | `{}` | Runtime configuration |
| `timeout_seconds` | `float` | `60.0` | Max execution time |
| `retry_count` | `int` | `0` | Retries on failure |

**Methods:** `to_dict()`, `from_dict()`, `to_json()`, `from_json()`

**Built-in skills:** `navigate_to`, `rotate`, `stop`, `pick_object`, `place_object`, `report_status`

---

### `SkillGraph`

```python
from apyrobo import SkillGraph
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_skill()` | `(skill, depends_on=[], parameters={})` | Add a skill node |
| `get_execution_order()` | `() -> list[Skill]` | Topological sort |
| `get_execution_layers()` | `() -> list[list[Skill]]` | Parallel grouping |
| `get_parameters()` | `(skill_id: str) -> dict` | Runtime parameters |
| `__len__()` | `() -> int` | Number of skills |

---

### `SkillExecutor`

```python
from apyrobo import SkillExecutor
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `execute_graph()` | `(graph, parallel=False, trace_id=None) -> TaskResult` | Execute full graph |
| `execute_skill()` | `(skill, parameters=None) -> SkillStatus` | Execute single skill |
| `on_event()` | `(listener: Callable) -> None` | Register event callback |
| `check_preconditions()` | `(skill, robot) -> tuple[bool, str]` | Check skill preconditions |

**Properties:** `state: ExecutionState`

---

### `Agent`

```python
from apyrobo import Agent
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__()` | `(provider="rule", **kwargs)` | Create agent with LLM provider |
| `plan()` | `(task: str, robot: Robot) -> SkillGraph` | Plan a task |
| `execute()` | `(task, robot, on_event=None, parallel=False, urgency=None) -> TaskResult` | Plan + execute |

**Providers:** `"rule"` (no API key), `"llm"` (any LiteLLM model), `"tool_calling"` (function calling), `"multi_turn"` (clarification)

---

## Safety

### `SafetyEnforcer`

```python
from apyrobo import SafetyEnforcer
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__()` | `(robot, policy="default")` | Wrap robot with safety |
| `move()` | `(x, y, speed=None)` | Move with full enforcement |
| `stop()` | `()` | Always allowed |
| `rotate()` | `(angle_rad, speed=None)` | Angular speed enforcement |
| `check_watchdog()` | `() -> dict \| None` | Check odometry divergence |
| `escalate()` | `(reason, context=None) -> bool` | Trigger human escalation |
| `acknowledge_escalation()` | `()` | Human ACK |
| `swap_policy()` | `(new_policy) -> SafetyPolicy` | Hot-swap policy |
| `add_collision_zone()` | `(zone: dict)` | Add zone at runtime |
| `check_battery()` | `(distance_m=0) -> dict` | Battery status |

**Properties:** `violations`, `interventions`, `audit_log`, `policy`, `robot_id`

---

### `SafetyPolicy`

```python
from apyrobo import SafetyPolicy
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"default"` | Policy name |
| `max_speed` | `float` | `1.5` | Speed cap (m/s) |
| `max_angular_speed` | `float` | `2.0` | Angular speed cap (rad/s) |
| `collision_zones` | `list[dict]` | `[]` | No-go rectangles |
| `human_proximity_limit` | `float` | `0.5` | Min human distance (m) |
| `move_timeout` | `float` | `120.0` | Auto-stop after (s) |
| `watchdog_tolerance` | `float` | `2.0` | Max odometry divergence (m) |
| `min_battery_pct` | `float` | `15.0` | Min battery to accept tasks |

**Built-in policies:** `"default"`, `"strict"`

---

## Swarm

### `SwarmBus`

```python
from apyrobo import SwarmBus
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `register()` | `(robot: Robot)` | Add robot to swarm |
| `unregister()` | `(robot_id: str)` | Remove robot |
| `send()` | `(sender, target, message, msg_type)` | Send targeted message |
| `broadcast()` | `(sender, message, msg_type)` | Broadcast to all |
| `on_message()` | `(robot_id, handler)` | Per-robot handler |
| `on_any()` | `(handler)` | Global handler |

**Properties:** `robot_ids`, `robot_count`, `message_log`

### `SwarmCoordinator`

```python
from apyrobo import SwarmCoordinator
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `split_task()` | `(task, agent) -> list[RobotAssignment]` | Split task across robots |
| `execute_task()` | `(task, agent, on_event=None) -> TaskResult` | Full swarm execution |

**Properties:** `assignments`, `events`

---

## Observability

### `MetricsCollector`

```python
from apyrobo.observability import MetricsCollector, on_event
```

| Method | Description |
|--------|-------------|
| `handle_event(event)` | Process an observability event |
| `prometheus_text()` | Prometheus text exposition format |
| `get_skill_metrics(skill_id=None)` | Per-skill or all-skill metrics |
| `summary()` | High-level metrics summary |

### `AlertManager`

```python
from apyrobo.observability import AlertManager, AlertRule
```

| Method | Description |
|--------|-------------|
| `add_rule(rule)` | Register an alert rule |
| `add_callback(fn)` | Callback when rule fires |
| `check_event(event)` | Check rules against event |

**Metrics:** `"skill_failure_rate"`, `"graph_failure_rate"`, `"avg_skill_latency_ms"`, `"event_rate"`

### `TimeSeriesStore`

```python
from apyrobo.observability import TimeSeriesStore
```

| Method | Description |
|--------|-------------|
| `handle_event(event)` | Record event as InfluxDB line protocol |
| `lines()` | Get buffered line protocol lines |
| `flush_to_file(path)` | Write to file |
| `record(measurement, tags, fields)` | Record arbitrary metric |

---

## Persistence

### `StateStore` / `SQLiteStateStore` / `RedisStateStore`

```python
from apyrobo.persistence import StateStore, SQLiteStateStore, RedisStateStore
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `begin_task()` | `(task_id, metadata, robot_id, total_steps)` | Record task start |
| `update_task()` | `(task_id, step, total_steps, status)` | Update progress |
| `complete_task()` | `(task_id, result)` | Mark completed |
| `fail_task()` | `(task_id, error)` | Mark failed |
| `get_task()` | `(task_id) -> TaskJournalEntry` | Get task by ID |
| `get_interrupted_tasks()` | `() -> list` | Tasks running at crash |
| `get_recent_tasks()` | `(limit=20) -> list` | Recent tasks |
| `set()` / `get()` | Key-value store | Generic state |

---

## REST API (OperationsApiServer)

### Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/health` | Health check | Optional |
| `GET` | `/robots` | List robots with capabilities | Optional |
| `POST` | `/tasks` | Submit a task (returns 202) | Optional |
| `GET` | `/tasks/{id}` | Get task status | Optional |
| `DELETE` | `/tasks/{id}` | Cancel a task | Optional |

### Authentication

When an `AuthManager` is provided, all endpoints require an `X-API-Key` header.
Requests without a valid key receive `401 Unauthorized`.

### Usage

```python
from apyrobo.operations import OperationsApiServer
from apyrobo.auth import AuthManager

auth = AuthManager()
auth.add_user("admin", role="admin", api_key="my-secret-key")

server = OperationsApiServer(
    port=8081,
    auth_manager=auth,
    swarm_bus=bus,       # optional: auto-discover robots
    state_store=store,   # optional: persist task state
    agent=agent,         # optional: execute tasks via Agent
)
server.start()
```

### `POST /tasks` Request/Response

```json
// Request
{"task": "deliver package to room 3", "robot_id": "tb4"}

// Response (202)
{"task_id": "a1b2c3d4e5", "status": "queued"}
```

---

## Scheduled Tasks (ScheduledTaskRunner)

### `ScheduledTaskRunner`

```python
from apyrobo.operations import ScheduledTaskRunner
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_interval_job()` | `(name, interval_s, fn)` | Register a periodic callable |
| `add_task()` | `(name, cron_expr, task_description, robot, agent)` | Register an agent-based cron task |
| `start()` | `()` | Start the background scheduler thread |
| `stop()` | `()` | Stop the scheduler and join thread |

### Cron Expressions

| Expression | Interval |
|------------|----------|
| `*/30 * * * *` | Every 30 minutes |
| `0 2 * * *` | Daily at 2:00 AM |
| `0 */4 * * *` | Every 4 hours |
| `* * * * *` | Every minute |

### Usage

```python
runner = ScheduledTaskRunner(state_store=store)
runner.add_task("patrol", "*/30 * * * *", "patrol warehouse", robot, agent)
runner.add_interval_job("heartbeat", 60, lambda: print("alive"))
runner.start()
```

---

## Generating Full API Docs

To generate browsable HTML docs from docstrings:

```bash
# Using pdoc
pip install pdoc
pdoc apyrobo --html --output-dir docs/api/

# Using mkdocs + mkdocstrings
pip install mkdocs mkdocs-material mkdocstrings[python]
mkdocs build
```

See [mkdocs.yml](../mkdocs.yml) for configuration (if present).

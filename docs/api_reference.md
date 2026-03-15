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

# Skill Authoring Guide

How to write, test, package, and publish custom skills for APYROBO.

---

## What Is a Skill?

A **Skill** is a named, reusable robot action with:
- **Preconditions** — what must be true before execution
- **Postconditions** — what becomes true after execution
- **Parameters** — runtime configuration
- **Required capability** — what the robot must be able to do
- **Timeout + retry** — fault tolerance

Skills are the atoms of behavior. The **Skill Graph** chains them into complex task plans.

---

## 1. Define a Skill

```python
from apyrobo.skills.skill import Skill, Condition
from apyrobo.core.schemas import CapabilityType

# Define a custom skill
scan_area = Skill(
    skill_id="scan_area",
    name="Scan Area",
    description="Rotate 360 degrees while collecting sensor data",
    required_capability=CapabilityType.ROTATE,
    preconditions=[
        Condition(
            name="robot_idle",
            description="Robot is not currently moving",
            check_type="state",
            parameters={"key": "robot_idle", "value": True},
        ),
    ],
    postconditions=[
        Condition(
            name="area_scanned",
            description="360-degree scan complete",
            check_type="state",
            parameters={"key": "area_scanned", "value": True},
        ),
    ],
    parameters={
        "rotation_speed": 0.5,  # rad/s
        "scan_duration": 10.0,  # seconds
    },
    timeout_seconds=30.0,
    retry_count=1,
)
```

### Check Types

| Type | Purpose | Parameters |
|------|---------|------------|
| `capability` | Robot has the required capability | (automatic from `required_capability`) |
| `state` | Execution state flag is set | `{"key": "flag_name", "value": true}` |
| `speed` | Speed parameter within robot limits | (automatic from `parameters.speed`) |
| `custom` | Always passes (for extension) | Any |

---

## 2. Register as a Built-in

To make your skill available globally:

```python
from apyrobo.skills.skill import BUILTIN_SKILLS

# Register it
BUILTIN_SKILLS["scan_area"] = scan_area

# Now the agent can plan with it
from apyrobo import Agent, Robot

agent = Agent(provider="rule")
robot = Robot.discover("mock://bot")
# The agent can now include "scan_area" in task plans
```

---

## 3. Implement a Custom Handler

By default, the executor dispatches skills to robot commands based on `skill_id`. For custom behavior, subclass `SkillExecutor`:

```python
from apyrobo.skills.executor import SkillExecutor
from apyrobo.skills.skill import Skill
from apyrobo.core.robot import Robot
from typing import Any
import time

class CustomExecutor(SkillExecutor):
    """Executor with custom skill handlers."""

    def _dispatch_skill(self, skill: Skill, params: dict[str, Any]) -> bool:
        """Route skill execution to the appropriate handler."""

        if skill.skill_id == "scan_area":
            return self._handle_scan_area(skill, params)

        # Fall back to default handling
        return super()._dispatch_skill(skill, params)

    def _handle_scan_area(self, skill: Skill, params: dict[str, Any]) -> bool:
        """Custom handler for scan_area skill."""
        speed = params.get("rotation_speed", 0.5)
        duration = params.get("scan_duration", 10.0)

        # Rotate slowly while collecting data
        self._robot.rotate(angle_rad=6.28, speed=speed)  # full rotation

        # Mark scan complete in execution state
        self._state.set("area_scanned", True)

        return True
```

---

## 4. Chain Skills in a Graph

```python
from apyrobo import SkillGraph, BUILTIN_SKILLS

graph = SkillGraph()

# Navigate to scan position
graph.add_skill(
    BUILTIN_SKILLS["navigate_to"],
    parameters={"x": 5.0, "y": 5.0, "speed": 0.5},
)

# Scan the area (depends on arriving first)
graph.add_skill(
    scan_area,
    depends_on=["navigate_to"],
)

# Navigate home after scanning
from apyrobo.skills.skill import Skill
go_home = Skill(
    skill_id="go_home",
    name="Go Home",
    required_capability=CapabilityType.NAVIGATE,
    parameters={"x": 0.0, "y": 0.0, "speed": 0.5},
)
graph.add_skill(go_home, depends_on=["scan_area"])

# Execute
executor = CustomExecutor(robot)
result = executor.execute_graph(graph)
```

### Parallel Execution

Independent skills run concurrently when `parallel=True`:

```python
graph = SkillGraph()
graph.add_skill(skill_a)  # no dependencies
graph.add_skill(skill_b)  # no dependencies
graph.add_skill(skill_c, depends_on=["skill_a", "skill_b"])  # waits for both

result = executor.execute_graph(graph, parallel=True)
# skill_a and skill_b run concurrently, then skill_c runs
```

---

## 5. Test Your Skill

```python
import pytest
from apyrobo import Robot, SkillExecutor, SkillGraph
from apyrobo.skills.skill import SkillStatus

def test_scan_area_completes():
    robot = Robot.discover("mock://test_bot")
    executor = CustomExecutor(robot)

    graph = SkillGraph()
    graph.add_skill(scan_area)

    result = executor.execute_graph(graph)
    assert result.status.value == "completed"
    assert executor.state.is_set("area_scanned")

def test_scan_area_events():
    robot = Robot.discover("mock://test_bot")
    executor = CustomExecutor(robot)
    events = []
    executor.on_event(lambda e: events.append(e))

    executor.execute_skill(scan_area)

    statuses = [e.status for e in events]
    assert SkillStatus.COMPLETED in statuses

def test_scan_area_timeout():
    """Skill fails gracefully on timeout."""
    slow_scan = Skill(
        skill_id="scan_area",
        name="Slow Scan",
        required_capability=CapabilityType.ROTATE,
        timeout_seconds=0.01,  # very short timeout
    )

    class SlowExecutor(SkillExecutor):
        def _dispatch_skill(self, skill, params):
            import time
            time.sleep(5)  # too slow
            return True

    executor = SlowExecutor(Robot.discover("mock://test"))
    status = executor.execute_skill(slow_scan)
    assert status == SkillStatus.FAILED
```

Run tests:

```bash
pytest tests/test_skills/ -v -o "addopts="
```

---

## 6. Serialization

Skills serialize to JSON for storage, transport, and configuration:

```python
# To JSON
json_str = scan_area.to_json()
print(json_str)

# From JSON
loaded = Skill.from_json(json_str)
assert loaded.skill_id == "scan_area"

# To/from dict
d = scan_area.to_dict()
loaded = Skill.from_dict(d)
```

---

## 7. Skill Design Checklist

- [ ] **Descriptive `skill_id`** — snake_case, unique across the system
- [ ] **Correct `required_capability`** — matches what the robot needs
- [ ] **Preconditions** — document what must be true before execution
- [ ] **Postconditions** — update execution state for downstream skills
- [ ] **Reasonable timeout** — longer than expected execution, shorter than acceptable wait
- [ ] **Retry count** — set > 0 for transient failures (gripper, navigation)
- [ ] **Parameters with defaults** — every parameter has a sensible default
- [ ] **Tests** — at least one happy path + one failure/timeout test
- [ ] **Docstring** — explain what the skill does and when to use it

---

## Built-in Skills Reference

| Skill ID | Capability | Description | Retries |
|----------|-----------|-------------|---------|
| `navigate_to` | NAVIGATE | Move to (x, y) position | 0 |
| `rotate` | ROTATE | Rotate by angle (radians) | 0 |
| `stop` | NAVIGATE | Halt all motion | 0 |
| `pick_object` | PICK | Close gripper on object | 2 |
| `place_object` | PLACE | Open gripper at position | 0 |
| `report_status` | CUSTOM | Report robot status | 0 |

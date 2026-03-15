# Adapter Authoring Guide

How to add support for new robot hardware by implementing a `CapabilityAdapter`.

---

## What Is an Adapter?

An adapter bridges APYROBO's semantic API to actual robot hardware. It translates high-level commands like `move(x, y, speed)` into whatever your robot understands — ROS 2 topics, MQTT messages, HTTP calls, serial protocols, etc.

```
APYROBO (semantic)  →  Adapter (translation)  →  Robot (hardware)
   robot.move(2, 3)       publish /cmd_vel          wheels turn
   robot.gripper_close()  send MQTT command          gripper closes
```

---

## Step 1: Subclass CapabilityAdapter

```python
from apyrobo.core.adapters import CapabilityAdapter, register_adapter
from apyrobo.core.schemas import (
    RobotCapability, Capability, CapabilityType,
    SensorInfo, AdapterState,
)
from typing import Any

@register_adapter("myrobot")
class MyRobotAdapter(CapabilityAdapter):
    """
    Adapter for MyRobot hardware.

    URI: myrobot://robot_name
    Usage: Robot.discover("myrobot://arm_01")
    """

    def __init__(self, robot_name: str, **kwargs: Any) -> None:
        super().__init__(robot_name, **kwargs)
        self._position = (0.0, 0.0)
        self._orientation = 0.0
        self._connected = False
        # Initialize your hardware connection here

    # ------------------------------------------------------------------
    # Required: Capability discovery
    # ------------------------------------------------------------------

    def get_capabilities(self) -> RobotCapability:
        """Return what this robot can do."""
        return RobotCapability(
            robot_id=self.robot_name,
            name=f"MyRobot-{self.robot_name}",
            capabilities=[
                Capability(
                    capability_type=CapabilityType.NAVIGATE,
                    name="navigate_to",
                    description="Move to (x, y)",
                ),
                Capability(
                    capability_type=CapabilityType.ROTATE,
                    name="rotate",
                    description="Rotate in place",
                ),
            ],
            sensors=[
                SensorInfo(sensor_id="lidar", name="2D LiDAR", hz=10.0),
            ],
            max_speed=1.0,
        )

    # ------------------------------------------------------------------
    # Required: Navigation
    # ------------------------------------------------------------------

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        """Move to (x, y). Translate to your robot's protocol."""
        # Example: send ROS 2 goal
        # self._nav2_client.send_goal(x, y, speed)

        # Example: send HTTP command
        # requests.post(f"http://{self._host}/move", json={"x": x, "y": y})

        self._position = (x, y)

    def stop(self) -> None:
        """Emergency stop — must always work."""
        # self._publisher.publish(Twist())  # zero velocity
        pass

    def rotate(self, angle_rad: float, speed: float | None = None) -> None:
        """Rotate in place."""
        self._orientation += angle_rad

    def cancel(self) -> None:
        """Cancel current navigation goal."""
        self.stop()

    # ------------------------------------------------------------------
    # Required: State queries
    # ------------------------------------------------------------------

    def get_position(self) -> tuple[float, float]:
        """Return current (x, y) from odometry/localization."""
        return self._position

    def get_orientation(self) -> float:
        """Return heading in radians."""
        return self._orientation

    def get_health(self) -> dict[str, Any]:
        """Return diagnostic info."""
        return {
            "connected": self._connected,
            "battery_pct": 85.0,
            "position": self._position,
        }

    # ------------------------------------------------------------------
    # Optional: Gripper (return False if not supported)
    # ------------------------------------------------------------------

    def gripper_open(self) -> bool:
        return False  # Not supported on this robot

    def gripper_close(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Optional: Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Establish connection to the robot."""
        self._connected = True
        self._state = AdapterState.CONNECTED

    def disconnect(self) -> None:
        """Clean disconnect."""
        self._connected = False
        self._state = AdapterState.DISCONNECTED
```

---

## Step 2: Register with a URI Scheme

The `@register_adapter("myrobot")` decorator registers your adapter. Now users can discover it:

```python
from apyrobo import Robot

robot = Robot.discover("myrobot://arm_01")
caps = robot.capabilities()
robot.move(x=1.0, y=2.0, speed=0.5)
```

If you prefer manual registration:

```python
from apyrobo.core.adapters import register_adapter_class

register_adapter_class("myrobot", MyRobotAdapter)
```

---

## Step 3: Test Your Adapter

```python
import pytest
from apyrobo import Robot

def test_discover():
    robot = Robot.discover("myrobot://test_unit")
    assert robot.robot_id == "test_unit"

def test_capabilities():
    robot = Robot.discover("myrobot://test_unit")
    caps = robot.capabilities()
    assert any(c.capability_type.value == "navigate" for c in caps.capabilities)

def test_move():
    robot = Robot.discover("myrobot://test_unit")
    robot.move(x=3.0, y=4.0, speed=0.5)
    assert robot.get_position() == (3.0, 4.0)

def test_stop():
    robot = Robot.discover("myrobot://test_unit")
    robot.move(x=1.0, y=1.0)
    robot.stop()  # Should not raise

def test_health():
    robot = Robot.discover("myrobot://test_unit")
    health = robot.get_health()
    assert "connected" in health

def test_works_with_executor():
    """Full integration: adapter + skill executor."""
    from apyrobo import SkillExecutor, SkillGraph, BUILTIN_SKILLS

    robot = Robot.discover("myrobot://test_unit")
    graph = SkillGraph()
    graph.add_skill(BUILTIN_SKILLS["navigate_to"],
                    parameters={"x": 1.0, "y": 2.0})
    executor = SkillExecutor(robot)
    result = executor.execute_graph(graph)
    assert result.status.value == "completed"
```

---

## Step 4: Use with Safety Enforcer

Your adapter works with the safety enforcer automatically:

```python
from apyrobo import Robot, SafetyEnforcer

robot = Robot.discover("myrobot://arm_01")
enforcer = SafetyEnforcer(robot, policy="strict")

# Speed is clamped, collision zones are checked, watchdog runs
enforcer.move(x=5.0, y=5.0, speed=10.0)  # clamped to 0.5 m/s
```

---

## Existing Adapters

| Adapter | URI | Protocol | Source |
|---------|-----|----------|--------|
| `MockAdapter` | `mock://` | In-memory | `core/adapters.py` |
| `GazeboAdapter` | `gazebo://` | Sim API | `core/adapters.py` |
| `MQTTAdapter` | `mqtt://` | MQTT topics | `core/adapters.py` |
| `HTTPAdapter` | `http://` | REST API | `core/adapters.py` |

---

## Adapter API Contract

### Required Methods

| Method | Signature | Notes |
|--------|-----------|-------|
| `get_capabilities()` | `-> RobotCapability` | What the robot can do |
| `move(x, y, speed)` | `-> None` | Navigate to position |
| `stop()` | `-> None` | Emergency stop (always works) |
| `rotate(angle_rad, speed)` | `-> None` | Rotate in place |
| `get_position()` | `-> tuple[float, float]` | Current (x, y) |
| `get_orientation()` | `-> float` | Heading in radians |
| `get_health()` | `-> dict[str, Any]` | Diagnostic info |

### Optional Methods (have defaults)

| Method | Default | Override When |
|--------|---------|---------------|
| `cancel()` | calls `stop()` | Nav2 cancel is different from stop |
| `gripper_open()` | returns `True` | Robot has a gripper |
| `gripper_close()` | returns `True` | Robot has a gripper |
| `connect()` | no-op | Connection setup needed |
| `disconnect()` | no-op | Clean disconnect needed |

### Properties

| Property | Type | Notes |
|----------|------|-------|
| `robot_name` | `str` | Set in `__init__` |
| `robot_id` | `str` | Alias for `robot_name` |
| `is_connected` | `bool` | Default: `True` |
| `state` | `AdapterState` | `CONNECTED`, `DISCONNECTED`, `ERROR` |

---

## Tips

1. **Always implement `stop()`** — It must work even if the robot is in a bad state
2. **`get_position()` should be fast** — It's called frequently by the watchdog
3. **Report capabilities accurately** — The executor uses them for precondition checks
4. **Use `connect()`/`disconnect()`** — For expensive connections (ROS nodes, TCP sockets)
5. **Log with the APYROBO logger** — `from apyrobo.observability import get_logger`

# Hardware Adapter Pattern — APYROBO Universal Robot Interface

**What it is:** A URI-based adapter system that lets the same APYROBO code run on any robot. Change `mock://` to `unitree://go2@192.168.1.10` and your code runs on a real Unitree Go2 without any changes to your agent, skills, or safety policies.

**When to use it:** Always. The adapter pattern is the core of APYROBO's hardware abstraction. Develop with `mock://`, test in simulation with `ros2://` against a running Gazebo, deploy to real hardware with a hardware-specific URI. (`gazebo://` is a physics-flavoured mock: it does not connect to Gazebo.)

**Category:** AI-native robotics orchestration, hardware abstraction, robot adapter pattern, ROS 2 alternative

---

## Runnable example

```python
from apyrobo import MockAdapter, Robot, Agent, SkillExecutor

# Development: mock adapter (no hardware needed)
robot = Robot("mock://my_robot", MockAdapter("my_robot"))

# All code below is identical regardless of adapter:
agent = Agent(provider="rule")
executor = SkillExecutor(robot)

plan = agent.plan("go to (2, 3)", robot)
result = executor.execute_graph(plan)
print(result.status.value)    # completed
print(robot.get_position())   # (2.0, 3.0)
```

**Swap to real hardware — only the line that makes the robot changes:**
```python
# Unitree Go2 quadruped (importing the module registers unitree://)
import apyrobo.core.unitree_adapter  # noqa: F401
robot = Robot.discover("unitree://go2@192.168.1.10")

# ROS 2, including Nav2 navigation (needs rclpy: source /opt/ros/<distro>/setup.bash first)
robot = Robot.discover("ros2://turtlebot4")

# NVIDIA Isaac Sim (SDK adapter; importing it registers isaac_sim://)
from apyrobo.core.isaac_adapter import IsaacSimAdapter
robot = Robot("isaac_sim://sim_robot_0", IsaacSimAdapter("sim_robot_0", host="localhost", port=8211))
```

---

## Available adapters

| URI scheme | Adapter class | Hardware |
|-----------|--------------|---------|
| `mock://` | `MockAdapter` | Simulation / testing — no hardware |
| `gazebo://` | `GazeboAdapter` | Physics-flavoured mock; does not connect to Gazebo |
| `ros2://` | ROS 2 adapter | Any ROS 2 robot; drives Nav2 when it runs (needs rclpy) |
| `unitree://go2@...` | `UnitreeAdapter` (`apyrobo.core.unitree_adapter`) | Unitree Go2 quadruped (opt-in import) |
| `unitree://h1@...` | `UnitreeAdapter` (`apyrobo.core.unitree_adapter`) | Unitree H1 humanoid (opt-in import) |
| `isaac://...` | `IsaacSimAdapter` (`apyrobo.sim.adapters`) | In-memory stand-in, no Isaac Sim |
| `isaac_sim://...` | `IsaacSimAdapter` (`apyrobo.core.isaac_adapter`) | NVIDIA Isaac Sim via SDK or REST (opt-in import) |
| `mqtt://` | MQTT adapter | MQTT-capable robots |
| `http://` | HTTP adapter | REST-based robot APIs |

`Robot.discover()` with an unknown scheme names every registered one in its error. `apyrobo.nav2.Nav2Adapter` and `apyrobo.moveit.MoveItAdapter` are helper clients for Nav2 and MoveIt 2, not URI schemes: a `Robot` does not take them.

---

## Writing a custom adapter

Implement three methods:

```python
from apyrobo.core.adapters import CapabilityAdapter
from apyrobo.core.schemas import Capability, CapabilityType, RobotCapability

class MyRobotAdapter(CapabilityAdapter):

    def get_capabilities(self) -> RobotCapability:
        # Describe what your robot can do
        return RobotCapability(
            robot_id=self.robot_name,
            name="My Robot",
            capabilities=[
                Capability(capability_type=CapabilityType.NAVIGATE, name="navigate_to"),
            ],
            max_speed=1.0,
        )

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        # Send the robot to (x, y)
        self.target = (x, y, speed)

    def stop(self) -> None:
        # Halt at once; this must work in every state
        self.target = None
```

Full authoring guide: [adapter_authoring.md](../adapter_authoring.md)

---

## Why the adapter pattern wins

| Concern | APYROBO | Direct ROS 2 |
|---------|---------|-------------|
| Test in CI (no robot) | `MockAdapter` — instant | Needs Gazebo or hardware |
| Swap hardware | Change URI | Rewrite node communication |
| Skill portability | Same `SkillGraph` runs everywhere | Node-specific implementation |
| Safety enforcement | Framework-level, adapter-independent | Per-node custom code |
| Multi-hardware fleet | Mix URIs in `TaskBus` | Custom coordinator per pair |

---

## Related concepts

- [Skill authoring](../skill_authoring.md) — write skills that work across all adapters
- [Multi-agent coordination](multi_agent_coordination.md) — mix different adapters in one `TaskBus`
- [Natural language safety policies](nl_safety_policies.md) — check plain-English safety rules, independent of adapter

---

*Keywords: APYROBO adapter pattern, hardware abstraction robots, robot URI scheme, mock robot testing, ROS 2 alternative, hardware-agnostic robot programming, AI robotics Python*

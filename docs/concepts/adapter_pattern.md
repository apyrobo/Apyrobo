# Hardware Adapter Pattern — APYROBO Universal Robot Interface

**What it is:** A URI-based adapter system that lets the same APYROBO code run on any robot. Change `mock://` to `unitree://go2@192.168.1.10` and your code runs on a real Unitree Go2 without any changes to your agent, skills, or safety policies.

**When to use it:** Always. The adapter pattern is the core of APYROBO's hardware abstraction. Develop with `mock://`, test in simulation with `gazebo://`, deploy to real hardware with a hardware-specific URI.

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

plan = agent.plan("navigate to point A and pick up object", robot)
result = executor.execute_graph(plan)
print(result.status)  # completed
```

**Swap to real hardware — only the first two lines change:**
```python
# Unitree Go2 quadruped
from apyrobo.core.unitree_adapter import UnitreeAdapter
robot = Robot("unitree://go2@192.168.1.10", UnitreeAdapter("go2", "192.168.1.10"))

# ROS 2 navigation stack
from apyrobo.nav2 import Nav2Adapter
robot = Robot("nav2://turtlebot4", Nav2Adapter("turtlebot4"))

# NVIDIA Isaac Sim (SDK adapter; importing it registers isaac_sim://)
from apyrobo.core.isaac_adapter import IsaacSimAdapter
robot = Robot("isaac_sim://sim_robot_0", IsaacSimAdapter("sim_robot_0", host="localhost", port=8211))
```

---

## Available adapters

| URI scheme | Adapter class | Hardware |
|-----------|--------------|---------|
| `mock://` | `MockAdapter` | Simulation / testing — no hardware |
| `gazebo://` | Gazebo adapter | Gazebo simulator |
| `nav2://` | `Nav2Adapter` | ROS 2 Nav2 navigation stack |
| `moveit://` | MoveIt adapter | ROS 2 MoveIt 2 manipulation |
| `unitree://go2@...` | `UnitreeAdapter` | Unitree Go2 quadruped |
| `unitree://h1@...` | `UnitreeAdapter` | Unitree H1 humanoid |
| `isaac://...` | `IsaacSimAdapter` (`apyrobo.sim.adapters`) | In-memory stand-in, no Isaac Sim |
| `isaac_sim://...` | `IsaacSimAdapter` (`apyrobo.core.isaac_adapter`) | NVIDIA Isaac Sim via SDK or REST (opt-in import) |
| `mqtt://` | MQTT adapter | MQTT-capable robots |
| `http://` | HTTP adapter | REST-based robot APIs |

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

# APYROBO packages

Two kinds of things live here. **Read this before assuming a package drives
real hardware.**

## Real (drive a robot / speak a real protocol today)

| Package | What it is |
|---------|------------|
| [`apyrobo-client-ts`](apyrobo-client-ts/) | TypeScript client for the wire protocol — talks to `apyrobo serve` for real (CI runs it against the Python server). |
| [`apyrobo-skills-ros-nav`](apyrobo-skills-ros-nav/) | Real ROS 2 Nav2 skills — `navigate_to_pose` etc. call the live `NavigateToPose` action via `rclpy`; raises a clear error when ROS 2 isn't sourced. |

The **`ros2://` adapter** in the core package ([`apyrobo/core/ros2_bridge.py`](../apyrobo/core/ros2_bridge.py))
is also real: it publishes `/cmd_vel`, subscribes `/odom`, and is verified in
CI driving a **physics-simulated TurtleBot3 in Gazebo**
([`tests/integration/test_gazebo_turtlebot.py`](../tests/integration/test_gazebo_turtlebot.py)).
That is the path to use for real mobile robots today.

### The flagship reference stack

`ros2://` adapter + `apyrobo-skills-ros-nav` + the Gazebo integration CI job
+ [docs/TURTLEBOT4.md](../docs/TURTLEBOT4.md) together form the **flagship
reference stack** — the one end-to-end path (natural-language task → plan →
real Nav2 action → simulated robot moves) that is exercised in CI on every
commit. It is the canonical example for anyone writing an adapter
([docs/adapter_authoring.md](../docs/adapter_authoring.md)) or wiring a
scaffold to a vendor SDK: when a scaffold below graduates to "real", the bar
is "does what the flagship does, for its hardware."

## Reference scaffolds (templates — they do **not** move hardware)

These print the motion they *would* perform and return success. They are
starting points to wire to a vendor SDK, **not** working hardware support.
Each one warns at registration time and says so in its README.

| Package | Robot | Wire it to |
|---------|-------|-----------|
| [`apyrobo-skills-ur`](apyrobo-skills-ur/) | Universal Robots | `ur_rtde` / UR ROS 2 driver |
| [`apyrobo-skills-spot`](apyrobo-skills-spot/) | Boston Dynamics Spot | `bosdyn` SDK |
| [`apyrobo-skills-franka`](apyrobo-skills-franka/) | Franka Panda | `franky` / `libfranka` |
| [`apyrobo-skills-drone-px4`](apyrobo-skills-drone-px4/) | PX4 drones | MAVSDK / `pymavlink` |
| [`apyrobo-skills-agv`](apyrobo-skills-agv/) | Generic AGVs | fleet API (VDA5050) |
| [`apyrobo-skills-turtlebot4`](apyrobo-skills-turtlebot4/) | TurtleBot 4 | use the real `ros2://` adapter instead |

Why ship scaffolds at all? They fix the skill *shape* (names, parameters,
capability tags, tests) so wiring one to a real SDK is a fill-in-the-body job,
not a design job — see [docs/skill_authoring.md](../docs/skill_authoring.md).
But until that wiring exists, treat them as examples, not drivers.

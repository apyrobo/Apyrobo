# From `mock://` to your first real robot

This is the order to go in when you move `apyrobo mcp` from the simulated robot
to hardware. It assumes a ROS 2 mobile robot running Nav2, and that the
[try-out in `docs/mcp.md`](mcp.md) already works for you.

**Read this first.** APYROBO is a software check between a model and a robot. It
is not a certified safety system and it does not replace a hardware e-stop. Keep
the robot's own e-stop within reach for every step below, and keep its bumpers,
cliff sensors and other onboard protection on.

CI runs this stack against a simulated TurtleBot3 with Nav2 in Gazebo, not
against real hardware. Each step below was run on `mock://` or printed as a
config. The commands in step 1 and in "Start on the robot" need a live ROS 2
robot, so CI does not run them.

## 1. Find your coordinates in the map frame

No-go zones and allowed areas are axis-aligned rectangles in metres, in the map
frame: goals go to Nav2 in `map`, and Nav2's route comes back in it.

Check `position_frame` in `robot_status`. On `ros2://` the position comes from
the map-to-base transform when TF has a fresh one, and the field says `map`. If
it says `odom`, localisation is not running or has stopped: the position is raw
odometry, which only matches the map if the robot started at the map's origin
and has not drifted, and the checks that use the robot's position are off by
that difference. Fix localisation before you trust them.

To read a point, open your map in RViz with the fixed frame set to `map`, use
the Publish Point tool on the spot, and watch the result:

```bash
ros2 topic echo /clicked_point
```

Take the corners of each place the robot must stay out of, and of the area it
may work in. Add a margin: the robot has a width, and zones are checked against
its centre.

## 2. Write a policy file

Start from a comment-rich example in
[`examples/policies/`](../examples/policies/), or from this:

```yaml
name: first-robot
base: default
max_speed: 0.2          # m/s; start low
max_angular_speed: 0.5  # rad/s

no_go_zones:
  - name: stairs
    x_min: 2.0
    y_min: 2.0
    x_max: 4.0
    y_max: 4.0

allowed_areas:
  - name: room
    x_min: -1.0
    y_min: -1.0
    x_max: 5.0
    y_max: 5.0
```

With `allowed_areas` set, a target outside every area is refused, so a place you
forgot to write down is off limits. Nav2's planned route is checked too: a plan
that leaves the areas on the way to an allowed target is refused, so two areas
with a gap between them need a third area joining them. A robot that starts
outside every area may drive in.

## 3. Check it on `mock://`

```bash
apyrobo mcp selftest --policy-file first-robot.yaml --confirm-moves
```

This moves no hardware. It runs a normal move, shows an over-limit move clamped,
and tries each zone and a target outside your areas. It exits 0 if all of that
held. Read the `warn` lines too: they name what your policy leaves unset.

A pass only says that the limits you wrote hold. It cannot know what is missing
from your map.

## 4. Start on the robot, slow, with confirmation

Run the server on the robot, in a shell where ROS 2 is sourced:

```bash
source /opt/ros/jazzy/setup.bash
apyrobo mcp --robot ros2:// --policy-file first-robot.yaml --confirm-moves
```

`--confirm-moves` makes each `move_to`, `rotate` and `execute_task` wait for your
yes, and shows the speed the policy will allow. Your MCP client must support
form elicitation. If it does not, every motion command is refused, which is the
safe way round.

For the first session, clear the area of people, send one short move at a time
into open floor, and then send one target inside a no-go zone and check that it
is refused and the robot stays still. Compare where the robot ends up with where
you sent it. If they disagree, fix that before you widen anything.

## 5. Connect your laptop over ssh

The client usually runs on a laptop. Print a config that runs the server on the
robot, so the policy is enforced there:

```bash
apyrobo mcp config claude-desktop --ssh ubuntu@robot.local \
    --ssh-setup /opt/ros/jazzy/setup.bash \
    --robot ros2:// --policy-file /home/ubuntu/first-robot.yaml --confirm-moves
```

The file path is on the robot, and login must be key-based. See
[Client on your laptop, robot elsewhere](mcp.md#client-on-your-laptop-robot-elsewhere).

There is no rosbridge to install and no websocket port to open. `ros2://` talks
to ROS 2 through `rclpy` inside the server's own process, so the server has to
run where ROS 2 is sourced and on the robot's `ROS_DOMAIN_ID`. On the robot, or
over `--ssh` as above, that is the robot's own ROS 2.

If your robot uses a domain other than 0, set it in the file you pass to
`--ssh-setup`. A non-interactive ssh shell may skip `.bashrc`, and an `env` block
in the client config stays on your laptop:

```bash
# /home/ubuntu/apyrobo-ros.bash on the robot
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=30
```

The server stops the robot when its client goes away, tested on `mock://` and a
slow test adapter, not yet on a real Nav2 goal. A dead network link is not seen
until TCP gives up, so set `ClientAliveInterval` and `ClientAliveCountMax` in the
robot's `sshd_config` to bound the wait.

## 6. Add a keepout filter for anywhere the robot must never enter

On `ros2://` with Nav2 the server checks Nav2's planned route before it sends the
goal. It does not watch the drive: the controller can stray from the plan and
Nav2 can replan. For an area that matters, add a Nav2 keepout filter to both the
global and the local costmap. That is the stronger layer.
[`apyrobo mcp keepout`](mcp.md#write-a-nav2-keepout-mask) writes the mask from
your zones and prints the parameters, but it is experimental and the wiring is
yours: check in simulation that Nav2 routes around the zone.

Only raise `max_speed` or drop `--confirm-moves` once short supervised runs have
behaved as you expect.

## What APYROBO does not protect against

- **People and obstacles it was not told about.** It does not read sensors to
  stop for a person. That is Nav2's costmaps and your robot's own protection.
- **Anything outside the limits you wrote.** A hazard you left off the policy
  looks like free space, unless `allowed_areas` happens to rule out the target.
- **A wrong map or a bad localisation estimate.** Zones are only as good as the
  frame they are drawn in.
- **The path actually driven.** See step 6.
- **Software that bypasses it.** Limits apply to commands that go through
  `apyrobo mcp`. Anything else publishing to `/cmd_vel` or sending Nav2 goals is
  not checked.
- **Failure of the machine it runs on.** It is ordinary software on an ordinary
  computer. That is what the hardware e-stop is for.

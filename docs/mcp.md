# `apyrobo mcp` — let Claude, Cursor or any MCP client drive a robot, safely

`apyrobo mcp` is a [Model Context Protocol](https://modelcontextprotocol.io)
server. Point an MCP client at it and the model gets tools to look at a
robot, plan tasks and move it. Every motion command goes through APYROBO's
`SafetyEnforcer` first:

- a speed above the policy limit is **clamped**, and the result says so;
- a target inside a no-go zone is **refused**; the robot does not move and the
  model gets a readable `REFUSED by safety policy …` error, so it can pick
  another target;
- `stop` is always allowed.

It needs no extra dependencies and defaults to `mock://turtlebot4`, so you can
try it on a laptop without ROS.

> **Availability:** `apyrobo mcp` is newer than the latest PyPI release
> (4.0.0), so `pip install apyrobo` does not include it yet. Until the next
> release, run it from GitHub as shown below. After that release the shorter
> form `uvx apyrobo mcp` will work too.

## Try it in a minute

With [uv](https://docs.astral.sh/uv/) installed (and `git`), one command
fetches APYROBO from GitHub into a throwaway environment and starts the
server:

```bash
uvx --from git+https://github.com/apyrobo/Apyrobo apyrobo mcp --max-speed 0.5 --no-go 2,2,4,4
```

It prints one line to stderr and then waits for an MCP client on stdio
(Ctrl-C to quit); normally your client starts it for you. With
[pipx](https://pipx.pypa.io) instead:

```bash
pipx run --spec git+https://github.com/apyrobo/Apyrobo apyrobo mcp --max-speed 0.5 --no-go 2,2,4,4
```

Or install it into your own environment:

```bash
pip install "git+https://github.com/apyrobo/Apyrobo"
apyrobo mcp --max-speed 0.5 --no-go 2,2,4,4
```

Then add it to your client (below) and ask something like:

> Check the robot's status, drive to (1, 1) at 2 m/s, then drive to (3, 3).

The model sees the first move's speed clamped to 0.5 m/s, and the move to
(3, 3) refused because that point is inside the no-go zone from (2, 2) to
(4, 4). Ask for "deliver the package to (5, 1)" and it plans and runs a
four-step task.

To see what a client gets back before installing anything, read the
[safety demo transcript](mcp_demo_transcript.md): a scripted session against
the real server on `mock://turtlebot4` with a normal move, a clamped move, a
refused no-go move, `stop` and `safety_report`. From a clone you can replay it
with `python examples/05_mcp_safety_demo.py`; a test keeps the committed
transcript in step with the script.

### Claude Code

```bash
claude mcp add apyrobo -- uvx --from git+https://github.com/apyrobo/Apyrobo apyrobo mcp --max-speed 0.5 --no-go 2,2,4,4
```

If you installed with `pip`, use `-- apyrobo mcp --max-speed 0.5 --no-go 2,2,4,4`
instead.

### Claude Desktop

Add to `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "apyrobo": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/apyrobo/Apyrobo",
        "apyrobo", "mcp", "--max-speed", "0.5", "--no-go", "2,2,4,4"
      ]
    }
  }
}
```

Then fully quit Claude Desktop (from the menu bar or system tray, not just
closing the window) and open it again; it only reads the config at start-up.
If you installed with `pip`, use `"command": "apyrobo"` and
`"args": ["mcp", "--max-speed", "0.5", "--no-go", "2,2,4,4"]`.

### Cursor

Add the same `mcpServers` block to `~/.cursor/mcp.json` (or `.cursor/mcp.json`
in a project).

GUI clients don't inherit your shell's `PATH`. If the client can't find
`uvx` or `apyrobo`, use the full path to the executable as `command` (run
`which uvx` or `which apyrobo` to get it, for example
`/home/you/.local/bin/uvx`).

The first start with `uvx` downloads and builds APYROBO (about 15 seconds
in our test); later starts reuse uv's cache.

## Tools

| Tool | Moves the robot | What it does |
| --- | --- | --- |
| `robot_status` | no | Robot id, position (m), heading (rad), health, active safety policy |
| `list_capabilities` | no | What the robot can do and which sensors it has |
| `move_to(x, y, speed?)` | yes | Drive to a point; speed clamped, no-go targets refused |
| `rotate(angle_rad, speed?)` | yes | Rotate in place |
| `stop()` | stops | Stop all motion; always allowed |
| `plan_task(task)` | no | Natural-language task → ordered skills, parameters, confidence |
| `execute_task(task)` | yes | Plan and run; stops at the first refused step and says why |
| `safety_report()` | no | Every clamp and refusal since the server started |

`plan_task` and `execute_task` use the built-in rule planner by default, so no
API key is needed. The MCP client's own model does the reasoning; APYROBO
checks and executes.

## Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--robot URI` | `mock://turtlebot4` | Any APYROBO robot URI (`ros2://…`, `gazebo://…`, `mujoco://…`, `vda5050://…`) |
| `--policy` | `default` | Base policy: `default` (1.5 m/s) or `strict` (0.5 m/s, speed ramp) |
| `--max-speed M_S` | from policy | Override the speed limit |
| `--no-go X_MIN,Y_MIN,X_MAX,Y_MAX` | none | Add a rectangular no-go zone in metres; repeat for more |
| `--provider` | `rule` | Planner for `plan_task` / `execute_task` |

## On a ROS 2 robot or a Gazebo sim

`mock://` is for trying it out. To drive a real ROS 2 robot, or a Gazebo
simulation of one, give the server a `ros2://` URI:

```bash
apyrobo mcp --robot ros2://burger --max-speed 0.22 --no-go 0,0,1,1
```

Run it in a shell where ROS 2 is sourced (`rclpy` and `nav2_msgs` must
import) and on the same `ROS_DOMAIN_ID` as the robot. The `ros2://` adapter
subscribes to `/odom` for the pose that `robot_status`, `move_to` and the
safety checks use, and sends each `move_to` to Nav2 as a `navigate_to_pose`
goal. If no Nav2 action server is found it falls back to driving `/cmd_vel`
directly, which has no obstacle avoidance. Clamps and no-go refusals happen
in APYROBO before the goal reaches Nav2, so a refused target is never sent.
If Nav2 rejects or aborts the goal, never answers it, navigation times out,
or `stop` cancels it, `move_to` comes back as an error (`isError`) naming
the reason and the robot's current position, not as a success.

**What CI checks.** On every push,
[`tests/integration/test_gazebo_nav2_e2e.py`](../tests/integration/test_gazebo_nav2_e2e.py)
starts `apyrobo mcp --robot ros2://burger` against a physics-simulated
TurtleBot3 with Nav2 and slam_toolbox (ROS 2 Humble, Gazebo Classic) and
talks to it over stdio as an MCP client would: an over-limit `move_to` is
clamped and reported, goes through Nav2 and gets the robot to the target
region; a target in the no-go zone is refused and the robot does not move;
`stop` and `safety_report` answer. To run the same stack locally with
Docker:

```bash
docker compose -f docker/docker-compose.yml --profile gazebo-nav up \
    --build --abort-on-container-exit --exit-code-from gazebo-nav-test
```

Not covered yet: real hardware in CI, and the ROS 2 Jazzy + Gazebo Harmonic
image ([`docs/gz_sim_port.md`](gz_sim_port.md)) is not wired into CI.

## What this does and doesn't guarantee

The enforcer checks each command against the policy you set (speed limits and
no-go zones here; see [`apyrobo/safety/enforcer.py`](../apyrobo/safety/enforcer.py)
for the full list). It is a software guard between a model and a robot, not a
certified safety system: keep hardware e-stops and site safety measures in
place, and try new setups on `mock://` or a simulator first.

The server speaks MCP over stdio. Clients that open with `initialize`
(protocol versions 2025-11-25, 2025-06-18, 2025-03-26 and 2024-11-05) get
`initialize`, `ping`, `tools/list` and `tools/call`; bad tool arguments come
back as tool errors (`isError`) the model can read and correct, as 2025-11-25
asks. Clients on protocol 2026-07-28 send no
handshake: each request carries its version in `params._meta`, and the server
answers `server/discover`, `tools/list` and `tools/call` statelessly. The
2026-07-28 support is checked against the specification text by unit tests,
not yet against a shipping 2026-07-28 client.

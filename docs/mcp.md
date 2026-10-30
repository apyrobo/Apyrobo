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
> (4.1.0). Until the next release, install from GitHub:
> `pip install "git+https://github.com/apyrobo/Apyrobo"`.

## Try it in a minute

```bash
pip install "git+https://github.com/apyrobo/Apyrobo"
apyrobo mcp --max-speed 0.5 --no-go 2,2,4,4   # Ctrl-C to quit; a client normally starts it
```

Then add it to your client (below) and ask something like:

> Check the robot's status, drive to (1, 1) at 2 m/s, then drive to (3, 3).

The model sees the first move's speed clamped to 0.5 m/s, and the move to
(3, 3) refused because that point is inside the no-go zone from (2, 2) to
(4, 4). Ask for "deliver the package to (5, 1)" and it plans and runs a
four-step task.

### Claude Code

```bash
claude mcp add apyrobo -- apyrobo mcp --max-speed 0.5 --no-go 2,2,4,4
```

### Claude Desktop

Add to `claude_desktop_config.json` (Settings → Developer → Edit Config), then
restart Claude Desktop:

```json
{
  "mcpServers": {
    "apyrobo": {
      "command": "apyrobo",
      "args": ["mcp", "--max-speed", "0.5", "--no-go", "2,2,4,4"]
    }
  }
}
```

### Cursor

Add the same `mcpServers` block to `~/.cursor/mcp.json` (or `.cursor/mcp.json`
in a project).

If `apyrobo` is installed in a virtualenv, use the full path to the
executable as `command` (for example `/home/you/.venvs/robot/bin/apyrobo`),
because GUI clients don't inherit your shell's `PATH`.

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

## What this does and doesn't guarantee

The enforcer checks each command against the policy you set (speed limits and
no-go zones here; see [`apyrobo/safety/enforcer.py`](../apyrobo/safety/enforcer.py)
for the full list). It is a software guard between a model and a robot, not a
certified safety system: keep hardware e-stops and site safety measures in
place, and try new setups on `mock://` or a simulator first.

The server speaks MCP over stdio (protocol versions 2025-06-18, 2025-03-26 and
2024-11-05) and implements `initialize`, `ping`, `tools/list` and `tools/call`.

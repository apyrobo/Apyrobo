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
four-step task from where the robot stands.

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

### Let apyrobo print the snippet

```bash
apyrobo mcp config claude-desktop --max-speed 0.5 --no-go 2,2,4,4
```

prints the block above with the absolute path of `uvx` on this machine (GUI
clients don't read your shell's `PATH`) and says where the config file is on
your OS. Use `cursor` or `claude-code` instead of `claude-desktop` for those
clients, and `--launcher apyrobo` to start the installed `apyrobo` rather
than `uvx`. It takes the same `--robot`, `--policy`, `--policy-file`,
`--max-speed`, `--no-go`, `--allow-area`, `--confirm-moves` and `--audit-log`
flags as `apyrobo mcp`, checks them, and makes relative `--policy-file` and
`--audit-log` paths absolute. It only prints: merge the output
into your config yourself. The `uvx` command it prints installs from GitHub
until `apyrobo mcp` is in a PyPI release, then `uvx apyrobo mcp`: the form
follows your installed version. Without installing, run it as
`uvx --from git+https://github.com/apyrobo/Apyrobo apyrobo mcp config claude-desktop`.

If the tools don't show up, see [Troubleshooting](#troubleshooting).

### Client on your laptop, robot elsewhere

ROS 2 usually runs on the robot and the MCP client on a laptop. MCP over stdio
crosses that gap through plain ssh, with no port to open:

```bash
apyrobo mcp config claude-desktop --ssh ubuntu@robot.local \
    --ssh-setup /opt/ros/jazzy/setup.bash \
    --robot ros2:// --max-speed 0.5 --no-go 2,2,4,4
```

prints a config whose `command` is `ssh` (`-T`, `BatchMode=yes`) running
`apyrobo mcp` on the robot, so the safety policy is enforced there. What's
different from a local config:

- **Key-based login is required.** Nothing can answer a password prompt over
  stdio. `ssh -o BatchMode=yes ubuntu@robot.local true` must succeed on the
  laptop first.
- **`apyrobo` must be installed on the robot** and on the `PATH` of a
  non-interactive ssh shell. `--ssh-setup PATH` names a script on the robot to
  source first: a non-interactive shell has not sourced ROS 2, and `ros2://`
  needs it. Add your own `export PATH=…` to a script of yours if `apyrobo` is
  in `~/.local/bin`. `--launcher uvx` runs `uvx` on the robot instead.
- **`--policy-file` is a path on the robot,** absolute, and isn't checked from
  the laptop. The other flags are checked as usual.
- **A custom port** goes in `~/.ssh/config` (`Host`, `Port`); `--ssh` takes the
  host name.
- **Nothing on the robot's `stdout` but the server.** A start-up file that
  prints text breaks the MCP stream.

Run the printed `ssh` command in a terminal first: the server's start-up line
appears, then it waits for a client.

### When the client goes away

`apyrobo mcp` stops the robot when its client goes away: when stdin closes,
even during a move, and on `SIGTERM` or `SIGHUP`. A motion request already read
but not started is refused. So input piped in and then closed
(`printf ... | apyrobo mcp`) has its moves refused; a client must keep stdin open
until it has its replies. This is tested on `mock://` and a slow test adapter,
not yet on a real Nav2 goal.

A dead network link is different. Until TCP gives up, the server sees no end of
input, so set `ClientAliveInterval` and `ClientAliveCountMax` in the robot's
`sshd_config` to bound the wait. Keep your robot's own e-stop within reach.

### Check your limits first

`apyrobo mcp selftest` takes the same policy flags as the server and prints
what they do on `mock://`, before you touch a client config. It lists the
tools, runs a normal move, shows an over-limit move clamped to your limit, and
tries a move into each no-go zone. With allowed areas set, it also tries a
target outside them. It exits 0 if all of that held, 1 if not.

```bash
apyrobo mcp selftest --max-speed 0.5 --no-go 2,2,4,4
```

It refuses any robot other than `mock://`, so it cannot move hardware. It does
not exercise `--confirm-moves`, which needs a client that can ask you.

A pass only says the limits you wrote down hold. So the report also warns about
what your policy leaves unset: no no-go zones, no allowed area, the built-in
speed limits, and `--confirm-moves` off (pass that flag to `selftest` too if you run the server
with it). The warnings do not change the exit code. They cannot cover what is
missing from your map: a zone you never wrote down looks like free space.

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

The rule planner has no map, so a drive needs coordinates in the task: `(3, 3)`,
`3 3` after "to" or "at", or `x=3 y=3`. A task without them, such as "go to the
kitchen", moves nothing: `execute_task` returns an error saying what to give,
and `plan_task` returns `can_proceed: false` with the same reason. A delivery
with one point starts from where the robot is; "deliver from (1, 1) to (5, 1)"
names both. A speed in the task is not read, so use `move_to` to set one.

## Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--robot URI` | `mock://turtlebot4` | Any APYROBO robot URI (`ros2://…`, `gazebo://…`, `mujoco://…`, `vda5050://…`) |
| `--policy` | `default` | Base policy: `default` (1.5 m/s) or `strict` (0.5 m/s) |
| `--policy-file PATH` | none | JSON or YAML policy (`name`, `base`, `max_speed`, `max_angular_speed`, `no_go_zones`, `allowed_areas`); flags on the command line override it |
| `--max-speed M_S` | from policy | Override the speed limit |
| `--no-go X_MIN,Y_MIN,X_MAX,Y_MAX` | none | Add a rectangular no-go zone in metres; repeat for more |
| `--allow-area X_MIN,Y_MIN,X_MAX,Y_MAX` | none | Add a rectangle move targets may be in; with any set, a target outside all of them is refused; repeat for more |
| `--confirm-moves` | off | Ask the user to confirm every `move_to`, `rotate` and `execute_task` before the robot moves (see below) |
| `--audit-log PATH` | off | Append one JSON line per motion decision to `PATH`; if a line cannot be written, motion is refused (see below) |
| `--provider` | `rule` | Planner for `plan_task` / `execute_task` |

For a starting point, copy a commented policy from
[`examples/policies/`](../examples/policies/) (warehouse aisle, lab bench, home
with stairs), edit the zones for your map, and check it with
`apyrobo mcp selftest --policy-file your_policy.yaml`.

### Deny by default with allowed areas

A no-go zone only covers the places you remembered. With at least one allowed
area, set by `--allow-area` or `allowed_areas` in a policy file, the rule flips:
a target outside every allowed area is refused, so a place you never wrote down
is off limits too. Unset, targets outside the no-go zones are accepted as
before. No-go zones still win inside an allowed area, and `--allow-area` adds
to the file's areas. An `allowed_areas` key with an empty list is an error, not
"anywhere".

```bash
apyrobo mcp --max-speed 0.5 --allow-area 0,0,10,6 --no-go 2,2,4,4
```

### Confirm every move

With `--confirm-moves`, each motion command waits for a yes from the user.
The server asks through MCP elicitation and shows the target and the speed
the policy will actually allow (after clamping); declining leaves the robot
where it is and returns a plain `declined` result. Approval never lifts a
policy limit, and a target in a no-go zone is refused without asking.
`stop` and the read-only tools never ask.

This works with clients that declare form elicitation, in both protocol
eras: on 2026-07-28 through the multi-round-trip `input_required` result,
and in sessions that open with `initialize` (2025-06-18 and later) through
an `elicitation/create` request the server sends mid-call and waits on.
Both are tested against the official MCP Python SDK client. Clients that
lack elicitation get every motion command refused with a readable error,
because the operator asked for confirmation and the server cannot get it
from them. `apyrobo mcp config <client> --confirm-moves` puts the flag in
the generated config.

### Keep an audit log

With `--audit-log PATH`, every motion decision is appended to `PATH` as one
JSON line, synced to disk before the server goes on. The file is opened for
each line, so you can rotate it while the server runs.

```bash
apyrobo mcp --max-speed 0.5 --no-go 2,2,4,4 --audit-log ~/apyrobo-audit.jsonl
```

```json
{"time":"2026-11-04T10:15:02.114+00:00","tool":"move_to","arguments":{"x":1.0,"y":0.0,"speed":9},"decision":"started","enforced_speed":null,"policy":"default+cli"}
{"time":"2026-11-04T10:15:02.931+00:00","tool":"move_to","arguments":{"x":1.0,"y":0.0,"speed":9},"decision":"clamped","enforced_speed":0.5,"policy":"default+cli"}
```

Each line has `time` (UTC), `tool` (`move_to`, `rotate`, `execute_task` or
`stop`), `arguments` as the client sent them, `decision`, `enforced_speed`, the
speed the robot was given where one applies, and `policy`. A refusal or failure
adds `detail`, cut to 500 characters. Read-only tools are not logged.

| `decision` | Meaning |
| --- | --- |
| `started` | The motion call passed the client checks and is about to run; the next line says how it ended |
| `ran` | It finished as asked (`stop` logs only this) |
| `clamped` | It finished with a speed the policy lowered |
| `refused` | The policy said no (a no-go zone, an allowed area), the client had gone, or it cannot confirm; earlier steps of an `execute_task` may have run |
| `failed` | Bad arguments or an adapter error |
| `declined` | The user said no under `--confirm-moves` |
| `client_lost` | The client went away and the server stopped the robot; `detail` says how, and whether the stop failed |

`started` is written before the robot is asked to move, and a move whose line
cannot be written is refused, so a motion command never runs unlogged. A path
that cannot be written stops the server at start. A log that breaks during a
move shows at the next one: that move has already run and only its closing line
is lost, and the next motion command is refused. `stop` always runs, logged or
not.

The log is a plain file on the machine that runs the server, with no signature
or hash chain, so whoever can write there can change it. `apyrobo mcp config
<client> --audit-log PATH` puts the flag in the generated config.

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
in APYROBO before the goal reaches Nav2, so a refused move is never sent.
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
a clear target whose Nav2 route crosses a zone is refused and no goal is sent;
`stop` and `safety_report` answer. To run the same stack locally with
Docker:

```bash
docker compose -f docker/docker-compose.yml --profile gazebo-nav up \
    --build --abort-on-container-exit --exit-code-from gazebo-nav-test
```

Not covered yet: real hardware in CI, and the ROS 2 Jazzy + Gazebo Harmonic
image ([`docs/gz_sim_port.md`](gz_sim_port.md)) is not wired into CI.

### Write a Nav2 keepout mask

Experimental. This turns the zones the server refuses into a mask for Nav2's
keepout filter, so that Nav2 plans and drives around them:

```bash
apyrobo mcp keepout --policy-file your_policy.yaml --out ~/maps
```

It takes `--policy-file` and `--no-go` as the server does, plus `--resolution`
(metres per pixel, default 0.05). It writes `keepout_mask.pgm` and
`keepout_mask.yaml` to `--out`, replacing any earlier pair, and prints the Nav2
parameters to paste: two servers, which need a lifecycle manager listing both,
and the filter for both costmaps. Run it again whenever the zones change, and
restart those servers.

A pixel goes black if any part of it touches a zone, so a zone can grow by up to
a pixel on each side but not shrink. The mask is in the map frame, covering the
zones plus 1 m. Allowed areas are not drawn.

What has been tested: Nav2's map server on Jazzy and Humble loaded a mask, published
it with the cells in the right place, and the filter info server took the printed
parameters. What has not: no test yet has Nav2 route around a zone with
this mask, so the costmap block is untested, and the feature stays experimental
until one does. Check it in simulation first.

## Troubleshooting

**Check the server on its own first.** Run the same command your client
config uses in a terminal, for example
`uvx --from git+https://github.com/apyrobo/Apyrobo apyrobo mcp`. A working
server prints one line to stderr (`apyrobo mcp — robot=… (waiting for an MCP
client on stdio)`) and then waits; Ctrl-C quits. If it prints `Error: …`
instead, fix that before looking at the client.

**The client can't start it (`command not found`, `spawn uvx ENOENT`).** GUI
clients such as Claude Desktop and Cursor don't run your shell's start-up
files, so a `PATH` set in `~/.zshrc` or `~/.bashrc` doesn't reach them. Use
the full path to the executable as `command`: run `which uvx` (or
`which apyrobo`) in a terminal and paste the result, for example
`/home/you/.local/bin/uvx` or `/Users/you/.local/bin/uvx`.

**The tools don't appear in Claude Desktop.** It reads the config only at
start-up, so quit it fully (menu bar or system tray) and reopen it. Open the
file through Settings → Developer → Edit Config, so you edit the one it
reads (`~/Library/Application Support/Claude/claude_desktop_config.json` on
macOS, `%APPDATA%\Claude\claude_desktop_config.json` on Windows), and check
it is valid JSON (a stray trailing comma is the usual culprit). If the `apyrobo` entry
is missing after a restart, add it again with the app closed.

**The first start is slow or times out.** The first `uvx` start downloads
and builds APYROBO (about 15 seconds in our test); later starts reuse uv's
cache. If your client gives up during that first start, run the `uvx`
command once in a terminal to fill the cache, then restart the client.

**Where the logs are.** The server writes its start-up line and any errors
to stderr; stdout carries only MCP messages. Claude Desktop saves each
server's stderr in `~/Library/Logs/Claude/` (macOS) or
`%APPDATA%\Claude\logs\` (Windows), in a file named after the server
(`mcp-server-apyrobo.log`). In Claude Code, `claude mcp list` shows whether
the server connected, and `/mcp` in a session shows its status.

**`ros2://`: "a ROS 2 import failed".** The server could not import `rclpy`,
so it exits at start. Two causes, often together.

*The environment isn't sourced.* A client starts the server with its own
environment, not your terminal's, so a ROS 2 setup you sourced in a shell
doesn't apply. Wrap the command so it sources ROS 2 first, for example
`"command": "bash"` with
`"args": ["-c", "source /opt/ros/humble/setup.bash && exec apyrobo mcp --robot ros2://burger"]`
(use your distro's `setup.bash` and the full path to `apyrobo` if needed).

*`uvx` runs in an isolated environment.* Sourcing `setup.bash` puts `rclpy`
and the ROS message packages on `PYTHONPATH`, which `uvx` passes through, but
the Python packages apt installs next to them — `numpy` and `transforms3d` —
live in `/usr/lib/python3/dist-packages`, which it does not. `uvx` may also
pick a different Python from the one `rclpy` was built for, and then the
import fails on `rclpy._rclpy_pybind11` or hangs. Pin the interpreter to your
distro's (3.12 on Jazzy, 3.10 on Humble) and add the two packages:

```bash
source /opt/ros/jazzy/setup.bash
uvx --no-managed-python --python 3.12 --with 'numpy<2' --with transforms3d \
  --from git+https://github.com/apyrobo/Apyrobo apyrobo mcp --robot ros2://burger
```

Putting the apt packages on the path instead of the two `--with` flags works
as well (`export PYTHONPATH="$PYTHONPATH:/usr/lib/python3/dist-packages"`).
In a client config, both forms go inside the wrapped `bash -c` command above;
`UV_PYTHON=3.12` and `UV_PYTHON_PREFERENCE=only-system` in an `env` block are
the same pin. Checked 4 November 2026 in the `osrf/ros:jazzy-desktop` image
on uv 0.11.32; without the pin, an installed `uvx`-managed Python was used
instead of the distro's.

`ros2://` also needs the `nav2_msgs` and `tf_transformations` packages, which
that image does not carry:
`sudo apt install ros-$ROS_DISTRO-nav2-msgs ros-$ROS_DISTRO-tf-transformations`.
Installing APYROBO into the ROS 2 Python instead of running it through `uvx`
(`pip install` in a `--system-site-packages` virtualenv, then `apyrobo mcp`)
avoids the pin and the `--with` flags.

**`ros2://`: the robot never moves, or its position doesn't change.** The
server is most likely on a different `ROS_DOMAIN_ID` from the robot, so it
sees no `/odom` and no Nav2. Set the same value for the server, either in
the wrapped command above or in the client's `env` block
(`"env": {"ROS_DOMAIN_ID": "30"}` in the JSON config, or
`claude mcp add -e ROS_DOMAIN_ID=30 …` for Claude Code). Check from the same
shell with `ros2 topic echo /odom --once`.

## How APYROBO fits with other robot MCP servers

Several MCP servers let a model drive a robot. They solve different problems,
and you can pick by what you are driving. This table states only what each
project's README and code show on their main branch, checked 3 November 2026;
open an issue if it has gone stale.

| | What it drives | Speed limits | No-go zones | Asks a human to confirm | Try without a robot | Install |
|---|---|---|---|---|---|---|
| [ROS-MCP Server](https://github.com/robotmcp/ros-mcp-server) | Any ROS 1 or ROS 2 robot through rosbridge: topics, services, actions, parameters, camera images | None in the server | None in the server | No | Examples use turtlesim or Isaac Sim (ROS needed) | `uvx ros-mcp`, plus rosbridge on the robot |
| [Nav2 MCP Server](https://github.com/ajtudela/nav2_mcp_server) | Nav2 on ROS 2 Jazzy: `navigate_to_pose`, `follow_waypoints`, spin, back up, dock, costmaps, lifecycle | Back-up speed capped at 1.0 m/s; navigation speed is whatever Nav2's controller is configured for | None in the server | No | Needs ROS 2 and Nav2 (robot or sim) | `git clone` + `uv sync`, or its Docker image |
| [Amazing ROS 2 MCP](https://github.com/proxi666/amazing-ros2-mcp) | Any ROS 2 robot through native `rclpy`, no rosbridge: topics, services, actions, parameters, camera images, Nav2 goals and waypoints | `/cmd_vel` values clamped to limits set in `config.py`; Nav2 goals use the controller's speed | None in the server; it can block listed topics and has a dry-run mode | No | Needs ROS 2 (robot or sim) | `git clone` + `pip install -e .` |
| [armguard-mcp](https://github.com/Lynn-hh/armguard-mcp) | ROS 2 robot arms (MoveIt 2, `ros2_control`; Franka FR3): plan, preview, execute | Speed and step-size caps, plus joint, force and rate limits | Workspace box and keep-out zones, checked along the whole tool path | Yes, via MCP elicitation, for moves outside the envelope | Bundled simulated Franka FR3, no ROS | `uvx armguard-mcp` |
| APYROBO (`apyrobo mcp`) | Mobile robots: `mock://`, or a ROS 2 robot through Nav2 (`ros2://`) | Clamped to `--max-speed` (and `max_angular_speed` in a `--policy-file`) | Rectangles from `--no-go` or `--policy-file`; a move into or across one is refused, and on `ros2://` Nav2's planned route is checked before the goal is sent | Opt-in `--confirm-moves`, via MCP elicitation; refuses motion if the client can't ask | `mock://turtlebot4`, no ROS | `uvx --from git+https://github.com/apyrobo/Apyrobo apyrobo mcp` (`uvx apyrobo mcp` after the next release) |

In short:

- **A robot arm:** armguard-mcp is built for manipulators; APYROBO is not.
- **Full ROS access** (inspect and call any topic, service or parameter):
  ROS-MCP Server, or Amazing ROS 2 MCP if you don't want rosbridge.
- **Nav2's actions as tools:** Nav2 MCP Server gives an agent Nav2's actions;
  APYROBO puts speed limits, no-go zones and confirmation in front of the
  robot.
- **A mobile robot where the model must stay inside limits you set:**
  APYROBO, and you can try the limits on `mock://` before connecting the real
  robot.

## What this does and doesn't guarantee

The enforcer checks each command against the policy you set (speed limits and
no-go zones here; see [`apyrobo/safety/enforcer.py`](../apyrobo/safety/enforcer.py)
for the full list). It is a software guard between a model and a robot, not a
certified safety system: keep hardware e-stops and site safety measures in
place, and try new setups on `mock://` or a simulator first.

It checks policy, not live sensors. A move is judged against the numbers and
zones you wrote, the robot's reported position and, with Nav2, the planned
route. Nothing refuses a move because of what a scanner or camera sees now, and
the enforcer's human-proximity check does not run under `apyrobo mcp`, because
the server gives it no detections to read. Stopping for a person or an obstacle
stays with Nav2's costmaps and the robot's own protection.

The tool list is the fixed one under [Tools](#tools). The server exposes no raw
ROS topic, service or action call, so a model cannot publish to a topic through
it.

No-go zones are checked at the target on every adapter. The route is checked
too where the robot drives a straight line: `mock://`, `gazebo_native://`,
`mujoco://`, `ros2://` without Nav2 and the other adapters without a planner
refuse a move whose straight line enters a zone, and a robot already inside a
zone may drive out. That check assumes the adapter reports a true position and
drives straight, so it is no substitute for a planner that knows your map.

On `ros2://` with Nav2, and with zones set, the server asks Nav2 for its route
through the `ComputePathToPose` action before sending the goal, and refuses the
move if any stretch of that route enters a zone. If no plan comes back
(`plan_timeout_sec`, 10 seconds by default), the plan is empty, or it holds
non-finite points, the move is refused rather than sent. A robot already inside a zone may
drive out. With no zones set, no plan is requested.

This checks the plan, not the drive: Nav2's controller may stray from the plan
and Nav2 may replan during the move, and those paths are not checked. If the
robot must never enter an area, add a Nav2 keepout filter to both the global
and the local costmap, so that planning and driving both respect it; it is the
stronger layer. `apyrobo mcp keepout` writes one from the same zones (see
[Write a Nav2 keepout mask](#write-a-nav2-keepout-mask)); wiring it into Nav2 is
yours to do.

Allowed areas are checked at the target only, on every adapter. The route is
not compared with them: a robot sent from one allowed area to another may pass
through space outside both, and Nav2 may route outside an area to reach a target
inside it.

Moving from `mock://` to hardware? [Your first real robot](first-real-robot.md)
goes through these checks in order, and lists what they do not cover.

The server speaks MCP over stdio. Clients that open with `initialize`
(protocol versions 2025-11-25, 2025-06-18, 2025-03-26 and 2024-11-05) get
`initialize`, `ping`, `tools/list` and `tools/call`; bad tool arguments come
back as tool errors (`isError`) the model can read and correct, as 2025-11-25
asks. Clients on protocol 2026-07-28 send no
handshake: each request carries its version in `params._meta`, and the server
answers `server/discover`, `tools/list` and `tools/call` statelessly. The
2026-07-28 support is checked against the specification text by unit tests,
and both eras are tested in CI against the official MCP Python SDK client
(`mcp` 2.x, `tests/test_mcp_official_client.py`): it connects, lists the
tools, and sees a normal move run, an over-limit move clamped and a no-go
move refused. Desktop hosts (Claude Desktop, Cursor) have not been tried on
2026-07-28 yet.

# TypeScript → Wire Protocol → Robot

The cross-language interop showcase: a Node script uses the
[reference TypeScript client](../../packages/apyrobo-client-ts) — zero
dependencies, no Python imports — to submit a natural-language task to the
Python reference server. The server plans it and, in `--execute` mode, runs
it on the robot; the outcome comes back in `metadata.execution`. Everything
that crosses the language boundary is spec-1.0 JSON
([spec/wire-protocol.md](../../spec/wire-protocol.md)) — the
"it's a protocol, not a library" proof point.

Execute mode stays inside the frozen spec: the response status is still
`"planned"`, and the execution report rides in an extra metadata key, which
§1 requires clients to tolerate. A client that has never heard of execute
mode still works unchanged.

## Run it (mock robot, laptop, seconds)

```bash
pip install apyrobo
cd packages/apyrobo-client-ts && npm install && npm run build && cd ../..
node demos/ts_interop/demo.mjs
```

The script spawns `apyrobo serve --execute` over stdio, submits
*"deliver package from (1, 2) to (5, 5)"*, prints the plan the Python
server returned, and the execution outcome. CI runs exactly this on every
commit (the `ts-client` job).

## Run it against a physics robot (Gazebo, Linux)

The same demo, but the server sits in a container with a live Nav2 stack
and the task drives a physics-simulated TurtleBot3 — the TypeScript side is
byte-for-byte identical, only the URL and robot URI change:

```bash
docker compose -f docker/docker-compose.yml --profile gazebo-nav-interop up -d
# wait until port 8765 is listening (the server opens it after ros2://burger discovery)
node demos/ts_interop/demo.mjs --ws ws://localhost:8765 \
    --robot ros2://burger --task "navigate to (-1.2, -0.5)" --timeout 300000
```

CI runs this in the `gazebo-nav-interop` job
([integration.yml](../../.github/workflows/integration.yml)).

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--robot <uri>` | `mock://interop_bot` | Target robot URI, passed on the wire |
| `--task <text>` | `deliver package from (1, 2) to (5, 5)` | Natural-language task |
| `--ws <url>` | *(spawn stdio server)* | Connect to a running WebSocket server instead |
| `--timeout <ms>` | `120000` | Give up waiting for the response |

`APYROBO_SERVE` overrides the spawned server command (default
`apyrobo serve`), e.g. `APYROBO_SERVE="python -m apyrobo serve"`.

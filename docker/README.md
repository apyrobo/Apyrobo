# APYROBO Demo Environment

Clone the repo and run:

```bash
docker compose -f docker/docker-compose-demo.yml up
```

Then open http://localhost:8000 in your browser.

Three services start, all on mock robots (no ROS or Gazebo required):

| Service | What it is | Where |
|---------|------------|-------|
| `apyrobo-demo` | Orchestration server: `apyrobo serve --execute` with the rule planner, speaking the [wire protocol](../spec/wire-protocol.md) over WebSocket. It plans each task and runs it on a mock robot through the safety enforcer (1.5 m/s limit). | `ws://localhost:8765` |
| `dashboard` | Web dashboard for its own `mock://turtlebot4` | http://localhost:8000 |
| `mock-fleet` | Three simulated robots (turtlebot4, ur5, spot) that print position and battery lines to the log | `docker compose -f docker/docker-compose-demo.yml logs -f mock-fleet` |

The three run side by side: the dashboard and the server each connect their own
mock robot, and the mock fleet is not shown on the dashboard.

To send the server a task from another terminal, use the TypeScript client demo
(build it once as in [demos/ts_interop](../demos/ts_interop/README.md)):

```bash
node demos/ts_interop/demo.mjs --ws ws://localhost:8765
```

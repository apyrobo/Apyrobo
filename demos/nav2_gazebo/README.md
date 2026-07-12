# Natural Language → Nav2 → Gazebo

The flagship-stack demo, recorded live: a natural-language task —
`"navigate to (-1.2, -0.5)"` — is planned by the rule agent (no LLM, no API
key) and executed through the real **Nav2 `NavigateToPose`** action,
driving a **physics-simulated TurtleBot3** through `turtlebot3_world` with
SLAM building the map as it goes.

![demo](demo.gif)

Nothing in the recording is mocked. It is the same pipeline CI verifies on
every commit ([test_gazebo_nav2_e2e.py](../../tests/integration/test_gazebo_nav2_e2e.py));
the demo just narrates it: discovery, the plan, live position ticks while
Nav2 navigates, and the goal-reached check.

## Run it (Linux)

Gazebo Classic does not run under Docker Desktop on macOS — this demo is a
Linux/CI target (see [tests/integration/README_gazebo.md](../../tests/integration/README_gazebo.md)).

```bash
# Start the sim (gzserver + robot_state_publisher + Nav2 in SLAM mode)
docker compose -f docker/docker-compose.yml --profile gazebo-nav up -d gazebo-nav-sim

# Run the demo against it
docker compose -f docker/docker-compose.yml --profile gazebo-nav \
    run --rm gazebo-nav-demo
```

## Re-record

`./record.sh` (Linux, needs [vhs](https://github.com/charmbracelet/vhs)) —
or dispatch the **Demo Recording** workflow in GitHub Actions, which renders
on a clean runner and uploads `demo.gif` / `demo.mp4` as artifacts.

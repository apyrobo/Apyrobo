# APYROBO Benchmark Suite

Reproducible benchmark comparing APYROBO against raw ROS 2 implementations
across 5 canonical robotics tasks.

## Run locally

```bash
pip install apyrobo
python benchmarks/benchmark_suite.py
python benchmarks/benchmark_suite.py --iterations 50   # more samples
python benchmarks/benchmark_suite.py --json            # machine-readable
```

## Tasks

| Task | APYROBO LOC | Raw ROS 2 LOC | Ratio |
|------|-------------|---------------|-------|
| Navigate to point | 9 | 58 | 6.4× less code |
| Pick-and-place | 14 | 84 | 6.0× less code |
| Patrol route (5 waypoints) | 19 | 63 | 3.3× less code |
| Fleet coordination (3 robots) | 26 | 127 | 4.9× less code |
| Safety policy enforcement | 15 | 96 | 6.4× less code |
| **Total** | **83** | **428** | **5.2× less code** |

Every task gives the planner coordinates it can read (for example
`go to (3, 4)` and `deliver from (1, 1) to (5, 1)`), and every run is checked:
the robot must end where the task said and the safety check must flag exactly
the three violating actions. `success_rate` in the JSON is the share of runs
that completed, and the script exits 1 if any run did not.

## Metrics

- **Setup ms** — cold-start time to init `Robot` + `Agent`
- **Plan ms** — time to generate a skill graph from a task string
- **Exec ms** — time to run the skill graph through the adapter
- **P95 ms** — 95th-percentile execution latency
- **Rec ms** — time to detect a policy violation and substitute a safe action

## CI

Copy `benchmarks/.github/workflows/benchmark.yml` to your repo's `.github/workflows/` to run
benchmarks automatically on every push that touches `apyrobo/` or `benchmarks/`.
Results are posted to the GitHub Actions job summary and saved as artifacts.

## Methodology note

APYROBO timings use `MockAdapter` (no real hardware, no network I/O) to isolate
framework overhead. Raw ROS 2 LOC counts are from representative ROS 2 Python nodes
that implement the same task with standard libraries (nav2_simple_commander, MoveIt
Python API, rclpy action clients). They measure code size, not runtime speed —
ROS 2 startup overhead and message serialization are separate concerns.

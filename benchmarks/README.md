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

## Metrics

- **Setup ms** — cold-start time to init `Robot` + `Agent`
- **Plan ms** — time to generate a skill graph from a task string
- **Exec ms** — time to run the skill graph through the adapter
- **P95 ms** — 95th-percentile execution latency
- **Rec ms** — time to detect a policy violation and substitute a safe action

## CI

Copy `.github/workflows/benchmark.yml` to your repo's `.github/workflows/` to run
benchmarks automatically on every push that touches `apyrobo/` or `benchmarks/`.
Results are posted to the GitHub Actions job summary and saved as artifacts.

## Methodology note

APYROBO timings use `MockAdapter` (no real hardware, no network I/O) to isolate
framework overhead. Raw ROS 2 LOC counts are from representative ROS 2 Python nodes
that implement the same task with standard libraries (nav2_simple_commander, MoveIt
Python API, rclpy action clients). They measure code size, not runtime speed —
ROS 2 startup overhead and message serialization are separate concerns.

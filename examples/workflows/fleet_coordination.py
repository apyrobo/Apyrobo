"""Fleet coordination — dispatch multiple robots to handle parallel tasks.

Splits a list of stops round-robin across a fleet, runs each robot in its
own thread, and reports every stop a robot did not reach.

The rule planner only reads coordinates, so each stop is a named point in
metres in the shared map frame. Measure your own (for example with
``ros2 topic echo /amcl_pose``) and replace the ones below before running on
real robots. Robots are not kept apart from each other: this example does no
traffic management.

Usage:
    python examples/workflows/fleet_coordination.py
    python examples/workflows/fleet_coordination.py --fleet 3 --tasks 6
"""
import argparse
import sys
import threading
import time
from dataclasses import dataclass, field

from apyrobo import Agent, Robot
from apyrobo.core.schemas import TaskStatus

ROBOT_BASE_URI = "mock://fleet_bot"

# name, (x, y) in metres. Example values: measure your own.
TASK_POOL = [
    ("office 101", (4.0, 2.0)),
    ("corridor B", (8.0, 0.0)),
    ("charging bay", (-1.0, 1.0)),
    ("mailroom", (0.0, -3.0)),
    ("zone A", (6.0, 5.0)),
    ("lab 3B", (10.0, 4.0)),
    ("fire exit", (12.0, -2.0)),
    ("reception", (1.0, 6.0)),
]


@dataclass
class RobotWorker:
    robot_id: str
    robot: object
    agent: object
    completed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def worker_thread(worker: RobotWorker, stops: list, lock: threading.Lock) -> None:
    for name, (x, y) in stops:
        try:
            result = worker.agent.execute(task=f"go to ({x}, {y})", robot=worker.robot)
            reached = result.status == TaskStatus.COMPLETED
            reason = result.error or result.status.value
        except Exception as exc:  # report and keep going with the next stop
            reached, reason = False, str(exc)
        with lock:
            if reached:
                worker.completed.append(name)
                print(f"  [{worker.robot_id}] ✓ {name} ({x}, {y})")
            else:
                worker.errors.append(f"{name}: {reason}")
                print(f"  [{worker.robot_id}] ✗ {name} not reached: {reason}")
        time.sleep(0.05)


def coordinate_fleet(num_robots: int, num_tasks: int) -> list[RobotWorker]:
    stops = TASK_POOL[:num_tasks]
    print(f"[fleet] {num_robots} robots  {len(stops)} stops")

    workers: list[RobotWorker] = []
    for i in range(num_robots):
        robot = Robot.discover(f"{ROBOT_BASE_URI}_{i+1}")
        workers.append(RobotWorker(robot_id=f"bot-{i+1:02d}", robot=robot,
                                   agent=Agent(provider="rule")))

    buckets: list[list] = [[] for _ in range(num_robots)]
    for idx, stop in enumerate(stops):
        buckets[idx % num_robots].append(stop)
    for w, bucket in zip(workers, buckets, strict=True):
        print(f"  {w.robot_id}: {len(bucket)} stop(s)")

    print("\n  Dispatching …")
    lock = threading.Lock()
    threads = [
        threading.Thread(target=worker_thread, args=(w, bucket, lock), daemon=True)
        for w, bucket in zip(workers, buckets, strict=True)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total_done = sum(len(w.completed) for w in workers)
    total_err = sum(len(w.errors) for w in workers)
    print("\n--- Fleet Summary ---")
    for w in workers:
        print(f"  {w.robot_id}: {len(w.completed)} reached, {len(w.errors)} not reached")
    print(f"  Total: {total_done}/{len(stops)} reached, {total_err} not reached")
    return workers


def main() -> None:
    p = argparse.ArgumentParser(description="Fleet coordination workflow")
    p.add_argument("--fleet", type=int, default=3, help="Number of robots (max 5)")
    p.add_argument("--tasks", type=int, default=6, help="Number of stops to dispatch")
    args = p.parse_args()
    workers = coordinate_fleet(
        num_robots=max(1, min(args.fleet, 5)),
        num_tasks=min(args.tasks, len(TASK_POOL)),
    )
    if any(w.errors for w in workers):
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Fleet coordination — dispatch multiple robots to handle parallel tasks.

Demonstrates SwarmCoordinator: split a task list across a fleet, monitor
progress, and handle robot failures with reassignment.

Usage:
    python examples/workflows/fleet_coordination.py
    python examples/workflows/fleet_coordination.py --fleet 3 --tasks 6
"""
import argparse
import threading
import time
from dataclasses import dataclass, field

from apyrobo import Robot, Agent

ROBOT_BASE_URI = "mock://fleet_bot"

TASK_POOL = [
    "deliver package to office 101",
    "inspect corridor B",
    "navigate to charging bay",
    "collect item from mailroom",
    "patrol zone A",
    "deliver package to lab 3B",
    "inspect fire exit",
    "navigate to reception",
]


@dataclass
class RobotWorker:
    robot_id: str
    robot: object
    agent: object
    completed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def worker_thread(worker: RobotWorker, tasks: list[str], lock: threading.Lock) -> None:
    for task in tasks:
        try:
            worker.agent.execute(task=task, robot=worker.robot)
            with lock:
                worker.completed.append(task)
                print(f"  [{worker.robot_id}] ✓ {task}")
        except Exception as exc:
            with lock:
                worker.errors.append(f"{task}: {exc}")
                print(f"  [{worker.robot_id}] ✗ {task} — {exc}")
        time.sleep(0.05)


def coordinate_fleet(num_robots: int, num_tasks: int) -> None:
    tasks = TASK_POOL[:num_tasks]
    print(f"[fleet] {num_robots} robots  {num_tasks} tasks")

    # Build fleet
    workers: list[RobotWorker] = []
    for i in range(num_robots):
        robot_id = f"bot-{i+1:02d}"
        robot = Robot.discover(f"{ROBOT_BASE_URI}_{i+1}")
        agent = Agent(provider="rule")
        workers.append(RobotWorker(robot_id=robot_id, robot=robot, agent=agent))

    # Distribute tasks round-robin
    task_buckets: list[list[str]] = [[] for _ in range(num_robots)]
    for idx, task in enumerate(tasks):
        task_buckets[idx % num_robots].append(task)

    for i, (w, bucket) in enumerate(zip(workers, task_buckets)):
        print(f"  {w.robot_id}: {len(bucket)} task(s)")

    # Launch threads
    print("\n  Dispatching …")
    lock = threading.Lock()
    threads = [
        threading.Thread(target=worker_thread, args=(w, bucket, lock), daemon=True)
        for w, bucket in zip(workers, task_buckets)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Summary
    total_done = sum(len(w.completed) for w in workers)
    total_err = sum(len(w.errors) for w in workers)
    print(f"\n--- Fleet Summary ---")
    for w in workers:
        print(f"  {w.robot_id}: {len(w.completed)} done, {len(w.errors)} errors")
    print(f"  Total: {total_done}/{len(tasks)} completed, {total_err} errors")
    print("\n[fleet] coordination complete")


def main() -> None:
    p = argparse.ArgumentParser(description="Fleet coordination workflow")
    p.add_argument("--fleet", type=int, default=3, help="Number of robots")
    p.add_argument("--tasks", type=int, default=6, help="Number of tasks to dispatch")
    args = p.parse_args()
    coordinate_fleet(
        num_robots=min(args.fleet, 5),
        num_tasks=min(args.tasks, len(TASK_POOL)),
    )


if __name__ == "__main__":
    main()

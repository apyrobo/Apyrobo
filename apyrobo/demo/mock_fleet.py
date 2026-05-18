"""Mock fleet — simulates 3 robots for the demo environment.

Run with:
    python -m apyrobo.demo.mock_fleet
"""

import math
import time

from apyrobo.core.robot import Robot

_ROBOTS = [
    ("turtlebot4", "mock://turtlebot4"),
    ("ur5",        "mock://ur5"),
    ("spot",       "mock://spot"),
]

# Simple deterministic position "animation" so the status lines look alive.
_OFFSETS = [0.0, math.pi / 3, 2 * math.pi / 3]
_BATTERIES = [87, 73, 95]


def _pos(index: int, tick: int) -> tuple[float, float]:
    angle = _OFFSETS[index] + tick * 0.1
    return (round(math.cos(angle) * 2 + index * 1.5, 2),
            round(math.sin(angle) * 2, 2))


def _battery(index: int, tick: int) -> int:
    # Slowly drain, wrap at 20 %.
    base = _BATTERIES[index]
    drained = (tick // 10) % (base - 20)
    return base - drained


def run(max_iterations: int | None = None) -> None:
    """Run the mock fleet loop.

    Args:
        max_iterations: Stop after this many ticks (None = run forever).
            Pass a small number in unit tests to avoid infinite loops.
    """
    print("Mock Fleet — 3 robots online", flush=True)

    robots: list[tuple[str, Robot]] = []
    for name, uri in _ROBOTS:
        try:
            robot = Robot.discover(uri)
            robots.append((name, robot))
            print(f"[mock-fleet] connected: {name} ({uri})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[mock-fleet] warning: could not connect {name}: {exc}", flush=True)
            robots.append((name, None))

    tick = 0
    try:
        while max_iterations is None or tick < max_iterations:
            for i, (name, _robot) in enumerate(robots):
                x, y = _pos(i, tick)
                batt = _battery(i, tick)
                print(
                    f"[mock-fleet] {name} pos=({x}, {y}) battery={batt}%",
                    flush=True,
                )
            tick += 1
            if max_iterations is None:
                time.sleep(5)
    except KeyboardInterrupt:
        print("[mock-fleet] shutting down", flush=True)


def main() -> None:
    run()


if __name__ == "__main__":
    main()

"""The odometry watchdog must not stop a robot that is still on its way.

It used to measure divergence as the distance left to the commanded
position, so any move with more than ``watchdog_tolerance`` (2 m by
default) still to go at the first check was stopped mid-way: a ros2:// goal
a few metres off was cancelled after 2 s. Divergence is now how far the
robot has fallen back from its closest approach to the target.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from apyrobo.core.robot import Robot
from apyrobo.safety.enforcer import SafetyEnforcer, SafetyPolicy


class DrivingRobot:
    """A robot whose move() takes time and passes through each point.

    *path* maps the fraction of the move done (0..1) to a position, so a
    test can drive straight at the target or away from it.
    """

    def __init__(self, path: Any = None, steps: int = 30, step_s: float = 0.01) -> None:
        self._robot = Robot.discover("mock://watchdog_progress")
        self.robot_id = "watchdog_progress"
        self.position = (0.0, 0.0)
        self.stopped = threading.Event()
        self.path = path
        self.steps = steps
        self.step_s = step_s

    def __getattr__(self, name: str) -> Any:
        return getattr(self._robot, name)

    def get_position(self) -> tuple[float, float]:
        return self.position

    def stop(self) -> None:
        self.stopped.set()

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        start = self.position
        for i in range(1, self.steps + 1):
            if self.stopped.is_set():
                return
            done = i / self.steps
            if self.path is None:
                self.position = (
                    start[0] + (x - start[0]) * done,
                    start[1] + (y - start[1]) * done,
                )
            else:
                self.position = self.path(done)
            time.sleep(self.step_s)


def _policy(tolerance: float = 0.3) -> SafetyPolicy:
    return SafetyPolicy(
        name="watchdog_progress",
        max_speed=2.0,
        watchdog_tolerance=tolerance,
        watchdog_interval=0.02,
    )


def _watchdog_fired(enforcer: SafetyEnforcer) -> bool:
    return any(i["type"] == "watchdog_triggered" for i in enforcer.interventions)


def test_a_long_move_is_not_stopped_on_its_way() -> None:
    robot = DrivingRobot()
    enforcer = SafetyEnforcer(robot, policy=_policy())  # type: ignore[arg-type]
    try:
        # 3 m to go, tolerance 0.3 m, a check every 20 ms during a 0.3 s move.
        enforcer.move(3.0, 0.0, speed=1.0)
        assert not robot.stopped.is_set()
        assert robot.position == pytest.approx((3.0, 0.0))
        assert not _watchdog_fired(enforcer)
        assert enforcer.watchdog_triggered_count == 0
    finally:
        enforcer.stop_watchdog()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_robot_driving_away_from_its_target_is_stopped() -> None:
    # Commanded to (3, 0), it backs off to (-1, 0) instead.
    robot = DrivingRobot(path=lambda done: (-1.0 * done, 0.0), steps=40)
    enforcer = SafetyEnforcer(robot, policy=_policy())  # type: ignore[arg-type]
    try:
        enforcer.move(3.0, 0.0, speed=1.0)
        assert robot.stopped.is_set()
        assert _watchdog_fired(enforcer)
        fired = [i for i in enforcer.interventions if i["type"] == "watchdog_triggered"]
        assert fired[0]["divergence_m"] > 0.3
    finally:
        enforcer.stop_watchdog()


def test_drift_after_arrival_beyond_the_tolerance_stops_the_robot() -> None:
    robot = DrivingRobot()
    enforcer = SafetyEnforcer(robot, policy=_policy())  # type: ignore[arg-type]
    enforcer.move(3.0, 0.0, speed=1.0)
    enforcer.stop_watchdog()  # check by hand, no timer racing the test

    robot.position = (3.0, 0.5)
    result = enforcer.check_watchdog()
    assert result is not None
    assert not result["ok"]
    assert result["divergence_m"] == pytest.approx(0.5)
    assert robot.stopped.is_set()


def test_drift_after_arrival_within_the_tolerance_is_ok() -> None:
    robot = DrivingRobot()
    enforcer = SafetyEnforcer(robot, policy=_policy())  # type: ignore[arg-type]
    enforcer.move(3.0, 0.0, speed=1.0)
    enforcer.stop_watchdog()

    robot.position = (3.0, 0.2)
    result = enforcer.check_watchdog()
    assert result is not None
    assert result["ok"]
    assert not robot.stopped.is_set()


def test_a_new_move_measures_from_its_own_start() -> None:
    robot = DrivingRobot()
    enforcer = SafetyEnforcer(robot, policy=_policy())  # type: ignore[arg-type]
    try:
        enforcer.move(3.0, 0.0, speed=1.0)
        # Back to the origin: 3 m from the old target, but that is the new one.
        enforcer.move(0.0, 0.0, speed=1.0)
        assert not _watchdog_fired(enforcer)
        assert robot.position == pytest.approx((0.0, 0.0))
    finally:
        enforcer.stop_watchdog()

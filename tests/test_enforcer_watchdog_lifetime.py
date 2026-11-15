"""The odometry watchdog must end once nothing holds the enforcer.

Its timer used to hold the enforcer strongly and re-arm itself every
``watchdog_interval`` until stop(), cancel() or disconnect(), so an enforcer
nobody held any more was never freed and kept polling its robot. Each one
left a timer thread behind; a full test run reached 300+ live threads.
"""

from __future__ import annotations

import gc
import threading
import time
import weakref

from apyrobo.core.robot import Robot
from apyrobo.safety.enforcer import SafetyEnforcer, SafetyPolicy


class CountingRobot:
    """A mock robot that counts get_position() calls."""

    def __init__(self) -> None:
        self._robot = Robot.discover("mock://watchdog_lifetime")
        self.robot_id = "watchdog_lifetime"
        self.position_reads = 0
        self.stops = 0

    def __getattr__(self, name: str):  # noqa: ANN204
        return getattr(self._robot, name)

    def get_position(self) -> tuple[float, float]:
        self.position_reads += 1
        return (0.0, 0.0)

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        pass

    def stop(self) -> None:
        self.stops += 1


def _policy(**kw: float) -> SafetyPolicy:
    return SafetyPolicy(name="lifetime", watchdog_interval=0.02, **kw)


def _wait_until(cond, timeout: float = 2.0) -> bool:  # noqa: ANN001
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def test_held_enforcer_keeps_watching() -> None:
    robot = CountingRobot()
    enforcer = SafetyEnforcer(robot, policy=_policy())
    try:
        enforcer.move(0.0, 0.0)
        reads = robot.position_reads
        assert _wait_until(lambda: robot.position_reads >= reads + 3)
    finally:
        enforcer.cancel()


def test_dropped_enforcer_is_freed_and_stops_polling() -> None:
    robot = CountingRobot()
    # A move timeout shorter than the test, so only the watchdog could keep
    # the enforcer alive once the move timer has fired.
    enforcer = SafetyEnforcer(robot, policy=_policy(move_timeout=0.05))
    enforcer.move(0.0, 0.0)
    ref = weakref.ref(enforcer)
    del enforcer

    def freed() -> bool:
        gc.collect()
        return ref() is None

    assert _wait_until(freed), "the watchdog timer kept the enforcer alive"
    time.sleep(0.1)  # let a tick already in flight finish
    reads = robot.position_reads
    time.sleep(0.2)  # ten watchdog intervals
    assert robot.position_reads == reads


def test_dropped_enforcers_leave_no_timer_threads() -> None:
    robot = CountingRobot()

    def timers_of_this_robot() -> int:
        # Count only the timers of enforcers on this test's robot. Counting
        # every live Timer made the result depend on the watchdogs of other
        # tests' enforcers re-arming, which on a loaded runner moved the
        # total by more than this test's own twenty.
        count = 0
        for t in threading.enumerate():
            if not (isinstance(t, threading.Timer) and t.is_alive()):
                continue
            owner = getattr(t.function, "__self__", None)
            if owner is None and t.args and isinstance(t.args[0], weakref.ref):
                owner = t.args[0]()
            if getattr(owner, "_robot", None) is robot:
                count += 1
            owner = None
        return count

    for _ in range(20):
        SafetyEnforcer(robot, policy=_policy(move_timeout=0.05)).move(0.0, 0.0)

    def settled() -> bool:
        gc.collect()
        return timers_of_this_robot() == 0

    # A leaked enforcer re-arms its watchdog forever, so a longer wait only
    # gives a slow runner time to finish; it never lets a leak pass.
    assert _wait_until(settled, timeout=10.0), (
        f"{timers_of_this_robot()} timer threads outlived their enforcers"
    )


def test_move_timer_keeps_dropped_enforcer_watching_until_timeout() -> None:
    robot = CountingRobot()
    enforcer = SafetyEnforcer(robot, policy=_policy(move_timeout=60.0))
    enforcer.move(0.0, 0.0)
    ref = weakref.ref(enforcer)
    del enforcer
    gc.collect()
    try:
        # Still within its move timeout: the watchdog keeps checking.
        assert ref() is not None
        reads = robot.position_reads
        assert _wait_until(lambda: robot.position_reads >= reads + 3)
    finally:
        held = ref()
        if held is not None:
            held.cancel()

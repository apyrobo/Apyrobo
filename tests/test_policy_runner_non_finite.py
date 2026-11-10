"""PolicyRunner must not pass a non-finite step to the robot.

A diverged policy can output NaN or inf. NaN compares False against
max_step_m, so before this fix such an action skipped the runner's own
bound and reached robot.move() — on a raw robot the episode ended at
(nan, 0.0) reported as a normal 'policy_done'.
"""

from __future__ import annotations

import math

import pytest

from apyrobo import Robot
from apyrobo.skills.policy import PolicyRunner


class _OneShot:
    def __init__(self, action):
        self.action = action
        self.calls = 0

    def select_action(self, obs):
        self.calls += 1
        return self.action if self.calls == 1 else None


class _RecordingRobot:
    """Robot-shaped stub that records every move it is sent."""

    def __init__(self, pos=(0.0, 0.0)):
        self.pos = pos
        self.moves: list[tuple[float, float]] = []
        self.stopped = 0

    def get_position(self):
        return self.pos

    def move(self, x, y, speed=None):
        self.moves.append((x, y))
        self.pos = (x, y)

    def stop(self):
        self.stopped += 1


NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize(
    "action",
    [
        {"dx": NAN},
        {"dy": NAN},
        {"dx": INF},
        {"dx": -INF, "dy": 0.0},
        {"x": NAN, "y": 0.0},
        {"x": 0.0, "y": INF},
    ],
)
def test_non_finite_action_never_reaches_the_robot(action):
    robot = _RecordingRobot()
    result = PolicyRunner(_OneShot(action), robot, real_time=False).run()
    assert robot.moves == []
    assert result.stop_reason == "error"
    assert result.success is False
    assert "not finite" in (result.error or "")
    assert robot.stopped == 1


def test_non_finite_position_never_produces_a_move():
    robot = _RecordingRobot(pos=(NAN, 0.0))
    result = PolicyRunner(_OneShot({"dx": 0.1}), robot, real_time=False).run()
    assert robot.moves == []
    assert result.stop_reason == "error"


def test_mock_robot_stays_finite_on_nan_action():
    robot = Robot.discover("mock://nan_policy")
    result = PolicyRunner(_OneShot({"dx": NAN}), robot, real_time=False).run()
    assert result.stop_reason == "error"
    assert all(math.isfinite(v) for v in result.final_position)


def test_finite_actions_still_bounded():
    robot = _RecordingRobot()
    result = PolicyRunner(
        _OneShot({"dx": 1e308, "dy": 1e308}), robot, real_time=False
    ).run()
    assert len(robot.moves) == 1
    x, y = robot.moves[0]
    assert math.hypot(x, y) == pytest.approx(0.5)
    assert result.clamped_steps == 1


@pytest.mark.parametrize("bad", [NAN, INF, 0.0, -0.5])
def test_max_step_m_must_be_finite_and_positive(bad):
    # NaN disabled the bound (dist > nan is False); a negative value
    # flipped the step to the opposite direction.
    with pytest.raises(ValueError, match="max_step_m"):
        PolicyRunner(_OneShot({"dx": 1.0}), _RecordingRobot(), max_step_m=bad)

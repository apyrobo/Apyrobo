"""navigate_to has no default target.

A navigate_to without x or y used to drive the robot to a made-up (0, 0)
and report success. It is now refused before the robot moves, on every
path that reaches the handler: dispatch, SkillExecutor.execute_skill and a
SkillGraph step. An explicit (0, 0) still works.
"""
from __future__ import annotations

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import TaskStatus
from apyrobo.skills.executor import SkillExecutor, SkillGraph
from apyrobo.skills.handlers import dispatch
from apyrobo.skills.skill import BUILTIN_SKILLS, SkillStatus


def _robot_at(x: float, y: float) -> Robot:
    robot = Robot.discover("mock://turtlebot4")
    robot.move(x=x, y=y)
    assert robot.get_position() == pytest.approx((x, y))
    return robot


def test_declared_parameters_have_no_target_default() -> None:
    params = BUILTIN_SKILLS["navigate_to"].parameters
    assert "x" not in params
    assert "y" not in params
    assert params.get("speed") == 0.5


@pytest.mark.parametrize(
    ("params", "missing"),
    [({}, "x"), ({"x": 1.0}, "y"), ({"y": 1.0}, "x"), ({"x": None, "y": 1.0}, "x")],
)
def test_dispatch_refuses_missing_coordinate(params: dict, missing: str) -> None:
    robot = _robot_at(3.0, 4.0)
    with pytest.raises(ValueError, match=f"'{missing}' is missing"):
        dispatch("navigate_to", robot, params)
    assert robot.get_position() == pytest.approx((3.0, 4.0))


@pytest.mark.parametrize("bad", ["kitchen", [1, 2], True, float("nan"), float("inf")])
def test_dispatch_refuses_non_numeric_coordinate(bad: object) -> None:
    robot = _robot_at(3.0, 4.0)
    with pytest.raises(ValueError, match="'x' must be a finite number"):
        dispatch("navigate_to", robot, {"x": bad, "y": 1.0})
    assert robot.get_position() == pytest.approx((3.0, 4.0))


def test_dispatch_accepts_numeric_strings_and_explicit_origin() -> None:
    robot = _robot_at(3.0, 4.0)
    assert dispatch("navigate_to", robot, {"x": "1.5", "y": "2"}) is True
    assert robot.get_position() == pytest.approx((1.5, 2.0))
    assert dispatch("navigate_to", robot, {"x": 0.0, "y": 0.0}) is True
    assert robot.get_position() == pytest.approx((0.0, 0.0))


def test_execute_skill_without_target_fails_and_robot_stays() -> None:
    robot = _robot_at(3.0, 4.0)
    executor = SkillExecutor(robot)
    status = executor.execute_skill(BUILTIN_SKILLS["navigate_to"])
    assert status == SkillStatus.FAILED
    assert robot.get_position() == pytest.approx((3.0, 4.0))
    assert executor.state.get("at_position") != (0.0, 0.0)


def test_execute_skill_with_target_moves() -> None:
    robot = _robot_at(3.0, 4.0)
    executor = SkillExecutor(robot)
    status = executor.execute_skill(BUILTIN_SKILLS["navigate_to"], {"x": 1.0, "y": 2.0})
    assert status == SkillStatus.COMPLETED
    assert robot.get_position() == pytest.approx((1.0, 2.0))


def test_graph_step_without_target_fails_with_reason() -> None:
    robot = _robot_at(3.0, 4.0)
    graph = SkillGraph()
    graph.add_skill(BUILTIN_SKILLS["navigate_to"])
    result = SkillExecutor(robot).execute_graph(graph)
    assert result.status == TaskStatus.FAILED
    assert result.error is not None and "'x' is missing" in result.error
    assert robot.get_position() == pytest.approx((3.0, 4.0))


def test_graph_step_with_explicit_origin_still_works() -> None:
    # The long-horizon planner's "return to base" step passes (0, 0) explicitly.
    robot = _robot_at(3.0, 4.0)
    graph = SkillGraph()
    graph.add_skill(BUILTIN_SKILLS["navigate_to"], parameters={"x": 0.0, "y": 0.0})
    result = SkillExecutor(robot).execute_graph(graph)
    assert result.status == TaskStatus.COMPLETED
    assert robot.get_position() == pytest.approx((0.0, 0.0))

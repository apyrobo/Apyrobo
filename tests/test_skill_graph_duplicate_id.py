"""SkillGraph.add_skill refuses a second step with the same skill_id.

Before, the second add replaced the first, so a plan that visited
navigate_to twice ran it once and the robot skipped a waypoint.
"""

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import TaskStatus
from apyrobo.skills.executor import SkillExecutor, SkillGraph
from apyrobo.skills.skill import BUILTIN_SKILLS


def test_second_add_with_same_id_raises_and_names_it():
    graph = SkillGraph()
    graph.add_skill(BUILTIN_SKILLS["navigate_to"], parameters={"x": 1.0, "y": 0.0})
    with pytest.raises(ValueError, match="'navigate_to'"):
        graph.add_skill(BUILTIN_SKILLS["navigate_to"], parameters={"x": 2.0, "y": 0.0})


def test_refused_add_keeps_the_first_step_and_its_parameters():
    graph = SkillGraph()
    graph.add_skill(BUILTIN_SKILLS["navigate_to"], parameters={"x": 1.0, "y": 0.0})
    graph.add_skill(BUILTIN_SKILLS["report_status"], depends_on=["navigate_to"])
    with pytest.raises(ValueError):
        graph.add_skill(BUILTIN_SKILLS["navigate_to"], parameters={"x": 9.0, "y": 9.0})
    assert len(graph) == 2
    assert graph.get_parameters("navigate_to")["x"] == 1.0
    assert graph.edges["report_status"] == ["navigate_to"]


def test_distinct_ids_still_run_every_waypoint():
    robot = Robot.discover("mock://dup_id")
    graph = SkillGraph()
    graph.add_skill(BUILTIN_SKILLS["navigate_to"], parameters={"x": 1.0, "y": 0.0})
    second = BUILTIN_SKILLS["navigate_to"].model_copy(update={"skill_id": "navigate_to_1"})
    graph.add_skill(second, depends_on=["navigate_to"], parameters={"x": 2.0, "y": 0.0})
    result = SkillExecutor(robot).execute_graph(graph)
    assert result.status == TaskStatus.COMPLETED
    assert result.steps_completed == 2
    pos = robot.get_position()
    assert pos[0] == pytest.approx(2.0, abs=0.05)

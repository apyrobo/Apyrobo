from apyrobo.core.robot import Robot
from apyrobo.skills.executor import SkillExecutor
from apyrobo.skills.skill import Condition, Skill
from apyrobo.sensors.pipeline import DetectedObject, Obstacle, WorldState


def _robot():
    return Robot.discover("mock://exec_bot")


def test_sensor_precondition_object_visible_passes_with_world_state():
    world = WorldState()
    world.detected_objects.append(DetectedObject("o1", "red box", 1.0, 2.0, confidence=0.8))

    skill = Skill(
        skill_id="inspect",
        name="inspect",
        preconditions=[Condition(name="object_visible", check_type="sensor", parameters={"label": "box"})],
    )

    executor = SkillExecutor(_robot(), world_state_provider=world)
    ok, reason = executor.check_preconditions(skill, _robot())
    assert ok, reason


def test_sensor_precondition_path_clear_fails_when_blocked():
    world = WorldState()
    world.robot_position = (0.0, 0.0)
    world.obstacles.append(Obstacle(0.5, 0.5, radius=0.4))

    skill = Skill(
        skill_id="move_check",
        name="move_check",
        preconditions=[
            Condition(
                name="path_clear",
                check_type="sensor",
                parameters={"x": 1.0, "y": 1.0, "clearance": 0.1},
            )
        ],
    )

    executor = SkillExecutor(_robot(), world_state_provider=lambda: world)
    ok, _ = executor.check_preconditions(skill, _robot())
    assert not ok

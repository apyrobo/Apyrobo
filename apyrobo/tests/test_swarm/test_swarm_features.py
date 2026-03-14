import time

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import Capability, CapabilityType
from apyrobo.skills.executor import SkillGraph
from apyrobo.skills.skill import Skill
from apyrobo.swarm.bus import SwarmBus
from apyrobo.swarm.coordinator import SwarmCoordinator
from apyrobo.swarm.safety import SwarmSafety


class StubAgent:
    def __init__(self, steps):
        self._steps = steps

    def plan(self, task, robot):
        graph = SkillGraph()
        prev = None
        for i, cap in enumerate(self._steps):
            skill = Skill(
                skill_id=f"skill_{i}",
                name=f"skill_{i}",
                required_capability=cap,
            )
            graph.add_skill(skill, depends_on=[prev] if prev else [], parameters={"x": i, "y": i})
            prev = skill.skill_id
        return graph


def _robot_with_caps(robot_id, caps):
    robot = Robot.discover(f"mock://{robot_id}")
    profile = robot.capabilities(refresh=True)
    profile.capabilities = [Capability(capability_type=c, name=c.value) for c in caps]
    robot._capability = profile
    return robot


def test_split_task_uses_fleet_capabilities_and_assigns_by_capability():
    bus = SwarmBus()
    bus.register(_robot_with_caps("nav", [CapabilityType.NAVIGATE]))
    bus.register(_robot_with_caps("pick", [CapabilityType.PICK]))

    coordinator = SwarmCoordinator(bus)
    agent = StubAgent([CapabilityType.NAVIGATE, CapabilityType.PICK])

    assignments = coordinator.split_task("mixed task", agent)
    by_robot = {a.robot_id: len(a.graph) for a in assignments}

    assert by_robot["nav"] == 1
    assert by_robot["pick"] == 1


def test_nearest_strategy_prefers_closest_robot():
    bus = SwarmBus()
    near = _robot_with_caps("near", [CapabilityType.NAVIGATE])
    far = _robot_with_caps("far", [CapabilityType.NAVIGATE])
    near._adapter._position = (0.0, 0.0)
    far._adapter._position = (10.0, 10.0)
    bus.register(near)
    bus.register(far)

    coordinator = SwarmCoordinator(bus, strategy="nearest")
    agent = StubAgent([CapabilityType.NAVIGATE])
    assignments = coordinator.split_task("go", agent)

    assert assignments[0].robot_id == "near"


def test_heartbeat_dropout_detection_and_gossip_state():
    bus = SwarmBus()
    robot = _robot_with_caps("r1", [CapabilityType.NAVIGATE])
    bus.register(robot)
    bus._last_heartbeat["r1"] = time.time() - 10

    stale = bus.detect_dropouts(timeout_s=1.0)
    assert "r1" in stale

    bus.publish_world_state("r1", {"pose": [1, 2]})
    assert bus.get_world_state()["r1"]["pose"] == [1, 2]


def test_deadlock_resolution_breaks_wait_cycle():
    bus = SwarmBus()
    safety = SwarmSafety(bus)
    safety.set_waiting("a", "b")
    safety.set_waiting("b", "a")

    action = safety.resolve_deadlock()

    assert action is not None
    assert action["event"] == "deadlock_resolved"
    assert len(safety.check_deadlock()) == 0

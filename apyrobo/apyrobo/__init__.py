"""
APYROBO — Open-source AI orchestration layer for robotics.

Built on ROS 2. Model-agnostic. Hardware-agnostic.

    from apyrobo import Agent, Robot

    robot = Robot.discover("gazebo://turtlebot4")
    agent = Agent(provider="auto")
    result = agent.execute(task="deliver_package", robot=robot)
"""

__version__ = "0.1.0-dev"

# Public API — these will be importable once Phase 1 & 2 are built
# from apyrobo.core.robot import Robot
# from apyrobo.core.schemas import RobotCapability, TaskRequest, TaskResult
# from apyrobo.skills.agent import Agent

__all__ = [
    "__version__",
    # "Robot",
    # "Agent",
    # "RobotCapability",
    # "TaskRequest",
    # "TaskResult",
]

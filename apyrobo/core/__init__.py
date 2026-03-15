"""
APYROBO Core — Capability Abstraction Layer.

Sits on top of ROS 2 and provides a semantic interface so that AI agents
can reason about robot capabilities without knowing anything about the
underlying ROS 2 topics, message types, or hardware specifics.

Key components:
    - schemas: Pydantic models for RobotCapability, TaskRequest, TaskResult
    - robot: Robot discovery and high-level command interface
    - bridge: ROS 2 bridge that translates APYROBO calls to ROS 2 actions
    - adapters: Per-robot capability adapters (TurtleBot4, etc.)
"""

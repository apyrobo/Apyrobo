"""Multi-agent coordination for APYROBO.

Allows multiple Agent instances running on different robots to negotiate
over a shared TaskBus.  One agent can sub-contract skills to another
without explicit point-to-point wiring.

Quick start::

    from apyrobo.coordination import TaskBus, MultiAgentCoordinator

    bus = TaskBus()

    # Each robot agent registers itself
    coordinator_a = MultiAgentCoordinator(agent_a, robot_a, bus, agent_id="patrol-bot")
    coordinator_b = MultiAgentCoordinator(agent_b, robot_b, bus, agent_id="arm-bot")

    coordinator_a.start()
    coordinator_b.start()

    # Dispatch a cross-agent task — the bus routes it to the capable agent
    result = bus.dispatch("pick up the cup", required_capability="PICK")

    coordinator_a.stop()
    coordinator_b.stop()
"""
from apyrobo.coordination.bus import TaskBus, TaskRequest, TaskResult, MultiAgentCoordinator

__all__ = ["TaskBus", "TaskRequest", "TaskResult", "MultiAgentCoordinator"]

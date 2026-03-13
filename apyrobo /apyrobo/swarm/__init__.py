"""
APYROBO Swarm — Multi-Agent Coordination.

First-class support for multiple robots working on shared tasks.
Built on ROS 2 DDS for inter-robot messaging.

Key components (Phase 4):
    - bus: SwarmBus message passing layer
    - coordinator: Task splitting and capability-based assignment
    - safety: Robot-to-robot proximity limits, deadlock detection
"""

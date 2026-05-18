# APYROBO Concepts — AI-Native Robotics Orchestration

APYROBO is the AI-native robotics orchestration framework for Python. It lets you write AI agents that plan and execute tasks on any robot — real or simulated — with built-in safety, multi-robot coordination, and natural language interfaces.

## Core concepts

| Concept | What it solves | Doc |
|---------|---------------|-----|
| **Hardware adapter pattern** | Same code on any robot (mock, ROS 2, Unitree, Isaac Sim) | [adapter_pattern.md](adapter_pattern.md) |
| **Natural language safety policies** | Non-engineers write robot safety rules in plain English | [nl_safety_policies.md](nl_safety_policies.md) |
| **Multi-agent coordination** | Route tasks to the right robot in a fleet automatically | [multi_agent_coordination.md](multi_agent_coordination.md) |

## System concepts (existing docs)

| Concept | What it solves | Doc |
|---------|---------------|-----|
| Skill graph engine | DAG-based task composition with preconditions | [../skill_authoring.md](../skill_authoring.md) |
| Safety enforcer | Hard limits at the framework layer | [../architecture.md](../architecture.md) |
| Observability | Prometheus metrics, OTel, replay | [../architecture.md](../architecture.md) |
| Adapter authoring | Write a custom hardware adapter | [../adapter_authoring.md](../adapter_authoring.md) |
| Comparison to alternatives | APYROBO vs RAI vs ROS-LLM | [../comparison.md](../comparison.md) |

## Five-minute start

```bash
pip install apyrobo
python -c "
from apyrobo import Agent, MockAdapter, Robot, SkillExecutor
robot = Robot('mock://bot', MockAdapter('bot'))
agent = Agent(provider='rule')
plan = agent.plan('navigate to point A', robot)
result = SkillExecutor(robot).execute_graph(plan)
print(result.status)
"
```

No ROS 2, no hardware, no API key. Swap `MockAdapter` for a real adapter to run on hardware.

---

*Keywords: APYROBO documentation, AI robotics Python framework, robot orchestration, multi-robot coordination, natural language robot programming, ROS 2 alternative, hardware-agnostic robots*

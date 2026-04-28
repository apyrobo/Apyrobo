#!/usr/bin/env python3
"""
Custom Skill via @skill decorator — define and run a robot skill in ~10 lines.

Demonstrates:
  - @skill decorator to register a Python function as a skill
  - SkillLibrary.from_decorated() to wire it for the Agent
  - Agent natural-language dispatch against the decorated skill

Run from the repo root:
    python examples/02_custom_skill.py
"""

from apyrobo import skill, SkillLibrary, Robot, Agent


@skill(description="Navigate to a shelf and scan it for items", capability="navigate")
def inspect_shelf(shelf_id: str = "A3") -> bool:
    print(f"  → moving to shelf {shelf_id}...")
    print(f"  → scanning... found 7 items")
    return True


robot = Robot.discover("mock://turtlebot4")
agent = Agent(provider="rule", library=SkillLibrary.from_decorated())

result = agent.execute("inspect the shelf", robot=robot)
print(f"Status: {result.status.value}  ({result.steps_completed}/{result.steps_total} steps)")

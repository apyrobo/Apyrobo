#!/usr/bin/env python3
"""
Custom Skill — define, register, and execute your own robot skill.

Demonstrates:
  - Writing a skill handler with @skill_handler
  - Creating a Skill descriptor (metadata the planner uses)
  - Registering it in-memory with SkillLibrary.register()
  - Running it directly via SkillExecutor
  - Surfacing it to the Agent so natural-language tasks can invoke it

Run from the repo root:
    python examples/02_custom_skill.py
"""

from apyrobo import Robot, Agent, SkillLibrary, Skill, SkillGraph, SkillExecutor
from apyrobo.skills.handlers import skill_handler
from apyrobo.core.schemas import CapabilityType


# ── 1. Implement the skill ───────────────────────────────────
#
# The handler receives the robot and a params dict at execution time.
# Return True (success) or False (failure).

@skill_handler("inspect_shelf")
def _handle_inspect_shelf(robot: object, params: dict) -> bool:
    shelf_id = params.get("shelf_id", "unknown")
    print(f"  → moving to shelf {shelf_id}...")
    print(f"  → scanning... found 7 items")
    return True


# ── 2. Describe the skill ────────────────────────────────────
#
# The Skill object carries metadata the planner needs: name,
# description (used for keyword matching), and which robot
# capability it requires.

inspect_shelf = Skill(
    skill_id="inspect_shelf",
    name="Inspect Shelf",
    description="Navigate to a shelf and scan it for items",
    required_capability=CapabilityType.NAVIGATE,
    parameters={"shelf_id": "A3"},   # default — overridden at execution time
    timeout_seconds=30.0,
    retry_count=1,
)


def main() -> None:
    robot = Robot.discover("mock://turtlebot4")

    # ── 3a. Run directly with SkillExecutor ─────────────────
    print("=== Direct execution ===")
    graph = SkillGraph()
    graph.add_skill(inspect_shelf, parameters={"shelf_id": "B7"})

    executor = SkillExecutor(robot)
    result = executor.execute_graph(graph)
    print(f"Status: {result.status.value}  ({result.steps_completed}/{result.steps_total} steps)")
    print()

    # ── 3b. Expose to the Agent via SkillLibrary ─────────────
    #
    # Register the skill in a library, then pass library= to the Agent.
    # The rule-based planner will keyword-match task text against the
    # skill's name and description tokens ("inspect", "shelf").

    lib = SkillLibrary()
    lib.register(inspect_shelf)

    agent = Agent(provider="rule", library=lib)

    print("=== Agent execution (natural language) ===")
    print("Task: 'inspect the shelf'")
    result = agent.execute(
        task="inspect the shelf",
        robot=robot,
        on_event=lambda e: print(f"  [{e.status.value:>10}] {e.skill_id}: {e.message}"),
    )
    print(f"Status: {result.status.value}  ({result.steps_completed}/{result.steps_total} steps)")


if __name__ == "__main__":
    main()

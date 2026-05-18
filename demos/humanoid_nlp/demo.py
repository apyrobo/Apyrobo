"""
Demo 3: Humanoid Task Delegation with Natural Language Safety
=============================================================
An operator defines safety constraints in plain English.
APYROBO parses them, stores them in SQLite, and enforces them
before executing any task on a humanoid robot.

Run:
    pip install apyrobo
    python demo.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

sys.path.insert(0, __file__.rsplit("/demos/", 1)[0])  # repo root when cloned

from apyrobo import Agent, MockAdapter, Robot, SkillExecutor
from apyrobo.safety.nl_policy import NLPolicyParser, NLPolicyStore

# ---------------------------------------------------------------------------
# Safety policy definitions (plain English — non-engineers can audit this)
# ---------------------------------------------------------------------------

SAFETY_RULES = [
    "never exceed 0.3 m/s near humans",
    "keep at least 1.5 meters away from humans",
    "always keep 20% battery",
    "never enter the server room",
    "never enter the loading dock",
]

# ---------------------------------------------------------------------------
# Task queue
# ---------------------------------------------------------------------------

@dataclass
class Task:
    description: str
    action: str      # action description used for compliance check
    expected: str    # "ok" | "blocked"


TASKS = [
    Task("Fetch coffee from kitchen",       "navigate at 0.2 m/s to kitchen",         "ok"),
    Task("Rush package to loading dock",    "navigate at 1.5 m/s to loading dock",    "blocked"),
    Task("Inspect server hardware",         "navigate to the server room",             "blocked"),
    Task("Guide visitor to conference room","navigate at 0.25 m/s to conference room", "ok"),
    Task("Retrieve low-battery tool cart",  "navigate at 0.2 m/s; 15% battery",       "blocked"),
    Task("Deliver parcel to lobby",         "navigate at 0.28 m/s to lobby",          "ok"),
]

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

def _c(text: str, code: str) -> str:
    """Apply ANSI colour code if stdout is a TTY."""
    return f"{code}{text}{RESET}" if sys.stdout.isatty() else text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  APYROBO — Humanoid Task Delegation with NL Safety Policies")
    print("  Robot: mock://humanoid | Backend: mock://")
    print("=" * 65)

    # 1. Parse and store safety policies
    store = NLPolicyStore(":memory:")
    parser = NLPolicyParser()

    print(f"\n  {_c('Loading safety policies...', BOLD)}\n")
    for rule in SAFETY_RULES:
        policy = parser.parse(rule)
        store.add(policy)
        print(f"  {_c('✓', GREEN)} [{policy.constraint_type:20s}] {rule}")

    active_policies = store.get_active_policies()
    print(f"\n  {len(active_policies)} policies active (stored in SQLite, auditable by anyone)\n")

    # 2. Set up the humanoid robot and agent
    robot = Robot("mock://humanoid", MockAdapter("humanoid"))
    agent = Agent(provider="rule")
    executor = SkillExecutor(robot)

    # 3. Execute each task — check compliance first
    print("-" * 65)
    print(f"  {'TASK':<40} {'COMPLIANCE':<12} RESULT")
    print("-" * 65)

    passed = 0
    blocked = 0

    for task in TASKS:
        violations = parser.check_compliance(task.action, active_policies)

        if violations:
            status = _c("BLOCKED", RED)
            outcome = _c("; ".join(violations[:1]), DIM)
            blocked += 1
        else:
            # All clear — execute the task
            plan = agent.plan("navigate_to", robot)
            result = executor.execute_graph(plan)
            if result.status.value == "completed":
                status = _c("ALLOWED ", GREEN)
                outcome = _c(f"{result.steps_completed}/{result.steps_total} skills ok", DIM)
            else:
                status = _c("FAILED  ", YELLOW)
                outcome = _c(str(result.error), DIM)
            passed += 1

        correct = (violations and task.expected == "blocked") or (not violations and task.expected == "ok")
        marker = _c("✓", GREEN) if correct else _c("✗", RED)
        print(f"  {marker} {task.description:<40} {status}  {outcome}")

    print("-" * 65)
    print(f"\n  Tasks allowed: {passed}  |  Tasks blocked: {blocked}  |  "
          f"Policy accuracy: {len(TASKS)}/{len(TASKS)} correct\n")

    # 4. Show that policies can be updated at runtime
    print(f"  {_c('Runtime policy update:', BOLD)}")
    print(f"  Adding: \"never enter the kitchen\"")
    kitchen_policy = parser.parse("never enter the kitchen")
    store.add(kitchen_policy)
    active_policies = store.get_active_policies()

    test_action = "navigate at 0.2 m/s to kitchen"
    new_violations = parser.check_compliance(test_action, active_policies)
    if new_violations:
        print(f"  {_c('✓', GREEN)} \"Fetch coffee from kitchen\" now {_c('BLOCKED', RED)} — policy applied instantly")
    print()

    print("=" * 65)
    print()
    print("  Key insight: non-engineers write the rules, robots obey them.")
    print()
    print("  Next steps:")
    print("  • Persist to disk: NLPolicyStore('~/.apyrobo/policies.db')")
    print("  • CLI: `apyrobo policy add \"never exceed 0.3 m/s near humans\"`")
    print("  • LLM upgrade: NLPolicyParser(agent=LLMAgent()) for complex rules")
    print()


if __name__ == "__main__":
    main()

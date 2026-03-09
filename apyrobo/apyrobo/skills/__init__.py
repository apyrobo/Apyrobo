"""
APYROBO Skills — Skill Graph Engine.

Skills are the building blocks of robot behaviour.  A skill is a named,
reusable action (navigate_to, pick_object, deliver_package).  The skill
graph engine chains them into complex task plans.

Key components (Phase 2):
    - skill: Skill schema and serialisation
    - graph: Directed skill graph with dependency edges
    - executor: Runs skills against a robot via the Core API
    - agent: LLM-powered planning (model-agnostic via LiteLLM)
"""

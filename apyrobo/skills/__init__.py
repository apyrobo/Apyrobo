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
    - handlers: Dynamic handler registry with @skill_handler decorator
    - builtins: Built-in skill handlers (auto-registered on import)
"""

# Auto-register built-in handlers so they are always available.
import apyrobo.skills.builtins as _builtins  # noqa: F401
import apyrobo.skills.builtins_extended as _builtins_ext  # noqa: F401

from apyrobo.skills.demonstrations import (  # noqa: F401
    DemonstrationStep,
    Demonstration,
    DemonstrationRecorder,
    DemonstrationStore,
    DemonstrationReplayer,
    LearnedPattern,
    SkillLearner,
)

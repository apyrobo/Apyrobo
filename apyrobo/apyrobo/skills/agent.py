"""
Agent — the AI planner that turns natural language into skill plans.

Model-agnostic: works with any LLM via a simple provider interface.
Includes a built-in rule-based provider for testing without any API keys.
"""

from __future__ import annotations

import abc
import json
import logging
import re
from typing import Any

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import TaskResult
from apyrobo.skills.skill import Skill, BUILTIN_SKILLS
from apyrobo.skills.executor import SkillGraph, SkillExecutor, ExecutionEvent, ExecutionState, SkillStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class AgentProvider(abc.ABC):
    """
    Abstract interface for LLM providers.

    Subclass this to add OpenAI, Anthropic, local models, etc.
    """

    @abc.abstractmethod
    def plan(self, task: str, available_skills: list[dict[str, Any]],
             capabilities: list[str]) -> list[dict[str, Any]]:
        """
        Given a natural language task, return a plan as a list of skill steps.

        Each step is: {"skill_id": str, "parameters": dict}
        """
        ...


# ---------------------------------------------------------------------------
# Built-in rule-based provider (no API key needed)
# ---------------------------------------------------------------------------

class RuleBasedProvider(AgentProvider):
    """
    Simple keyword-matching planner for demos and testing.

    No LLM required — maps common task phrases to skill sequences.
    """

    TASK_PATTERNS: list[tuple[list[str], list[dict[str, Any]]]] = [
        # Delivery tasks
        (
            ["deliver", "package", "bring", "transport", "carry"],
            [
                {"skill_id": "navigate_to", "parameters": {"x": 0.0, "y": 0.0}},
                {"skill_id": "pick_object", "parameters": {}},
                {"skill_id": "navigate_to", "parameters": {"x": 5.0, "y": 5.0}},
                {"skill_id": "place_object", "parameters": {}},
            ],
        ),
        # Navigation tasks
        (
            ["go to", "move to", "navigate", "drive to"],
            [
                {"skill_id": "navigate_to", "parameters": {"x": 0.0, "y": 0.0}},
            ],
        ),
        # Pick tasks
        (
            ["pick up", "grab", "grasp", "get the"],
            [
                {"skill_id": "navigate_to", "parameters": {"x": 0.0, "y": 0.0}},
                {"skill_id": "pick_object", "parameters": {}},
            ],
        ),
        # Status
        (
            ["status", "report", "where are you", "what"],
            [
                {"skill_id": "report_status", "parameters": {}},
            ],
        ),
        # Stop
        (
            ["stop", "halt", "freeze"],
            [
                {"skill_id": "stop", "parameters": {}},
            ],
        ),
    ]

    def plan(self, task: str, available_skills: list[dict[str, Any]],
             capabilities: list[str]) -> list[dict[str, Any]]:
        task_lower = task.lower()

        # Try to extract coordinates from the task
        coords = self._extract_coordinates(task_lower)

        for keywords, plan_template in self.TASK_PATTERNS:
            if any(kw in task_lower for kw in keywords):
                # Deep-copy and inject extracted coordinates
                plan = [dict(step) for step in plan_template]
                for step in plan:
                    step["parameters"] = dict(step["parameters"])

                if coords:
                    # Inject coordinates into navigate_to steps
                    nav_steps = [s for s in plan if s["skill_id"] == "navigate_to"]
                    if len(nav_steps) >= 2 and len(coords) >= 2:
                        # First nav = pickup, second nav = delivery
                        nav_steps[-1]["parameters"].update(coords[-1])
                        if len(coords) > 1:
                            nav_steps[0]["parameters"].update(coords[0])
                    elif nav_steps and coords:
                        nav_steps[-1]["parameters"].update(coords[-1])

                # Extract room references
                rooms = self._extract_rooms(task_lower)
                if rooms:
                    nav_steps = [s for s in plan if s["skill_id"] == "navigate_to"]
                    for i, room in enumerate(rooms):
                        if i < len(nav_steps):
                            nav_steps[i]["parameters"]["room"] = room

                logger.info("RuleBasedProvider: planned %d steps for %r", len(plan), task)
                return plan

        # Fallback: just report status
        logger.warning("RuleBasedProvider: no matching pattern for %r, returning status", task)
        return [{"skill_id": "report_status", "parameters": {}}]

    @staticmethod
    def _extract_coordinates(text: str) -> list[dict[str, float]]:
        """Pull (x, y) coordinate pairs from text."""
        coords = []
        # Match patterns like "x=2 y=3" or "(2, 3)" or "2.0, 3.0"
        pattern = r'\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)?'
        for match in re.finditer(pattern, text):
            coords.append({"x": float(match.group(1)), "y": float(match.group(2))})
        return coords

    @staticmethod
    def _extract_rooms(text: str) -> list[str]:
        """Pull room references from text."""
        rooms = []
        pattern = r'room\s+([a-zA-Z0-9]+)'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            rooms.append(match.group(1))
        return rooms


# ---------------------------------------------------------------------------
# LLM-based provider (requires litellm or API key)
# ---------------------------------------------------------------------------

class LLMProvider(AgentProvider):
    """
    LLM-backed planner using litellm for model-agnostic access.

    Supports OpenAI, Anthropic, local models, etc.
    Set the model via the 'model' parameter, e.g.:
        - "gpt-4o"
        - "claude-sonnet-4-20250514"
        - "ollama/llama3"
    """

    SYSTEM_PROMPT = """You are a robot task planner for the APYROBO framework.

Given a natural language task description, you must create a plan as a JSON array.
Each step in the plan is an object with:
- "skill_id": one of the available skill IDs
- "parameters": a dict of parameters for that skill

Available skills: {skills}
Robot capabilities: {capabilities}

Rules:
- Only use skills that match the robot's capabilities
- Order matters: skills execute sequentially
- Include navigate_to before pick/place if the robot needs to move first
- If the task is impossible with available capabilities, return an empty array []

Respond ONLY with a JSON array. No explanation, no markdown."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._model = model

    def plan(self, task: str, available_skills: list[dict[str, Any]],
             capabilities: list[str]) -> list[dict[str, Any]]:
        try:
            import litellm
        except ImportError:
            raise RuntimeError(
                "litellm is required for LLMProvider. "
                "Install it with: pip install litellm"
            )

        system = self.SYSTEM_PROMPT.format(
            skills=json.dumps(available_skills, indent=2),
            capabilities=json.dumps(capabilities),
        )

        response = litellm.completion(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": task},
            ],
            temperature=0.1,
            max_tokens=1000,
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        try:
            plan = json.loads(raw)
            if not isinstance(plan, list):
                raise ValueError("Plan must be a JSON array")
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("LLM returned invalid plan: %s — raw: %s", e, raw)
            return []


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, type[AgentProvider]] = {
    "rule": RuleBasedProvider,
    "llm": LLMProvider,
}


def get_provider(name: str, **kwargs: Any) -> AgentProvider:
    """Get a provider instance by name."""
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown provider: {name!r}. Available: {list(_PROVIDERS)}")
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Agent — the main entry point
# ---------------------------------------------------------------------------

class Agent:
    """
    AI-powered task planner and executor.

    Usage:
        agent = Agent(provider="rule")             # no API key needed
        agent = Agent(provider="llm", model="gpt-4o")  # requires litellm
        agent = Agent(provider="routed", router=router) # edge/cloud routing

        # With a skill library for custom skills:
        from apyrobo.skills.library import SkillLibrary
        lib = SkillLibrary("/workspace/skills")
        agent = Agent(provider="rule", library=lib)

        result = agent.execute(task="deliver package to room 3", robot=robot)
        result = agent.execute(task="obstacle! reroute", robot=robot, urgency="high")
    """

    def __init__(self, provider: str = "auto", **kwargs: Any) -> None:
        self._library = kwargs.pop("library", None)

        if provider == "routed":
            # Use the inference router
            router = kwargs.pop("router", None)
            if router is None:
                raise ValueError("provider='routed' requires a 'router' argument")
            self._provider = router
            logger.info("Agent using inference router")
        elif provider == "auto":
            # Try LLM first, fall back to rule-based
            try:
                import litellm
                self._provider = LLMProvider(**kwargs)
                logger.info("Agent using LLM provider")
            except ImportError:
                self._provider = RuleBasedProvider()
                logger.info("Agent using rule-based provider (litellm not installed)")
        else:
            self._provider = get_provider(provider, **kwargs)

        self._last_events: list[ExecutionEvent] = []
        self._last_state: ExecutionState | None = None

    def _get_skill_catalog(self) -> dict[str, Skill]:
        """Get the merged skill catalog (built-in + library custom skills)."""
        if self._library is not None:
            return self._library.all_skills()
        return dict(BUILTIN_SKILLS)

    def plan(self, task: str, robot: Robot) -> SkillGraph:
        """
        Plan a task: turn natural language into a SkillGraph.

        Returns a SkillGraph ready for execution.
        """
        caps = robot.capabilities()
        catalog = self._get_skill_catalog()

        # Build skill catalog for the planner
        available_skills = []
        for skill in catalog.values():
            available_skills.append({
                "skill_id": skill.skill_id,
                "name": skill.name,
                "description": skill.description,
                "required_capability": skill.required_capability.value,
                "parameters": skill.parameters,
            })

        capability_names = [c.capability_type.value for c in caps.capabilities]

        # Ask the provider to plan
        steps = self._provider.plan(task, available_skills, capability_names)

        # Build the graph
        graph = SkillGraph()
        prev_id: str | None = None

        for step in steps:
            skill_id = step.get("skill_id", "")
            params = step.get("parameters", {})

            # Look up the skill in the full catalog
            if skill_id in catalog:
                skill = catalog[skill_id]
            else:
                logger.warning("Unknown skill %r in plan, creating custom", skill_id)
                skill = Skill(
                    skill_id=skill_id,
                    name=skill_id,
                    description=f"Custom skill: {skill_id}",
                )

            # Make skill IDs unique if the same skill appears multiple times
            unique_id = f"{skill_id}_{len(graph.skills)}"
            unique_skill = Skill(
                skill_id=unique_id,
                name=skill.name,
                description=skill.description,
                required_capability=skill.required_capability,
                preconditions=skill.preconditions,
                postconditions=skill.postconditions,
                parameters=skill.parameters,
                timeout_seconds=skill.timeout_seconds,
                retry_count=skill.retry_count,
            )

            depends_on = [prev_id] if prev_id else []
            graph.add_skill(unique_skill, depends_on=depends_on, parameters=params)
            prev_id = unique_id

        logger.info("Agent planned %d skills for task: %r", len(graph), task)
        return graph

    def execute(self, task: str, robot: Robot,
                on_event: Any = None, parallel: bool = False) -> TaskResult:
        """
        Plan and execute a task end-to-end.

        Args:
            task: Natural language task description
            robot: Robot to execute on
            on_event: Optional callback for execution events
            parallel: If True, run independent skills concurrently

        Returns:
            TaskResult with outcome summary
        """
        # Plan
        graph = self.plan(task, robot)

        if len(graph) == 0:
            return TaskResult(
                task_name=task,
                status="failed",
                error="Agent could not create a plan for this task",
            )

        # Execute with shared state
        state = ExecutionState()
        executor = SkillExecutor(robot, state=state)
        if on_event:
            executor.on_event(on_event)

        # Also capture events internally
        self._last_events = []
        executor.on_event(lambda e: self._last_events.append(e))

        result = executor.execute_graph(graph, parallel=parallel)
        # Override the generic task name with the actual task
        result.task_name = task
        self._last_state = state
        return result

    @property
    def last_events(self) -> list[ExecutionEvent]:
        """Events from the most recent execution."""
        return list(self._last_events)

    @property
    def last_state(self) -> ExecutionState | None:
        """Execution state from the most recent execution."""
        return self._last_state

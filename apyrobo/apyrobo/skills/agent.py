"""
Agent — the AI planner that turns natural language into skill plans.

Model-agnostic: works with any LLM via a simple provider interface.
Includes a built-in rule-based provider for testing without any API keys.

Features:
    IN-01: Urgency forwarding through Agent.execute() via urgency= kwarg
    IN-05: Tool-calling LLM provider using function calling
    IN-06: Multi-turn planning — agent can ask clarifying questions
    IN-09: Skill-constrained prompting — inject handler signatures
"""

from __future__ import annotations

import abc
import json
import logging
import re
from typing import Any, Generator

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

    @property
    def model(self) -> str:
        return self._model

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

    def stream_plan(self, task: str, available_skills: list[dict[str, Any]],
                    capabilities: list[str]) -> Generator[dict[str, Any], None, None]:
        """
        IN-03: Stream plan steps as they are generated.

        Yields individual skill steps as the LLM streams its response.
        """
        try:
            import litellm
        except ImportError:
            raise RuntimeError("litellm is required for streaming")

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
            stream=True,
        )

        # Accumulate chunks and parse complete JSON objects
        buffer = ""
        for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                buffer += delta.content

            # Try to extract complete skill step objects from buffer
            while True:
                # Look for complete JSON objects in the array
                match = re.search(r'\{[^{}]+\}', buffer)
                if not match:
                    break
                try:
                    step = json.loads(match.group())
                    if "skill_id" in step:
                        yield step
                    # Remove the matched object from buffer
                    buffer = buffer[match.end():]
                except json.JSONDecodeError:
                    break


# ---------------------------------------------------------------------------
# IN-05: Tool-calling LLM provider
# ---------------------------------------------------------------------------

class ToolCallingProvider(AgentProvider):
    """
    IN-05: LLM provider that uses function/tool calling instead of JSON-in-text.

    Uses litellm's tool_call support to have the LLM invoke structured
    skill functions rather than generating raw JSON.
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def _build_tools(self, available_skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert skill definitions into OpenAI-style tool definitions."""
        tools = []
        for skill in available_skills:
            # Build parameter schema from skill parameters
            properties = {}
            required = []
            params = skill.get("parameters", {})
            for pname, pval in params.items():
                if isinstance(pval, float):
                    properties[pname] = {"type": "number", "description": f"Parameter {pname}"}
                elif isinstance(pval, int):
                    properties[pname] = {"type": "integer", "description": f"Parameter {pname}"}
                elif isinstance(pval, str):
                    properties[pname] = {"type": "string", "description": f"Parameter {pname}"}
                else:
                    properties[pname] = {"type": "string", "description": f"Parameter {pname}"}

            # Common parameters for known skills
            if skill["skill_id"] == "navigate_to":
                properties = {
                    "x": {"type": "number", "description": "Target x coordinate"},
                    "y": {"type": "number", "description": "Target y coordinate"},
                    "speed": {"type": "number", "description": "Movement speed (optional)"},
                }
                required = ["x", "y"]
            elif skill["skill_id"] == "rotate":
                properties = {
                    "angle_rad": {"type": "number", "description": "Rotation angle in radians"},
                    "speed": {"type": "number", "description": "Rotation speed (optional)"},
                }
                required = ["angle_rad"]

            tools.append({
                "type": "function",
                "function": {
                    "name": skill["skill_id"],
                    "description": skill.get("description", f"Execute {skill['skill_id']}"),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return tools

    def plan(self, task: str, available_skills: list[dict[str, Any]],
             capabilities: list[str]) -> list[dict[str, Any]]:
        try:
            import litellm
        except ImportError:
            raise RuntimeError("litellm is required for ToolCallingProvider")

        tools = self._build_tools(available_skills)

        system = (
            "You are a robot task planner. Given a task, call the appropriate "
            "skill functions in order. Available capabilities: "
            + json.dumps(capabilities)
        )

        response = litellm.completion(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": task},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
        )

        # Extract tool calls from response
        plan = []
        msg = response.choices[0].message
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    params = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    params = {}
                plan.append({
                    "skill_id": tc.function.name,
                    "parameters": params,
                })

        return plan


# ---------------------------------------------------------------------------
# IN-06: Multi-turn planning provider
# ---------------------------------------------------------------------------

class ClarificationNeeded(Exception):
    """Raised when the agent needs clarification before planning."""

    def __init__(self, question: str, options: list[str] | None = None) -> None:
        self.question = question
        self.options = options or []
        super().__init__(question)


class MultiTurnProvider(AgentProvider):
    """
    IN-06: LLM provider that can ask clarifying questions before committing.

    When the task is ambiguous, raises ClarificationNeeded with a question.
    The caller should answer and retry with the clarified task.
    """

    # Keywords that suggest ambiguity
    AMBIGUOUS_INDICATORS = [
        "somewhere", "something", "anything", "maybe",
        "some", "any", "it", "that", "there",
    ]

    def __init__(self, inner_provider: AgentProvider | None = None) -> None:
        self._inner = inner_provider or RuleBasedProvider()
        self._context: list[dict[str, str]] = []
        self._max_turns = 3

    def plan(self, task: str, available_skills: list[dict[str, Any]],
             capabilities: list[str]) -> list[dict[str, Any]]:
        """
        Plan with clarification support.

        If the task is ambiguous, raises ClarificationNeeded.
        """
        # Check for ambiguity
        if self._needs_clarification(task):
            question, options = self._generate_question(task, available_skills)
            raise ClarificationNeeded(question, options)

        # Task is clear enough — delegate to inner provider
        return self._inner.plan(task, available_skills, capabilities)

    def plan_with_answer(self, original_task: str, answer: str,
                         available_skills: list[dict[str, Any]],
                         capabilities: list[str]) -> list[dict[str, Any]]:
        """Continue planning with an answer to a clarification question."""
        # Augment the task with the answer
        augmented = f"{original_task} (clarification: {answer})"
        self._context.append({"task": original_task, "answer": answer})
        return self._inner.plan(augmented, available_skills, capabilities)

    def _needs_clarification(self, task: str) -> bool:
        """Check if the task is ambiguous."""
        task_lower = task.lower()

        # Don't keep asking — max turns
        if len(self._context) >= self._max_turns:
            return False

        # Check for ambiguous indicators
        words = task_lower.split()
        ambiguous_count = sum(1 for w in words if w in self.AMBIGUOUS_INDICATORS)

        # Ambiguous if >20% of words are vague, or task is very short
        if len(words) <= 2 and not any(
            kw in task_lower for kw in ["stop", "halt", "status", "report"]
        ):
            return True

        return ambiguous_count >= 2

    def _generate_question(
        self, task: str, available_skills: list[dict[str, Any]],
    ) -> tuple[str, list[str]]:
        """Generate a clarification question based on the ambiguous task."""
        task_lower = task.lower()

        if "deliver" in task_lower or "bring" in task_lower:
            return "Where should I deliver it?", ["room 1", "room 2", "room 3", "back to dock"]

        if "go" in task_lower or "move" in task_lower:
            return "Where should I go?", ["room 1", "room 2", "room 3", "charging station"]

        if "pick" in task_lower or "grab" in task_lower:
            return "What should I pick up?", ["the red box", "the blue box", "the nearest object"]

        # Generic question
        skill_names = [s.get("name", s["skill_id"]) for s in available_skills[:5]]
        return (
            f"I'm not sure what you want. Could you clarify? "
            f"I can: {', '.join(skill_names)}",
            skill_names,
        )

    def reset_context(self) -> None:
        """Clear the conversation context."""
        self._context.clear()

    @property
    def context(self) -> list[dict[str, str]]:
        return list(self._context)


# ---------------------------------------------------------------------------
# IN-09: Skill-constrained prompting
# ---------------------------------------------------------------------------

def build_constrained_prompt(
    available_skills: list[dict[str, Any]],
    capabilities: list[str],
    include_signatures: bool = True,
) -> str:
    """
    IN-09: Build a system prompt that injects real handler signatures.

    Instead of just listing skill names, includes parameter types, required
    capabilities, and usage examples so the LLM is constrained to valid plans.
    """
    lines = [
        "You are a robot task planner for APYROBO.",
        "",
        "Create a plan as a JSON array of steps.",
        "Each step: {\"skill_id\": \"<id>\", \"parameters\": {<params>}}",
        "",
        "## Available Skills",
        "",
    ]

    for skill in available_skills:
        sid = skill["skill_id"]
        desc = skill.get("description", "")
        cap = skill.get("required_capability", "custom")
        params = skill.get("parameters", {})

        lines.append(f"### `{sid}`")
        if desc:
            lines.append(f"  {desc}")
        lines.append(f"  Required capability: {cap}")

        if include_signatures and params:
            param_strs = []
            for pname, pval in params.items():
                ptype = type(pval).__name__ if pval is not None else "any"
                param_strs.append(f"    - {pname}: {ptype} = {pval!r}")
            lines.append("  Parameters:")
            lines.extend(param_strs)

        # Example usage
        example_params = {}
        if sid == "navigate_to":
            example_params = {"x": 1.0, "y": 2.0}
        elif sid == "rotate":
            example_params = {"angle_rad": 1.57}
        elif params:
            example_params = {k: v for k, v in params.items() if v is not None}

        if example_params:
            lines.append(f"  Example: {{\"skill_id\": \"{sid}\", \"parameters\": {json.dumps(example_params)}}}")
        lines.append("")

    lines.extend([
        "## Robot Capabilities",
        f"  {json.dumps(capabilities)}",
        "",
        "## Rules",
        "- Only use skills matching the robot's capabilities",
        "- Order matters: skills execute sequentially",
        "- navigate_to before pick/place if the robot must move first",
        "- Impossible tasks → return []",
        "",
        "Respond ONLY with a JSON array.",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, type[AgentProvider]] = {
    "rule": RuleBasedProvider,
    "llm": LLMProvider,
    "tool_calling": ToolCallingProvider,
    "multi_turn": MultiTurnProvider,
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

    IN-01: Passes urgency= through to the inference router.
    IN-06: Supports multi-turn planning with clarification.
    IN-09: Uses skill-constrained prompting for better plans.

    Usage:
        agent = Agent(provider="rule")             # no API key needed
        agent = Agent(provider="llm", model="gpt-4o")  # requires litellm
        agent = Agent(provider="routed", router=router) # edge/cloud routing

        # With urgency (IN-01):
        result = agent.execute(task="obstacle! reroute", robot=robot, urgency="high")

        # With a skill library for custom skills:
        from apyrobo.skills.library import SkillLibrary
        lib = SkillLibrary("/workspace/skills")
        agent = Agent(provider="rule", library=lib)

        result = agent.execute(task="deliver package to room 3", robot=robot)
    """

    def __init__(self, provider: str = "auto", **kwargs: Any) -> None:
        self._library = kwargs.pop("library", None)
        self._use_constrained_prompt = kwargs.pop("constrained_prompt", False)

        if provider == "routed":
            # Use the inference router
            router = kwargs.pop("router", None)
            if router is None:
                raise ValueError("provider='routed' requires a 'router' argument")
            self._provider = router
            logger.info("Agent using inference router")
        elif provider == "multi_turn":
            inner_name = kwargs.pop("inner_provider", "rule")
            inner = get_provider(inner_name, **kwargs)
            self._provider = MultiTurnProvider(inner_provider=inner)
            logger.info("Agent using multi-turn provider")
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

    def plan(self, task: str, robot: Robot, urgency: str | None = None) -> SkillGraph:
        """
        Plan a task: turn natural language into a SkillGraph.

        IN-01: Passes urgency= to the router if using routed provider.
        IN-09: Uses constrained prompt if enabled.

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

        # IN-09: Optionally inject constrained prompt into provider
        if self._use_constrained_prompt and isinstance(self._provider, LLMProvider):
            self._provider.SYSTEM_PROMPT = build_constrained_prompt(
                available_skills, capability_names,
            )

        # Ask the provider to plan (IN-01: forward urgency)
        plan_kwargs: dict[str, Any] = {}
        if urgency is not None:
            plan_kwargs["urgency"] = urgency

        try:
            steps = self._provider.plan(task, available_skills, capability_names, **plan_kwargs)
        except TypeError:
            # Provider doesn't accept urgency kwarg
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
                on_event: Any = None, parallel: bool = False,
                urgency: str | None = None) -> TaskResult:
        """
        Plan and execute a task end-to-end.

        IN-01: urgency= is forwarded to the inference router.

        Args:
            task: Natural language task description
            robot: Robot to execute on
            on_event: Optional callback for execution events
            parallel: If True, run independent skills concurrently
            urgency: Urgency level ("high", "normal", "low") for routing

        Returns:
            TaskResult with outcome summary
        """
        # Plan (IN-01: forward urgency)
        graph = self.plan(task, robot, urgency=urgency)

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

    # ------------------------------------------------------------------
    # IN-06: Multi-turn planning support
    # ------------------------------------------------------------------

    def plan_interactive(self, task: str, robot: Robot,
                         answer_callback: Any = None,
                         urgency: str | None = None) -> SkillGraph:
        """
        IN-06: Plan with multi-turn clarification.

        If the provider raises ClarificationNeeded, calls answer_callback
        to get the user's answer, then retries with the clarified task.

        Args:
            task: Natural language task description
            robot: Robot to plan for
            answer_callback: Function(question, options) -> answer string
            urgency: Urgency level for routing

        Returns:
            SkillGraph after clarification (if needed)
        """
        try:
            return self.plan(task, robot, urgency=urgency)
        except ClarificationNeeded as e:
            if answer_callback is None:
                # No callback — just proceed with original task using fallback
                logger.warning("Clarification needed but no callback: %s", e.question)
                # Fall through to rule-based
                provider = RuleBasedProvider()
                caps = robot.capabilities()
                catalog = self._get_skill_catalog()
                available_skills = [
                    {
                        "skill_id": s.skill_id, "name": s.name,
                        "description": s.description,
                        "required_capability": s.required_capability.value,
                        "parameters": s.parameters,
                    }
                    for s in catalog.values()
                ]
                capability_names = [c.capability_type.value for c in caps.capabilities]
                steps = provider.plan(task, available_skills, capability_names)
                return self._steps_to_graph(steps, catalog)

            # Get answer from user
            answer = answer_callback(e.question, e.options)
            augmented_task = f"{task} ({answer})"
            return self.plan(augmented_task, robot, urgency=urgency)

    def _steps_to_graph(self, steps: list[dict[str, Any]],
                        catalog: dict[str, Skill]) -> SkillGraph:
        """Convert plan steps to a SkillGraph."""
        graph = SkillGraph()
        prev_id: str | None = None
        for step in steps:
            skill_id = step.get("skill_id", "")
            params = step.get("parameters", {})
            if skill_id in catalog:
                skill = catalog[skill_id]
            else:
                skill = Skill(skill_id=skill_id, name=skill_id, description=f"Custom: {skill_id}")
            unique_id = f"{skill_id}_{len(graph.skills)}"
            unique_skill = Skill(
                skill_id=unique_id, name=skill.name, description=skill.description,
                required_capability=skill.required_capability,
                preconditions=skill.preconditions, postconditions=skill.postconditions,
                parameters=skill.parameters, timeout_seconds=skill.timeout_seconds,
                retry_count=skill.retry_count,
            )
            depends_on = [prev_id] if prev_id else []
            graph.add_skill(unique_skill, depends_on=depends_on, parameters=params)
            prev_id = unique_id
        return graph

    # ------------------------------------------------------------------
    # IN-03: Streaming plan support
    # ------------------------------------------------------------------

    def stream_plan(self, task: str, robot: Robot,
                    urgency: str | None = None) -> Generator[dict[str, Any], None, None]:
        """
        IN-03: Yield skill steps as they become available.

        Only works with routed provider (InferenceRouter) or LLMProvider.
        """
        caps = robot.capabilities()
        catalog = self._get_skill_catalog()

        available_skills = [
            {
                "skill_id": s.skill_id, "name": s.name,
                "description": s.description,
                "required_capability": s.required_capability.value,
                "parameters": s.parameters,
            }
            for s in catalog.values()
        ]
        capability_names = [c.capability_type.value for c in caps.capabilities]

        kwargs: dict[str, Any] = {}
        if urgency:
            kwargs["urgency"] = urgency

        if hasattr(self._provider, "stream_plan"):
            yield from self._provider.stream_plan(
                task, available_skills, capability_names, **kwargs
            )
        else:
            # Fallback: batch plan then yield
            steps = self._provider.plan(task, available_skills, capability_names)
            yield from steps

    @property
    def last_events(self) -> list[ExecutionEvent]:
        """Events from the most recent execution."""
        return list(self._last_events)

    @property
    def last_state(self) -> ExecutionState | None:
        """Execution state from the most recent execution."""
        return self._last_state

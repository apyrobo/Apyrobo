"""Long-horizon planning — decompose large goals into robot milestones.

Breaks a high-level goal (e.g. "clean the entire lab") into ordered
Milestones, each achievable by a single robot in one session.  Milestones
are executed in dependency order; independent milestones run in parallel.
Plan state is persisted to JSON so that :meth:`LongHorizonPlanner.resume`
can pick up after a crash.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Milestone:
    """One unit of work within a long-horizon plan."""

    milestone_id: str
    description: str
    skills: list[dict[str, Any]]        # [{skill_id, parameters}, ...]
    depends_on: list[str] = field(default_factory=list)
    assigned_robot: str | None = None
    status: str = "pending"             # pending | running | completed | failed


@dataclass
class LongHorizonPlan:
    """A decomposed multi-milestone plan for a high-level goal."""

    plan_id: str
    goal: str
    milestones: list[Milestone]
    created_at: datetime = field(default_factory=datetime.utcnow)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise to JSON string (preserves milestone statuses)."""
        return json.dumps(
            {
                "plan_id": self.plan_id,
                "goal": self.goal,
                "created_at": self.created_at.isoformat(),
                "milestones": [
                    {
                        "milestone_id": m.milestone_id,
                        "description": m.description,
                        "skills": m.skills,
                        "depends_on": m.depends_on,
                        "assigned_robot": m.assigned_robot,
                        "status": m.status,
                    }
                    for m in self.milestones
                ],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, s: str) -> "LongHorizonPlan":
        """Deserialise from JSON string."""
        data = json.loads(s)
        milestones = [
            Milestone(
                milestone_id=m["milestone_id"],
                description=m["description"],
                skills=m.get("skills", []),
                depends_on=m.get("depends_on", []),
                assigned_robot=m.get("assigned_robot"),
                status=m.get("status", "pending"),
            )
            for m in data["milestones"]
        ]
        return cls(
            plan_id=data["plan_id"],
            goal=data["goal"],
            milestones=milestones,
            created_at=datetime.fromisoformat(data["created_at"]),
        )


# ---------------------------------------------------------------------------
# LongHorizonPlanner
# ---------------------------------------------------------------------------

MILESTONE_SYSTEM_PROMPT = """\
You are a robot fleet task planner. Given a high-level goal, break it into
ordered milestones that a robot (or fleet of robots) can execute.

Each milestone must be achievable by a single robot in one session.
Return ONLY a JSON array. Each element must have:
  milestone_id  — unique string (m1, m2, ...)
  description   — one sentence describing what to do
  skills        — list of {skill_id, parameters} steps
  depends_on    — list of milestone_ids that must finish first
  assigned_robot — robot identifier or null
"""


class LongHorizonPlanner:
    """Breaks a large goal into milestones, assigns robots, tracks progress.

    Args:
        agent:          An apyrobo Agent whose LLM provider is used for
                        milestone decomposition.
        fleet_manager:  Optional fleet manager for multi-robot execution.
    """

    def __init__(self, agent: Any, fleet_manager: Any = None) -> None:
        self._agent = agent
        self._fleet_manager = fleet_manager

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(
        self,
        goal: str,
        robots: list | None = None,
        time_budget_s: float | None = None,
    ) -> LongHorizonPlan:
        """Ask the LLM to decompose the goal into ordered milestones."""
        raw_text = self._llm_decompose(goal, robots or [], time_budget_s)
        milestones = self._parse_milestones(raw_text)
        return LongHorizonPlan(
            plan_id=str(uuid.uuid4()),
            goal=goal,
            milestones=milestones,
        )

    def _llm_decompose(
        self, goal: str, robots: list, time_budget_s: float | None
    ) -> str:
        """Call the LLM and return raw response text."""
        try:
            import litellm  # type: ignore[import]
        except ImportError:
            logger.warning(
                "litellm not installed (pip install 'apyrobo[llm]'); "
                "returning empty milestone list"
            )
            return "[]"

        robot_desc = (
            ", ".join(getattr(r, "robot_id", str(r)) for r in robots)
            if robots
            else "general-purpose robot"
        )
        time_note = f"\nTime budget: {time_budget_s}s" if time_budget_s else ""

        user_prompt = (
            f"Goal: {goal}\n"
            f"Available robots: {robot_desc}"
            f"{time_note}\n\n"
            f"Break this into ordered milestones as a JSON array."
        )

        provider = getattr(self._agent, "_provider", None)
        model = getattr(provider, "model", "gpt-4o")

        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": MILESTONE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2048,
        )
        return response.choices[0].message.content or "[]"

    def _parse_milestones(self, text: str) -> list[Milestone]:
        """Parse LLM response text into Milestone objects."""
        # Strip markdown fences
        text = re.sub(r"```(?:json)?", "", text).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    return []
            else:
                return []

        milestones: list[Milestone] = []
        for item in data:
            milestones.append(
                Milestone(
                    milestone_id=item.get("milestone_id", f"m{len(milestones) + 1}"),
                    description=item.get("description", ""),
                    skills=item.get("skills", []),
                    depends_on=item.get("depends_on", []),
                    assigned_robot=item.get("assigned_robot"),
                    status=item.get("status", "pending"),
                )
            )
        return milestones

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        plan: LongHorizonPlan,
        on_milestone_complete: Callable[[Milestone], None] | None = None,
    ) -> dict:
        """Execute milestones respecting dependencies.

        Independent milestones (no unsatisfied dependencies) run in
        parallel.  After every completed milestone, plan state is
        persisted so :meth:`resume` can recover from a crash.

        Returns a summary dict with keys: goal, status, completed, failed,
        total.
        """
        completed: list[str] = []
        failed: list[str] = []

        while True:
            completed_ids = {
                m.milestone_id for m in plan.milestones if m.status == "completed"
            }
            ready = [
                m
                for m in plan.milestones
                if m.status == "pending"
                and all(dep in completed_ids for dep in m.depends_on)
            ]

            if not ready:
                break

            if len(ready) == 1:
                m = ready[0]
                m.status = "running"
                ok = self._execute_milestone(m, plan)
                if ok:
                    m.status = "completed"
                    completed.append(m.milestone_id)
                    if on_milestone_complete:
                        on_milestone_complete(m)
                else:
                    m.status = "failed"
                    failed.append(m.milestone_id)
                    break
            else:
                # Run independent milestones concurrently.
                for m in ready:
                    m.status = "running"

                results: dict[str, bool] = {}
                with ThreadPoolExecutor(max_workers=len(ready)) as pool:
                    futures = {
                        pool.submit(self._execute_milestone, m, plan): m
                        for m in ready
                    }
                    for fut in as_completed(futures):
                        m = futures[fut]
                        results[m.milestone_id] = fut.result()

                any_failed = False
                for m in ready:
                    ok = results[m.milestone_id]
                    if ok:
                        m.status = "completed"
                        completed.append(m.milestone_id)
                        if on_milestone_complete:
                            on_milestone_complete(m)
                    else:
                        m.status = "failed"
                        failed.append(m.milestone_id)
                        any_failed = True
                if any_failed:
                    break

        all_done = all(m.status == "completed" for m in plan.milestones)
        return {
            "goal": plan.goal,
            "status": "completed" if all_done else "failed",
            "completed": completed,
            "failed": failed,
            "total": len(plan.milestones),
        }

    def resume(self, plan: LongHorizonPlan) -> dict:
        """Resume a partially-completed plan (skips completed milestones).

        Milestones that were *running* at crash time are reset to pending.
        """
        for m in plan.milestones:
            if m.status == "running":
                m.status = "pending"
        return self.execute(plan)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_milestone(self, milestone: Milestone, plan: LongHorizonPlan) -> bool:
        """Execute one milestone's skill list via the agent."""
        try:
            robot = self._get_robot_for_milestone(milestone)
            if robot is None:
                logger.warning(
                    "No robot available for milestone %r", milestone.milestone_id
                )
                return False

            from apyrobo.core.schemas import TaskStatus
            from apyrobo.skills.executor import SkillExecutor, SkillGraph
            from apyrobo.skills.skill import Skill

            catalog = self._agent._get_skill_catalog()
            graph = SkillGraph()
            prev_id: str | None = None

            for i, step in enumerate(milestone.skills):
                skill_id = step.get("skill_id", "")
                params = step.get("parameters", {})
                skill = catalog.get(
                    skill_id,
                    Skill(skill_id=skill_id, name=skill_id, description=""),
                )
                unique_id = f"{skill_id}_{i}"
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
                graph.add_skill(
                    unique_skill,
                    depends_on=[prev_id] if prev_id else None,
                    parameters=params,
                )
                prev_id = unique_id

            executor = SkillExecutor(robot)
            result = executor.execute_graph(graph)
            status_val = (
                result.status.value
                if hasattr(result.status, "value")
                else str(result.status)
            )
            return status_val == "completed"
        except Exception as exc:
            logger.warning(
                "Milestone %r failed with exception: %s",
                milestone.milestone_id, exc,
            )
            return False

    def _get_robot_for_milestone(self, milestone: Milestone) -> Any:
        """Return the robot to use for a milestone, or None if unavailable."""
        if self._fleet_manager is not None:
            if milestone.assigned_robot:
                get = getattr(self._fleet_manager, "get_robot", None)
                if callable(get):
                    return get(milestone.assigned_robot)
            get_all = getattr(self._fleet_manager, "get_robots", None)
            if callable(get_all):
                robots = get_all()
                return robots[0] if robots else None
        return None


# ---------------------------------------------------------------------------
# MockLongHorizonPlanner
# ---------------------------------------------------------------------------

_DEFAULT_MILESTONES = [
    Milestone(
        milestone_id="m1",
        description="Navigate to zone A",
        skills=[{"skill_id": "navigate_to", "parameters": {"x": 1.0, "y": 0.0}}],
        depends_on=[],
    ),
    Milestone(
        milestone_id="m2",
        description="Inspect zone A",
        skills=[{"skill_id": "report_status", "parameters": {}}],
        depends_on=["m1"],
    ),
    Milestone(
        milestone_id="m3",
        description="Return to base",
        skills=[{"skill_id": "navigate_to", "parameters": {"x": 0.0, "y": 0.0}}],
        depends_on=["m2"],
    ),
]


class MockLongHorizonPlanner:
    """Deterministic planner for tests — always returns a fixed 3-milestone plan."""

    def __init__(self, agent: Any = None, fleet_manager: Any = None) -> None:
        self._agent = agent
        self._fleet_manager = fleet_manager
        self._fixed_milestones: list[Milestone] | None = None
        self.plan_calls: list[dict] = []

    def set_plan(self, milestones: list[Milestone]) -> None:
        """Override the default 3-milestone plan."""
        self._fixed_milestones = milestones

    def plan(
        self,
        goal: str,
        robots: list | None = None,
        time_budget_s: float | None = None,
    ) -> LongHorizonPlan:
        self.plan_calls.append({"goal": goal, "robots": robots, "time_budget_s": time_budget_s})
        source = self._fixed_milestones or _DEFAULT_MILESTONES
        milestones = [
            Milestone(
                milestone_id=m.milestone_id,
                description=m.description,
                skills=list(m.skills),
                depends_on=list(m.depends_on),
                assigned_robot=m.assigned_robot,
                status="pending",
            )
            for m in source
        ]
        return LongHorizonPlan(
            plan_id=str(uuid.uuid4()),
            goal=goal,
            milestones=milestones,
        )

    def execute(
        self,
        plan: LongHorizonPlan,
        on_milestone_complete: Callable[[Milestone], None] | None = None,
    ) -> dict:
        """Complete milestones in topological order (no real robot calls)."""
        completed: list[str] = []
        failed: list[str] = []

        while True:
            completed_ids = {
                m.milestone_id for m in plan.milestones if m.status == "completed"
            }
            ready = [
                m
                for m in plan.milestones
                if m.status == "pending"
                and all(dep in completed_ids for dep in m.depends_on)
            ]
            if not ready:
                break
            for m in ready:
                m.status = "completed"
                completed.append(m.milestone_id)
                if on_milestone_complete:
                    on_milestone_complete(m)

        all_done = all(m.status == "completed" for m in plan.milestones)
        return {
            "goal": plan.goal,
            "status": "completed" if all_done else "failed",
            "completed": completed,
            "failed": failed,
            "total": len(plan.milestones),
        }

    def resume(self, plan: LongHorizonPlan) -> dict:
        for m in plan.milestones:
            if m.status == "running":
                m.status = "pending"
        return self.execute(plan)

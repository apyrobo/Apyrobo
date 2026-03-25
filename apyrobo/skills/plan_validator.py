"""
Plan Validation — validate LLM-generated skill execution plans before running.

Classes:
    ValidationIssue   — single validation finding (error / warning / info)
    ValidationResult  — aggregate result with helper accessors
    PlanValidator     — validates a plan against registry + capabilities
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    """A single issue found during plan validation."""

    severity: str        # "error" | "warning" | "info"
    skill_name: str
    message: str
    suggestion: str = ""


@dataclass
class ValidationResult:
    """Aggregate result of plan validation."""

    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def errors(self) -> list[ValidationIssue]:
        """Return all issues with severity ``"error"``."""
        return [i for i in self.issues if i.severity == "error"]

    def warnings(self) -> list[ValidationIssue]:
        """Return all issues with severity ``"warning"``."""
        return [i for i in self.issues if i.severity == "warning"]


# ---------------------------------------------------------------------------
# PlanValidator
# ---------------------------------------------------------------------------

# Skills that require exclusive access to a resource.
# Two skills in the same plan that both claim the same exclusive resource
# produce a resource-conflict error.
_EXCLUSIVE_RESOURCES: dict[str, str] = {
    # skill_name → resource_name
    "pick_object": "gripper",
    "place_object": "gripper",
    "capture_image": "camera",
    "stream_video": "camera",
}


class PlanValidator:
    """
    Validates a skill execution plan against available capabilities.

    Checks performed:
    1. All skills in the plan exist in the registry / discovery library.
    2. Required robot capabilities are present.
    3. No circular dependencies in the skill graph.
    4. Parameter types match skill schemas (basic type check).
    5. Resource conflicts (two skills need exclusive access to the same resource).

    Args:
        discovery_registry: A ``DiscoveryRegistry`` instance.  If provided,
            capability requirements are read from skill manifests.
        skill_registry: A ``SkillRegistry`` instance.  If provided, skill
            existence is verified against installed packages too.
    """

    def __init__(
        self,
        discovery_registry: Any = None,
        skill_registry: Any = None,
    ) -> None:
        self._discovery = discovery_registry
        self._registry = skill_registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        plan: list[dict],
        available_capabilities: list[str] | None = None,
    ) -> ValidationResult:
        """
        Validate *plan* and return a :class:`ValidationResult`.

        Args:
            plan: List of step dicts:
                ``[{"skill": str, "params": dict, "depends_on": list[str]}]``
            available_capabilities: List of capability strings the robot has.
                Required for capability checks; if None those checks are skipped.

        Returns:
            :class:`ValidationResult` — ``valid`` is False if any errors exist.
        """
        issues: list[ValidationIssue] = []

        for step in plan:
            skill_name = step.get("skill", "")
            params = step.get("params", {})

            issues.extend(self._check_skill_exists(skill_name))
            if available_capabilities is not None:
                issues.extend(
                    self._check_capabilities(skill_name, available_capabilities)
                )
            issues.extend(self._check_param_types(skill_name, params))

        issues.extend(self._check_circular_deps(plan))
        issues.extend(self._check_resource_conflicts(plan))

        has_errors = any(i.severity == "error" for i in issues)
        result = ValidationResult(valid=not has_errors, issues=issues)

        logger.debug(
            "PlanValidator: %d step(s) — %d error(s), %d warning(s)",
            len(plan),
            len(result.errors()),
            len(result.warnings()),
        )
        return result

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _known_skill_names(self) -> set[str]:
        """Collect all known skill names from attached registries."""
        names: set[str] = set()
        if self._discovery is not None:
            try:
                for m in self._discovery.all_skills():
                    names.add(m.name)
            except Exception:
                pass
        if self._registry is not None:
            try:
                for skill in self._registry.all_skills():
                    sid = skill.get("id") or skill.get("name", "")
                    if sid:
                        names.add(sid)
            except Exception:
                pass
        return names

    def _check_skill_exists(self, skill_name: str) -> list[ValidationIssue]:
        if not skill_name:
            return [
                ValidationIssue(
                    severity="error",
                    skill_name="",
                    message="Plan step is missing a 'skill' name.",
                    suggestion="Add a 'skill' key to the step dict.",
                )
            ]
        known = self._known_skill_names()
        if known and skill_name not in known:
            return [
                ValidationIssue(
                    severity="error",
                    skill_name=skill_name,
                    message=f"Skill '{skill_name}' is not registered.",
                    suggestion=(
                        f"Available skills: {sorted(known)[:5]}…"
                        if len(known) > 5
                        else f"Available skills: {sorted(known)}"
                    ),
                )
            ]
        return []

    def _check_capabilities(
        self, skill_name: str, capabilities: list[str]
    ) -> list[ValidationIssue]:
        if self._discovery is None or not skill_name:
            return []
        manifest = None
        try:
            for m in self._discovery.all_skills():
                if m.name == skill_name:
                    manifest = m
                    break
        except Exception:
            return []
        if manifest is None:
            return []  # existence error already reported
        missing = [r for r in manifest.requirements if r not in capabilities]
        if missing:
            return [
                ValidationIssue(
                    severity="error",
                    skill_name=skill_name,
                    message=(
                        f"Skill '{skill_name}' requires capabilities "
                        f"{missing} which are not available."
                    ),
                    suggestion=f"Ensure robot has: {missing}",
                )
            ]
        return []

    def _check_param_types(
        self, skill_name: str, params: dict
    ) -> list[ValidationIssue]:
        """Basic schema type check — only runs when a manifest is available."""
        if self._discovery is None or not skill_name:
            return []
        manifest = None
        try:
            for m in self._discovery.all_skills():
                if m.name == skill_name:
                    manifest = m
                    break
        except Exception:
            return []
        if manifest is None or not manifest.parameters:
            return []

        issues: list[ValidationIssue] = []
        schema = manifest.parameters
        properties: dict = schema.get("properties", {})
        required: list[str] = schema.get("required", [])

        _TYPE_MAP: dict[str, type] = {
            "number": (int, float),
            "integer": int,
            "string": str,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        # Check required params are present
        for req_key in required:
            if req_key not in params:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        skill_name=skill_name,
                        message=f"Required parameter '{req_key}' missing for skill '{skill_name}'.",
                        suggestion=f"Add '{req_key}' to the params dict.",
                    )
                )

        # Check types of provided params
        for key, value in params.items():
            if key not in properties:
                continue
            expected_type_str: str = properties[key].get("type", "")
            expected_python = _TYPE_MAP.get(expected_type_str)
            if expected_python and not isinstance(value, expected_python):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        skill_name=skill_name,
                        message=(
                            f"Parameter '{key}' for skill '{skill_name}' "
                            f"has type {type(value).__name__!r}; "
                            f"expected '{expected_type_str}'."
                        ),
                        suggestion=f"Convert '{key}' to {expected_type_str}.",
                    )
                )
        return issues

    def _check_circular_deps(self, plan: list[dict]) -> list[ValidationIssue]:
        """Detect cycles in the depends_on graph using DFS."""
        # Build adjacency: step_name → set of dependencies
        adjacency: dict[str, set[str]] = {}
        skill_names = [step.get("skill", f"step_{i}") for i, step in enumerate(plan)]

        for step in plan:
            name = step.get("skill", "")
            deps = step.get("depends_on", [])
            adjacency[name] = set(deps)

        # DFS cycle detection
        UNVISITED, VISITING, VISITED = 0, 1, 2
        color = {n: UNVISITED for n in skill_names}
        cycle_members: list[str] = []

        def dfs(node: str) -> bool:
            color[node] = VISITING
            for neighbor in adjacency.get(node, set()):
                if neighbor not in color:
                    continue
                if color[neighbor] == VISITING:
                    cycle_members.append(node)
                    return True
                if color[neighbor] == UNVISITED and dfs(neighbor):
                    cycle_members.append(node)
                    return True
            color[node] = VISITED
            return False

        for node in skill_names:
            if color.get(node) == UNVISITED:
                if dfs(node):
                    break

        if cycle_members:
            involved = list(dict.fromkeys(reversed(cycle_members)))
            return [
                ValidationIssue(
                    severity="error",
                    skill_name=", ".join(involved),
                    message=f"Circular dependency detected among skills: {involved}",
                    suggestion="Remove the dependency cycle from the plan.",
                )
            ]
        return []

    def _check_resource_conflicts(self, plan: list[dict]) -> list[ValidationIssue]:
        """Flag two skills in the plan that both need exclusive access to the same resource."""
        resource_users: dict[str, list[str]] = {}
        for step in plan:
            skill_name = step.get("skill", "")
            resource = _EXCLUSIVE_RESOURCES.get(skill_name)
            if resource:
                resource_users.setdefault(resource, []).append(skill_name)

        issues: list[ValidationIssue] = []
        for resource, users in resource_users.items():
            if len(users) > 1:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        skill_name=", ".join(users),
                        message=(
                            f"Resource conflict: skills {users} all require "
                            f"exclusive access to '{resource}'."
                        ),
                        suggestion=(
                            f"Serialise '{resource}' access or remove one of: {users}"
                        ),
                    )
                )
        return issues

"""
CI-07: Property-based tests for skill graph using Hypothesis.

Fuzzes execution order, cycle detection, and dependency logic.
"""

from __future__ import annotations

from typing import Any

import pytest

try:
    from hypothesis import given, settings, assume, HealthCheck
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

from apyrobo.skills.skill import Skill, SkillStatus
from apyrobo.skills.executor import SkillGraph
from apyrobo.core.schemas import CapabilityType


pytestmark = pytest.mark.skipif(
    not HAS_HYPOTHESIS,
    reason="hypothesis not installed",
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def skill_id_strategy() -> Any:
    return st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz_",
        min_size=2, max_size=10,
    ).filter(lambda s: s[0].isalpha())


def skill_strategy() -> Any:
    return st.builds(
        Skill,
        skill_id=skill_id_strategy(),
        name=st.text(min_size=1, max_size=20),
        required_capability=st.just(CapabilityType.CUSTOM),
        timeout_seconds=st.floats(min_value=0.1, max_value=60.0),
        retry_count=st.integers(min_value=0, max_value=3),
    )


# ===========================================================================
# Property-based tests
# ===========================================================================


@given(skills=st.lists(skill_strategy(), min_size=1, max_size=10))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_graph_without_deps_has_all_skills(skills: list[Skill]) -> None:
    """A graph with no deps includes all skills in execution order."""
    # Deduplicate skill IDs
    seen: set[str] = set()
    unique_skills = []
    for s in skills:
        if s.skill_id not in seen:
            seen.add(s.skill_id)
            unique_skills.append(s)

    g = SkillGraph()
    for s in unique_skills:
        g.add_skill(s)

    order = g.get_execution_order()
    assert len(order) == len(unique_skills)
    order_ids = {s.skill_id for s in order}
    expected_ids = {s.skill_id for s in unique_skills}
    assert order_ids == expected_ids


@given(skills=st.lists(skill_strategy(), min_size=2, max_size=8))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_linear_chain_preserves_order(skills: list[Skill]) -> None:
    """A linear chain A → B → C → ... preserves dependency order."""
    # Deduplicate
    seen: set[str] = set()
    unique: list[Skill] = []
    for s in skills:
        if s.skill_id not in seen:
            seen.add(s.skill_id)
            unique.append(s)

    assume(len(unique) >= 2)

    g = SkillGraph()
    g.add_skill(unique[0])
    for i in range(1, len(unique)):
        g.add_skill(unique[i], depends_on=[unique[i - 1].skill_id])

    order = g.get_execution_order()
    order_ids = [s.skill_id for s in order]

    # Check dependency ordering
    for i in range(1, len(unique)):
        idx_dep = order_ids.index(unique[i - 1].skill_id)
        idx_cur = order_ids.index(unique[i].skill_id)
        assert idx_dep < idx_cur, (
            f"{unique[i-1].skill_id} should come before {unique[i].skill_id}"
        )


@given(n=st.integers(min_value=2, max_value=6))
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_cycle_always_detected(n: int) -> None:
    """Circular dependencies are always detected."""
    skills = [Skill(skill_id=f"s{i}", name=f"S{i}") for i in range(n)]
    g = SkillGraph()
    for i, s in enumerate(skills):
        dep = skills[(i + 1) % n].skill_id
        g.add_skill(s, depends_on=[dep])

    with pytest.raises(ValueError, match="Cycle"):
        g.get_execution_order()


@given(skills=st.lists(skill_strategy(), min_size=1, max_size=8))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_layers_cover_all_skills(skills: list[Skill]) -> None:
    """Execution layers include every skill exactly once."""
    seen: set[str] = set()
    unique: list[Skill] = []
    for s in skills:
        if s.skill_id not in seen:
            seen.add(s.skill_id)
            unique.append(s)

    g = SkillGraph()
    for s in unique:
        g.add_skill(s)

    layers = g.get_execution_layers()
    all_ids = [s.skill_id for layer in layers for s in layer]
    assert len(all_ids) == len(unique)
    assert set(all_ids) == {s.skill_id for s in unique}


@given(skills=st.lists(skill_strategy(), min_size=1, max_size=8))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_graph_len_matches_skill_count(skills: list[Skill]) -> None:
    """len(graph) equals the number of unique skills added."""
    seen: set[str] = set()
    g = SkillGraph()
    count = 0
    for s in skills:
        if s.skill_id not in seen:
            seen.add(s.skill_id)
            g.add_skill(s)
            count += 1
    assert len(g) == count

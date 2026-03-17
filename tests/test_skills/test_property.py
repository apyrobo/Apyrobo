"""
CI-07: Property-based tests for skill graph using Hypothesis.

Fuzzes execution order, cycle detection, and dependency logic.
"""

from __future__ import annotations

from typing import Any

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from apyrobo.skills.skill import Skill, SkillStatus
from apyrobo.skills.executor import SkillGraph
from apyrobo.core.schemas import CapabilityType


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


# ===========================================================================
# CI-02: Full Hypothesis property tests for skill graph
# ===========================================================================


def _build_dag(num_skills: int, edge_indices: list[tuple[int, int]]) -> SkillGraph:
    """
    Build a valid DAG from a list of (src, dst) index pairs.

    Only adds edge src -> dst when src < dst (ensures acyclicity).
    """
    skills = [Skill(skill_id=f"s{i}", name=f"Skill {i}") for i in range(num_skills)]
    g = SkillGraph()
    deps: dict[int, list[str]] = {i: [] for i in range(num_skills)}

    for src, dst in edge_indices:
        if 0 <= src < dst < num_skills:
            deps[dst].append(f"s{src}")

    for i, s in enumerate(skills):
        g.add_skill(s, depends_on=deps[i] if deps[i] else [])
    return g


@st.composite
def dag_strategy(draw: st.DrawFn) -> SkillGraph:
    """Generate random valid DAGs of 2-20 skills with arbitrary dependencies."""
    n = draw(st.integers(min_value=2, max_value=20))
    # Generate edges: only src < dst are kept (ensures DAG)
    edges = draw(st.lists(
        st.tuples(st.integers(0, n - 1), st.integers(0, n - 1)),
        min_size=0,
        max_size=n * 2,
    ))
    return _build_dag(n, edges)


@given(graph=dag_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_topological_sort_no_duplicates(graph: SkillGraph) -> None:
    """Topological sort of a valid DAG returns each skill exactly once."""
    order = graph.get_execution_order()
    ids = [s.skill_id for s in order]
    assert len(ids) == len(set(ids)), "Duplicate skill IDs in topological sort"
    assert len(ids) == len(graph), "Topological sort missing skills"


@given(graph=dag_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_layers_no_intra_or_forward_deps(graph: SkillGraph) -> None:
    """
    get_execution_layers() produces layers where no skill in layer N
    depends on a skill in layer N or later.
    """
    layers = graph.get_execution_layers()
    # Map skill_id -> layer index
    layer_of: dict[str, int] = {}
    for layer_idx, layer in enumerate(layers):
        for skill in layer:
            layer_of[skill.skill_id] = layer_idx

    edges = graph.edges
    for skill_id, deps in edges.items():
        my_layer = layer_of[skill_id]
        for dep_id in deps:
            dep_layer = layer_of[dep_id]
            assert dep_layer < my_layer, (
                f"Skill {skill_id} (layer {my_layer}) depends on "
                f"{dep_id} (layer {dep_layer}) — must be strictly earlier"
            )


@given(n=st.integers(min_value=2, max_value=10))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_adding_back_edge_raises_cycle(n: int) -> None:
    """Adding a back-edge to any valid chain always raises ValueError('Cycle detected')."""
    # Build a valid chain: s0 -> s1 -> ... -> s(n-1)
    skills = [Skill(skill_id=f"s{i}", name=f"S{i}") for i in range(n)]
    g = SkillGraph()
    g.add_skill(skills[0])
    for i in range(1, n):
        g.add_skill(skills[i], depends_on=[skills[i - 1].skill_id])

    # Add a back-edge from s0 depending on s(n-1), creating a cycle
    g._edges[skills[0].skill_id] = [skills[n - 1].skill_id]

    with pytest.raises(ValueError, match="Cycle"):
        g.get_execution_order()


@given(
    base_params=st.dictionaries(
        keys=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=5),
        values=st.one_of(st.integers(), st.floats(allow_nan=False), st.text(max_size=10)),
        min_size=1,
        max_size=5,
    ),
    runtime_params=st.dictionaries(
        keys=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=5),
        values=st.one_of(st.integers(), st.floats(allow_nan=False), st.text(max_size=10)),
        min_size=1,
        max_size=5,
    ),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_parameter_merge_runtime_wins(
    base_params: dict, runtime_params: dict
) -> None:
    """Parameter merge always prefers runtime params over skill defaults."""
    skill = Skill(skill_id="test_skill", name="Test", parameters=base_params)
    g = SkillGraph()
    g.add_skill(skill, parameters=runtime_params)

    merged = g.get_parameters("test_skill")

    # All runtime params must appear in merged result
    for key, val in runtime_params.items():
        assert merged[key] == val, (
            f"Runtime param {key}={val!r} was overridden by base {merged.get(key)!r}"
        )

    # Base params not overridden by runtime should still be present
    for key, val in base_params.items():
        if key not in runtime_params:
            assert merged[key] == val


@given(graph=dag_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_parallel_execution_same_result_as_sequential(graph: SkillGraph) -> None:
    """Parallel execution of independent layers produces same set of skills as sequential."""
    order = graph.get_execution_order()
    layers = graph.get_execution_layers()

    order_ids = set(s.skill_id for s in order)
    layer_ids = set(s.skill_id for layer in layers for s in layer)

    assert order_ids == layer_ids, (
        f"Sequential order has {order_ids - layer_ids} extra, "
        f"layers have {layer_ids - order_ids} extra"
    )

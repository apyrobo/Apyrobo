"""
SK-02: Tests for SkillRegistry integration with Agent.plan().

Covers:
    - Agent(registry=reg) uses installed skills in plan catalog
    - Installed skill takes priority over built-in with same ID
    - Agent with no registry still uses BUILTIN_SKILLS correctly
    - Auto-discovery of default registry
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import CapabilityType, TaskStatus
from apyrobo.skills.agent import Agent, RuleBasedProvider
from apyrobo.skills.handlers import UnknownSkillError
from apyrobo.skills.package import SkillPackage
from apyrobo.skills.registry import SkillRegistry
from apyrobo.skills.skill import Skill, BUILTIN_SKILLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_robot() -> Robot:
    return Robot.discover("mock://sk02_bot")


@pytest.fixture
def tmp_registry(tmp_path: Path) -> SkillRegistry:
    """An empty SkillRegistry in a temp directory."""
    return SkillRegistry(tmp_path / "registry")


def _make_package(
    tmp_path: Path,
    name: str,
    skills: list[Skill],
) -> Path:
    """Create and save a skill package in tmp_path, return the directory."""
    pkg = SkillPackage(name=name, version="1.0.0", skills=skills)
    pkg_dir = tmp_path / name
    pkg.save(pkg_dir)
    return pkg_dir


# ===========================================================================
# SK-02: Registry wired into Agent.plan() catalog
# ===========================================================================


class TestAgentRegistryIntegration:
    """Agent uses registry-installed skills in its planning catalog."""

    def test_plan_uses_installed_skill(
        self, tmp_path: Path, tmp_registry: SkillRegistry, mock_robot: Robot
    ) -> None:
        """
        Install a custom skill into registry, create Agent(registry=reg),
        and verify plan() includes the installed skill in the catalog.
        """
        custom_skill = Skill(
            skill_id="warehouse_sweep",
            name="Warehouse Sweep",
            description="Sweep the warehouse floor",
            required_capability=CapabilityType.NAVIGATE,
            parameters={"zone": "A1"},
        )
        pkg_dir = _make_package(tmp_path, "sweep-pkg", [custom_skill])
        tmp_registry.install(pkg_dir)

        agent = Agent(provider="rule", registry=tmp_registry)
        catalog = agent._get_skill_catalog()

        assert "warehouse_sweep" in catalog
        assert catalog["warehouse_sweep"].name == "Warehouse Sweep"

    def test_installed_skill_overrides_builtin(
        self, tmp_path: Path, tmp_registry: SkillRegistry, mock_robot: Robot
    ) -> None:
        """
        When a registry skill has the same ID as a built-in, registry wins.
        """
        # Create a skill with the same ID as a built-in
        override_skill = Skill(
            skill_id="navigate_to",
            name="Custom Navigate",
            description="Overridden navigate_to from registry",
            required_capability=CapabilityType.NAVIGATE,
            parameters={"x": 0.0, "y": 0.0, "speed": None, "precision": "high"},
        )
        pkg_dir = _make_package(tmp_path, "custom-nav", [override_skill])
        tmp_registry.install(pkg_dir)

        agent = Agent(provider="rule", registry=tmp_registry)
        catalog = agent._get_skill_catalog()

        assert catalog["navigate_to"].description == "Overridden navigate_to from registry"
        assert "precision" in catalog["navigate_to"].parameters

    def test_no_registry_uses_builtins(self, mock_robot: Robot) -> None:
        """Agent with no registry still returns BUILTIN_SKILLS."""
        # Pass registry=None explicitly and patch DEFAULT_REGISTRY_DIR to nonexistent
        with patch("apyrobo.skills.agent.DEFAULT_REGISTRY_DIR", Path("/nonexistent")):
            agent = Agent(provider="rule", registry=None)

        catalog = agent._get_skill_catalog()
        assert set(catalog.keys()) == set(BUILTIN_SKILLS.keys())

    def test_catalog_merges_all_sources(
        self, tmp_path: Path, tmp_registry: SkillRegistry, mock_robot: Robot
    ) -> None:
        """
        Catalog contains built-ins + registry skills.
        """
        custom_skill = Skill(
            skill_id="custom_scan",
            name="Custom Scan",
            required_capability=CapabilityType.CUSTOM,
        )
        pkg_dir = _make_package(tmp_path, "scan-pkg", [custom_skill])
        tmp_registry.install(pkg_dir)

        agent = Agent(provider="rule", registry=tmp_registry)
        catalog = agent._get_skill_catalog()

        # Should have all builtins plus the custom one
        for builtin_id in BUILTIN_SKILLS:
            assert builtin_id in catalog
        assert "custom_scan" in catalog

    def test_plan_executes_with_registry_skill(
        self, tmp_path: Path, tmp_registry: SkillRegistry, mock_robot: Robot
    ) -> None:
        """
        End-to-end: Agent.execute() works when registry provides skills.
        The rule-based provider still matches on built-in keywords, so we
        verify the full pipeline doesn't break with a registry attached.
        """
        custom_skill = Skill(
            skill_id="extra_sensor",
            name="Extra Sensor",
            required_capability=CapabilityType.CUSTOM,
        )
        pkg_dir = _make_package(tmp_path, "sensor-pkg", [custom_skill])
        tmp_registry.install(pkg_dir)

        agent = Agent(provider="rule", registry=tmp_registry)
        result = agent.execute("go to position 1, 2", mock_robot)
        assert result.steps_completed >= 1


class TestAgentRegistryAutoDiscovery:
    """Auto-discovery of the default registry directory."""

    def test_auto_discovers_populated_registry(
        self, tmp_path: Path
    ) -> None:
        """
        When DEFAULT_REGISTRY_DIR exists and has packages,
        Agent auto-loads it.
        """
        reg = SkillRegistry(tmp_path / "auto_reg")
        custom_skill = Skill(
            skill_id="auto_discovered",
            name="Auto Discovered",
        )
        pkg_dir = _make_package(tmp_path, "auto-pkg", [custom_skill])
        reg.install(pkg_dir)

        with patch("apyrobo.skills.agent.DEFAULT_REGISTRY_DIR", tmp_path / "auto_reg"):
            with patch("apyrobo.skills.agent.SkillRegistry", lambda: reg):
                agent = Agent(provider="rule")

        catalog = agent._get_skill_catalog()
        assert "auto_discovered" in catalog

    def test_no_auto_discovery_if_dir_missing(self) -> None:
        """
        When DEFAULT_REGISTRY_DIR does not exist, no registry is loaded.
        """
        with patch("apyrobo.skills.agent.DEFAULT_REGISTRY_DIR", Path("/nonexistent/path")):
            agent = Agent(provider="rule")

        assert agent._registry is None
        catalog = agent._get_skill_catalog()
        assert set(catalog.keys()) == set(BUILTIN_SKILLS.keys())

    def test_explicit_registry_overrides_auto_discovery(
        self, tmp_path: Path
    ) -> None:
        """
        When registry= is explicitly passed, auto-discovery is skipped.
        """
        explicit_reg = SkillRegistry(tmp_path / "explicit_reg")
        custom_skill = Skill(
            skill_id="explicit_skill",
            name="Explicit Skill",
        )
        pkg_dir = _make_package(tmp_path, "explicit-pkg", [custom_skill])
        explicit_reg.install(pkg_dir)

        agent = Agent(provider="rule", registry=explicit_reg)

        catalog = agent._get_skill_catalog()
        assert "explicit_skill" in catalog
        assert agent._registry is explicit_reg


# ===========================================================================
# GAP-1 / GAP-2: Custom skills planned, dispatched, and error on unknown
# ===========================================================================


class TestRuleBasedProviderCatalogLookup:
    """RuleBasedProvider falls back to available_skills when no pattern matches."""

    def test_matches_custom_skill_by_name(self) -> None:
        """Custom skill matched via name/id tokens in the task string."""
        provider = RuleBasedProvider()
        available = [
            {"skill_id": "warehouse_sweep", "name": "Warehouse Sweep",
             "description": "Sweep the warehouse floor",
             "required_capability": "custom", "parameters": {"zone": "A1"}},
        ]
        plan = provider.plan("sweep the warehouse", available, ["custom"])
        assert len(plan) == 1
        assert plan[0]["skill_id"] == "warehouse_sweep"

    def test_no_match_falls_back_to_status(self) -> None:
        """When nothing matches, still fall back to report_status."""
        provider = RuleBasedProvider()
        available = [
            {"skill_id": "warehouse_sweep", "name": "Warehouse Sweep",
             "description": "Sweep the warehouse floor",
             "required_capability": "custom", "parameters": {}},
        ]
        plan = provider.plan("xyzzy frobulate", available, ["custom"])
        assert plan[0]["skill_id"] == "report_status"

    def test_hardcoded_pattern_takes_priority(self) -> None:
        """Hardcoded TASK_PATTERNS still win over catalog matching."""
        provider = RuleBasedProvider()
        available = [
            {"skill_id": "custom_navigate", "name": "Custom Navigate",
             "description": "Navigate somewhere", "required_capability": "navigate",
             "parameters": {}},
        ]
        plan = provider.plan("navigate to (3, 4)", available, ["navigate"])
        # Should match the hardcoded "navigate" pattern, not the custom skill
        assert plan[0]["skill_id"] == "navigate_to"


class TestCustomSkillDispatchUnknownError:
    """
    GAP-1: A custom skill that has no registered handler must raise
    UnknownSkillError (propagated as FAILED), not silently succeed.
    """

    def test_execute_custom_skill_without_handler_fails(
        self, tmp_path: Path, tmp_registry: SkillRegistry, mock_robot: Robot,
    ) -> None:
        """
        Install a custom skill with no handler_module. When the planner
        includes it and the executor dispatches it, the result must be FAILED.
        """
        custom_skill = Skill(
            skill_id="warehouse_sweep",
            name="Warehouse Sweep",
            description="Sweep the warehouse floor",
            required_capability=CapabilityType.CUSTOM,
            parameters={"zone": "A1"},
        )
        pkg_dir = _make_package(tmp_path, "sweep-pkg", [custom_skill])
        tmp_registry.install(pkg_dir)

        agent = Agent(provider="rule", registry=tmp_registry)
        # "sweep warehouse" should match the custom skill via catalog lookup
        result = agent.execute("sweep the warehouse", mock_robot)
        # No handler registered → UnknownSkillError → FAILED
        assert result.status == TaskStatus.FAILED

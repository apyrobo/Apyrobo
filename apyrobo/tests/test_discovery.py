"""Tests for apyrobo.skills.discovery"""
import pytest
from apyrobo.skills.discovery import SkillManifest, SkillDiscovery, DiscoveryRegistry


class TestSkillManifest:
    def test_matches_all_requirements_met(self):
        m = SkillManifest("nav", "1.0", "desc", {}, requirements=["move"])
        assert m.matches_capabilities(["move", "gripper"]) is True

    def test_matches_missing_requirement(self):
        m = SkillManifest("pick", "1.0", "desc", {}, requirements=["gripper"])
        assert m.matches_capabilities(["move"]) is False

    def test_no_requirements_always_matches(self):
        m = SkillManifest("stop", "1.0", "desc", {}, requirements=[])
        assert m.matches_capabilities([]) is True

    def test_to_dict(self):
        m = SkillManifest("nav", "1.0.0", "Navigate", {"type": "object"}, ["move"], ["/cmd_vel"])
        d = m.to_dict()
        assert d["name"] == "nav"
        assert d["version"] == "1.0.0"
        assert "/cmd_vel" in d["ros_topics"]


class TestSkillDiscovery:
    def test_scan_library_returns_builtins(self):
        disc = SkillDiscovery()
        manifests = disc.scan_library()
        names = [m.name for m in manifests]
        assert "navigate_to" in names
        assert "stop" in names

    def test_match_to_capabilities_move_only(self):
        disc = SkillDiscovery()
        matched = disc.match_to_capabilities(["move"])
        names = [m.name for m in matched]
        assert "navigate_to" in names
        assert "stop" in names
        # gripper skills should not be included
        assert "pick_object" not in names

    def test_match_to_capabilities_all(self):
        disc = SkillDiscovery()
        matched = disc.match_to_capabilities(["move", "gripper", "voice"])
        names = [m.name for m in matched]
        assert "pick_object" in names
        assert "speak" in names

    def test_register_custom_manifest(self):
        disc = SkillDiscovery()
        custom = SkillManifest("patrol", "1.0", "Patrol area", {}, requirements=["move"])
        disc.register(custom)
        names = [m.name for m in disc.scan_library()]
        assert "patrol" in names

    def test_extra_manifests_passed_to_constructor(self):
        custom = SkillManifest("scan", "1.0", "Scan env", {}, requirements=["camera"])
        disc = SkillDiscovery(extra_manifests=[custom])
        names = [m.name for m in disc.scan_library()]
        assert "scan" in names


class TestDiscoveryRegistry:
    def test_refresh_populates_cache(self):
        reg = DiscoveryRegistry()
        reg.refresh(available_capabilities=["move"])
        skills = reg.available_skills()
        assert len(skills) > 0

    def test_get_returns_manifest(self):
        reg = DiscoveryRegistry()
        reg.refresh(available_capabilities=["move"])
        m = reg.get("navigate_to")
        assert m is not None
        assert m.name == "navigate_to"

    def test_get_returns_none_missing(self):
        reg = DiscoveryRegistry()
        reg.refresh(available_capabilities=[])
        assert reg.get("nonexistent") is None

    def test_all_skills_ignores_filter(self):
        reg = DiscoveryRegistry()
        reg.refresh(available_capabilities=[])
        # available_skills would be empty (no caps), but all_skills returns everything
        assert len(reg.all_skills()) > len(reg.available_skills())

    def test_refresh_with_different_caps(self):
        reg = DiscoveryRegistry()
        reg.refresh(available_capabilities=[])
        empty = reg.available_skills()
        reg.refresh(available_capabilities=["move", "gripper", "voice"])
        full = reg.available_skills()
        assert len(full) > len(empty)

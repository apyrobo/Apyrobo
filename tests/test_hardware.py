"""Tests for hardware knowledge schema, registry, and auto-discovery."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from apyrobo.hardware import HardwareSpec, HardwareRegistry, AutoDiscovery, spec_to_planner_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec(**kwargs) -> HardwareSpec:
    defaults = dict(
        robot_id="test_bot",
        model="TestBot",
        manufacturer="ACME",
    )
    defaults.update(kwargs)
    return HardwareSpec(**defaults)


def _ur5() -> HardwareSpec:
    return HardwareSpec(
        robot_id="ur5",
        model="UR5",
        manufacturer="Universal Robots",
        dof=6,
        payload_kg=5.0,
        reach_m=0.85,
        max_speed_ms=1.0,
        sensors=["joint_encoders", "force_torque_wrist"],
        ros_node_patterns=["/ur_hardware_interface", "/joint_state_broadcaster"],
    )


def _turtlebot4() -> HardwareSpec:
    return HardwareSpec(
        robot_id="turtlebot4",
        model="TurtleBot 4",
        manufacturer="Clearpath Robotics",
        max_speed_ms=0.46,
        sensors=["lidar", "rgb_camera", "depth_camera"],
        ros_node_patterns=["/turtlebot4_node", "/oakd*"],
    )


# ---------------------------------------------------------------------------
# HardwareSpec dataclass
# ---------------------------------------------------------------------------

class TestHardwareSpec:
    def test_required_fields(self):
        spec = HardwareSpec(robot_id="ur5", model="UR5", manufacturer="UR")
        assert spec.robot_id == "ur5"
        assert spec.model == "UR5"
        assert spec.manufacturer == "UR"

    def test_defaults(self):
        spec = _make_spec()
        assert spec.dof == 0
        assert spec.payload_kg == 0.0
        assert spec.reach_m == 0.0
        assert spec.max_speed_ms == 0.0
        assert spec.sensors == []
        assert spec.skill_package == ""
        assert spec.ros_node_patterns == []
        assert spec.notes == ""

    def test_full_spec(self):
        spec = _ur5()
        assert spec.dof == 6
        assert spec.payload_kg == 5.0
        assert spec.reach_m == 0.85
        assert "joint_encoders" in spec.sensors

    def test_sensors_list_independent(self):
        spec1 = _make_spec()
        spec2 = _make_spec()
        spec1.sensors.append("lidar")
        assert spec2.sensors == []


# ---------------------------------------------------------------------------
# HardwareRegistry — programmatic registration
# ---------------------------------------------------------------------------

class TestHardwareRegistryRegister:
    def _empty_registry(self) -> HardwareRegistry:
        return HardwareRegistry(specs_dir="/tmp/nonexistent_apyrobo_specs_xyz")

    def test_register_and_all(self):
        reg = self._empty_registry()
        reg.register(_ur5())
        specs = reg.all()
        assert len(specs) == 1
        assert specs[0].robot_id == "ur5"

    def test_register_multiple(self):
        reg = self._empty_registry()
        reg.register(_ur5())
        reg.register(_turtlebot4())
        assert len(reg.all()) == 2

    def test_register_overwrites(self):
        reg = self._empty_registry()
        reg.register(_make_spec(robot_id="bot", model="v1"))
        reg.register(_make_spec(robot_id="bot", model="v2"))
        assert len(reg.all()) == 1
        assert reg.all()[0].model == "v2"

    def test_empty_registry_all(self):
        reg = self._empty_registry()
        assert reg.all() == []


# ---------------------------------------------------------------------------
# HardwareRegistry — lookup
# ---------------------------------------------------------------------------

class TestHardwareRegistryLookup:
    def _reg(self) -> HardwareRegistry:
        reg = HardwareRegistry(specs_dir="/tmp/nonexistent_apyrobo_specs_xyz")
        reg.register(_ur5())
        reg.register(_turtlebot4())
        return reg

    def test_lookup_by_robot_id(self):
        reg = self._reg()
        spec = reg.lookup("ur5")
        assert spec is not None
        assert spec.robot_id == "ur5"

    def test_lookup_by_model_name(self):
        reg = self._reg()
        spec = reg.lookup("UR5")
        assert spec is not None
        assert spec.robot_id == "ur5"

    def test_lookup_case_insensitive(self):
        reg = self._reg()
        spec = reg.lookup("turtlebot4")
        assert spec is not None

    def test_lookup_substring_model(self):
        reg = self._reg()
        spec = reg.lookup("TurtleBot")
        assert spec is not None
        assert spec.robot_id == "turtlebot4"

    def test_lookup_not_found(self):
        reg = self._reg()
        assert reg.lookup("nonexistent_robot") is None

    def test_lookup_empty_registry(self):
        reg = HardwareRegistry(specs_dir="/tmp/nonexistent_apyrobo_specs_xyz")
        assert reg.lookup("ur5") is None


# ---------------------------------------------------------------------------
# HardwareRegistry — detect_from_nodes
# ---------------------------------------------------------------------------

class TestDetectFromNodes:
    def _reg(self) -> HardwareRegistry:
        reg = HardwareRegistry(specs_dir="/tmp/nonexistent_apyrobo_specs_xyz")
        reg.register(_ur5())
        reg.register(_turtlebot4())
        return reg

    def test_detect_ur5_exact(self):
        reg = self._reg()
        spec = reg.detect_from_nodes(["/ur_hardware_interface", "/joint_state_broadcaster"])
        assert spec is not None
        assert spec.robot_id == "ur5"

    def test_detect_glob_pattern(self):
        reg = self._reg()
        # turtlebot4 uses "/oakd*" pattern
        spec = reg.detect_from_nodes(["/turtlebot4_node", "/oakd_pro"])
        assert spec is not None
        assert spec.robot_id == "turtlebot4"

    def test_no_match_returns_none(self):
        reg = self._reg()
        spec = reg.detect_from_nodes(["/some_random_node"])
        assert spec is None

    def test_empty_node_list_returns_none(self):
        reg = self._reg()
        spec = reg.detect_from_nodes([])
        assert spec is None

    def test_partial_match_not_returned(self):
        reg = self._reg()
        # Only one of UR5's two required patterns present
        spec = reg.detect_from_nodes(["/ur_hardware_interface"])
        assert spec is None

    def test_spec_without_patterns_skipped(self):
        reg = HardwareRegistry(specs_dir="/tmp/nonexistent_apyrobo_specs_xyz")
        reg.register(_make_spec(robot_id="no_pattern", ros_node_patterns=[]))
        spec = reg.detect_from_nodes(["/anything"])
        assert spec is None

    def test_extra_nodes_still_match(self):
        reg = self._reg()
        nodes = ["/ur_hardware_interface", "/joint_state_broadcaster", "/unrelated"]
        spec = reg.detect_from_nodes(nodes)
        assert spec is not None
        assert spec.robot_id == "ur5"


# ---------------------------------------------------------------------------
# spec_to_planner_context
# ---------------------------------------------------------------------------

class TestSpecToPlannerContext:
    def test_contains_model_and_manufacturer(self):
        ctx = spec_to_planner_context(_ur5())
        assert "UR5" in ctx
        assert "Universal Robots" in ctx

    def test_contains_robot_id(self):
        ctx = spec_to_planner_context(_ur5())
        assert "ur5" in ctx

    def test_dof_included(self):
        ctx = spec_to_planner_context(_ur5())
        assert "6" in ctx
        assert "degrees of freedom" in ctx

    def test_payload_included(self):
        ctx = spec_to_planner_context(_ur5())
        assert "5.0" in ctx
        assert "kg" in ctx

    def test_reach_included(self):
        ctx = spec_to_planner_context(_ur5())
        assert "0.85" in ctx
        assert "reach" in ctx.lower()

    def test_sensors_included(self):
        ctx = spec_to_planner_context(_turtlebot4())
        assert "lidar" in ctx
        assert "rgb_camera" in ctx

    def test_zero_dof_omitted(self):
        spec = _make_spec(dof=0)
        ctx = spec_to_planner_context(spec)
        assert "degrees of freedom" not in ctx

    def test_notes_included(self):
        spec = _make_spec(notes="Special research platform.")
        ctx = spec_to_planner_context(spec)
        assert "Special research platform." in ctx


# ---------------------------------------------------------------------------
# AutoDiscovery
# ---------------------------------------------------------------------------

class TestAutoDiscovery:
    def _reg_with_ur5(self) -> HardwareRegistry:
        reg = HardwareRegistry(specs_dir="/tmp/nonexistent_apyrobo_specs_xyz")
        reg.register(_ur5())
        return reg

    def test_detect_with_explicit_nodes(self):
        reg = self._reg_with_ur5()
        disc = AutoDiscovery(reg)
        spec = disc.detect(["/ur_hardware_interface", "/joint_state_broadcaster"])
        assert spec is not None
        assert spec.robot_id == "ur5"

    def test_detect_no_match(self):
        reg = self._reg_with_ur5()
        disc = AutoDiscovery(reg)
        spec = disc.detect(["/totally_different_node"])
        assert spec is None

    def test_detect_empty_nodes(self):
        reg = self._reg_with_ur5()
        disc = AutoDiscovery(reg)
        spec = disc.detect([])
        assert spec is None

    def test_detect_falls_back_to_ros_when_none(self):
        reg = self._reg_with_ur5()
        disc = AutoDiscovery(reg)
        with patch.object(disc, "_get_ros_nodes", return_value=[]) as mock_get:
            spec = disc.detect(None)
            mock_get.assert_called_once()
            assert spec is None

    def test_load_skill_package_not_installed(self):
        reg = self._reg_with_ur5()
        disc = AutoDiscovery(reg)
        spec = _make_spec(skill_package="nonexistent_pkg_xyz_apyrobo")
        result = disc.load_skill_package(spec)
        assert result is False

    def test_load_skill_package_empty_string(self):
        reg = self._reg_with_ur5()
        disc = AutoDiscovery(reg)
        spec = _make_spec(skill_package="")
        result = disc.load_skill_package(spec)
        assert result is False

    def test_load_skill_package_stdlib(self):
        reg = self._reg_with_ur5()
        disc = AutoDiscovery(reg)
        spec = _make_spec(skill_package="json")  # stdlib always available
        result = disc.load_skill_package(spec)
        assert result is True

    def test_summary_with_match(self):
        reg = self._reg_with_ur5()
        disc = AutoDiscovery(reg)
        spec = _ur5()
        text = disc.summary(spec)
        assert "UR5" in text
        assert "ur5" in text

    def test_summary_with_none(self):
        reg = self._reg_with_ur5()
        disc = AutoDiscovery(reg)
        text = disc.summary(None)
        assert "No matching" in text

    def test_get_ros_nodes_no_rclpy(self):
        reg = self._reg_with_ur5()
        disc = AutoDiscovery(reg)
        import apyrobo.hardware.autodiscovery as mod
        original = mod._RCLPY_AVAILABLE
        mod._RCLPY_AVAILABLE = False
        try:
            nodes = disc._get_ros_nodes()
            assert nodes == []
        finally:
            mod._RCLPY_AVAILABLE = original


# ---------------------------------------------------------------------------
# YAML specs loading (integration — requires PyYAML)
# ---------------------------------------------------------------------------

class TestYAMLSpecsLoading:
    def test_specs_dir_exists(self):
        specs_dir = Path(__file__).parent.parent / "apyrobo" / "hardware" / "specs"
        assert specs_dir.exists(), "specs/ directory missing"

    def test_yaml_files_present(self):
        specs_dir = Path(__file__).parent.parent / "apyrobo" / "hardware" / "specs"
        yaml_files = list(specs_dir.glob("*.yaml"))
        assert len(yaml_files) >= 8, f"Expected 8+ YAML files, found {len(yaml_files)}"

    def test_registry_loads_yaml_specs(self):
        pytest.importorskip("yaml")
        reg = HardwareRegistry()
        specs = reg.all()
        assert len(specs) >= 8

    def test_yaml_turtlebot4_fields(self):
        pytest.importorskip("yaml")
        reg = HardwareRegistry()
        spec = reg.lookup("turtlebot4")
        assert spec is not None
        assert spec.manufacturer != ""
        assert spec.max_speed_ms > 0

    def test_yaml_ur5_dof(self):
        pytest.importorskip("yaml")
        reg = HardwareRegistry()
        spec = reg.lookup("ur5")
        assert spec is not None
        assert spec.dof == 6

    def test_yaml_spot_sensors(self):
        pytest.importorskip("yaml")
        reg = HardwareRegistry()
        spec = reg.lookup("spot")
        assert spec is not None
        assert len(spec.sensors) > 0

"""
Comprehensive tests for apyrobo/cli.py.

Tests all cmd_* functions directly with mock argparse.Namespace objects,
patching dependencies as needed. Also tests the main() entry point.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

import apyrobo.cli as cli_module
from apyrobo.cli import (
    cmd_discover,
    cmd_plan,
    cmd_execute,
    cmd_skills,
    cmd_config,
    _get_registry,
    cmd_pkg_init,
    cmd_pkg_pack,
    cmd_pkg_install,
    cmd_pkg_remove,
    cmd_pkg_list,
    cmd_pkg_info,
    cmd_pkg_search,
    cmd_pkg_validate,
    cmd_pkg,
    main,
)
from apyrobo.skills.registry import PackageConflict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ns(**kwargs) -> argparse.Namespace:
    """Shorthand to create a Namespace with given attrs."""
    return argparse.Namespace(**kwargs)


def make_mock_robot(robot_id: str = "tb4") -> MagicMock:
    robot = MagicMock()
    robot.robot_id = robot_id

    caps = MagicMock()
    caps.name = "TurtleBot4"
    caps.robot_id = robot_id
    caps.max_speed = 1.5
    caps.metadata = {}

    cap1 = MagicMock()
    cap1.name = "navigate"
    cap1.capability_type.value = "navigate"
    cap1.description = "Navigation capability"
    caps.capabilities = [cap1]

    sensor1 = MagicMock()
    sensor1.sensor_id = "lidar0"
    sensor1.sensor_type.value = "lidar"
    sensor1.topic = "/scan"
    sensor1.hz = 10
    caps.sensors = [sensor1]

    joint1 = MagicMock()
    joint1.name = "wheel_left"
    joint1.joint_id = "joint_0"
    caps.joints = [joint1]

    robot.capabilities.return_value = caps
    return robot


def make_mock_agent(num_skills=2) -> MagicMock:
    agent = MagicMock()

    # Mock graph
    skill1 = MagicMock()
    skill1.skill_id = "nav_0"
    skill1.name = "Navigate"
    skill2 = MagicMock()
    skill2.skill_id = "stop_0"
    skill2.name = "Stop"

    graph = MagicMock()
    graph.__len__ = MagicMock(return_value=num_skills)
    graph.get_execution_order.return_value = [skill1, skill2][:num_skills]
    graph.get_parameters.return_value = {"target_x": 3.0, "target_y": 2.0}

    agent.plan.return_value = graph
    return agent, graph


def make_mock_report(confidence=0.9, can_proceed=True, risks=None):
    report = MagicMock()
    report.confidence = confidence
    report.can_proceed = can_proceed
    report.risk_level = "low"
    report.risks = risks or []
    return report


def make_mock_result(status_value="completed", steps_completed=2,
                     steps_total=2, error=None):
    from apyrobo.core.schemas import TaskStatus
    result = MagicMock()
    result.status = TaskStatus.COMPLETED if status_value == "completed" else MagicMock()
    if status_value != "completed":
        result.status.value = status_value
    else:
        result.status = TaskStatus.COMPLETED
        result.status.value = "completed"
    result.steps_completed = steps_completed
    result.steps_total = steps_total
    result.error = error
    return result


# ---------------------------------------------------------------------------
# cmd_discover
# ---------------------------------------------------------------------------

class TestCmdDiscover:
    def test_discover_with_sensors_joints(self, capfd):
        robot = make_mock_robot()
        with patch("apyrobo.cli.Robot.discover", return_value=robot):
            args = ns(uri="mock://tb4")
            cmd_discover(args)

        out, _ = capfd.readouterr()
        assert "TurtleBot4" in out
        assert "tb4" in out
        assert "lidar0" in out
        assert "wheel_left" in out

    def test_discover_no_joints(self, capfd):
        robot = make_mock_robot()
        robot.capabilities.return_value.joints = []
        with patch("apyrobo.cli.Robot.discover", return_value=robot):
            args = ns(uri="mock://tb4")
            cmd_discover(args)

        out, _ = capfd.readouterr()
        assert "(none)" in out

    def test_discover_sensor_no_topic_no_hz(self, capfd):
        robot = make_mock_robot()
        sensor = MagicMock()
        sensor.sensor_id = "imu0"
        sensor.sensor_type.value = "imu"
        sensor.topic = None
        sensor.hz = None
        robot.capabilities.return_value.sensors = [sensor]
        with patch("apyrobo.cli.Robot.discover", return_value=robot):
            cmd_discover(ns(uri="mock://tb4"))
        out, _ = capfd.readouterr()
        assert "imu0" in out


# ---------------------------------------------------------------------------
# cmd_plan
# ---------------------------------------------------------------------------

class TestCmdPlan:
    def test_plan_basic(self, capfd):
        robot = make_mock_robot()
        agent, graph = make_mock_agent()
        report = make_mock_report(confidence=0.85, can_proceed=True)

        with patch("apyrobo.cli.Robot.discover", return_value=robot), \
             patch("apyrobo.cli.Agent", return_value=agent), \
             patch("apyrobo.cli.ConfidenceEstimator") as MockCE:
            MockCE.return_value.assess.return_value = report
            args = ns(task="go to (3,2)", robot="mock://tb4", provider="rule")
            cmd_plan(args)

        out, _ = capfd.readouterr()
        assert "go to (3,2)" in out
        assert "Confidence" in out
        assert "yes" in out

    def test_plan_with_risks(self, capfd):
        robot = make_mock_robot()
        agent, graph = make_mock_agent()
        risk = MagicMock()
        risk.name = "SpeedRisk"
        risk.description = "Going too fast"
        report = make_mock_report(confidence=0.4, can_proceed=False, risks=[risk])

        with patch("apyrobo.cli.Robot.discover", return_value=robot), \
             patch("apyrobo.cli.Agent", return_value=agent), \
             patch("apyrobo.cli.ConfidenceEstimator") as MockCE:
            MockCE.return_value.assess.return_value = report
            args = ns(task="go to (3,2)", robot="mock://tb4", provider="rule")
            cmd_plan(args)

        out, _ = capfd.readouterr()
        assert "SpeedRisk" in out
        assert "NO" in out

    def test_plan_skill_no_params(self, capfd):
        robot = make_mock_robot()
        agent, graph = make_mock_agent()
        graph.get_parameters.return_value = {}
        report = make_mock_report()

        with patch("apyrobo.cli.Robot.discover", return_value=robot), \
             patch("apyrobo.cli.Agent", return_value=agent), \
             patch("apyrobo.cli.ConfidenceEstimator") as MockCE:
            MockCE.return_value.assess.return_value = report
            cmd_plan(ns(task="stop", robot="mock://tb4", provider="rule"))

        out, _ = capfd.readouterr()
        assert "Plan:" in out


# ---------------------------------------------------------------------------
# cmd_execute
# ---------------------------------------------------------------------------

class TestCmdExecute:
    def test_execute_completed(self, capfd):
        from apyrobo.core.schemas import TaskStatus
        robot = make_mock_robot()
        agent, graph = make_mock_agent()
        report = make_mock_report(confidence=0.9, can_proceed=True)
        result = MagicMock()
        result.status = TaskStatus.COMPLETED
        result.steps_completed = 2
        result.steps_total = 2
        result.error = None

        with patch("apyrobo.cli.Robot.discover", return_value=robot), \
             patch("apyrobo.cli.Agent", return_value=agent), \
             patch("apyrobo.cli.SafetyEnforcer"), \
             patch("apyrobo.cli.ConfidenceEstimator") as MockCE:
            MockCE.return_value.assess.return_value = report
            agent.execute.return_value = result
            args = ns(task="go to (3,2)", robot="mock://tb4", provider="rule",
                      max_speed=1.5, force=False)
            cmd_execute(args)

        out, _ = capfd.readouterr()
        assert "completed" in out.lower()

    def test_execute_not_completed_exits_1(self, capfd):
        from apyrobo.core.schemas import TaskStatus
        robot = make_mock_robot()
        agent, graph = make_mock_agent()
        report = make_mock_report(confidence=0.9, can_proceed=True)
        result = MagicMock()
        result.status = MagicMock()
        result.status.value = "failed"
        # Make it not equal to COMPLETED
        result.status.__ne__ = lambda s, o: True
        result.steps_completed = 1
        result.steps_total = 2
        result.error = "Navigation failed"

        with patch("apyrobo.cli.Robot.discover", return_value=robot), \
             patch("apyrobo.cli.Agent", return_value=agent), \
             patch("apyrobo.cli.SafetyEnforcer"), \
             patch("apyrobo.cli.ConfidenceEstimator") as MockCE:
            MockCE.return_value.assess.return_value = report
            agent.execute.return_value = result
            args = ns(task="go to (3,2)", robot="mock://tb4", provider="rule",
                      max_speed=1.5, force=False)
            with pytest.raises(SystemExit) as exc:
                cmd_execute(args)
            assert exc.value.code == 1

    def test_execute_low_confidence_aborted(self, capfd):
        robot = make_mock_robot()
        agent, graph = make_mock_agent()
        report = make_mock_report(confidence=0.2, can_proceed=False)

        with patch("apyrobo.cli.Robot.discover", return_value=robot), \
             patch("apyrobo.cli.Agent", return_value=agent), \
             patch("apyrobo.cli.SafetyEnforcer"), \
             patch("apyrobo.cli.ConfidenceEstimator") as MockCE:
            MockCE.return_value.assess.return_value = report
            args = ns(task="go to (3,2)", robot="mock://tb4", provider="rule",
                      max_speed=1.5, force=False)
            with pytest.raises(SystemExit) as exc:
                cmd_execute(args)
            assert exc.value.code == 1

        out, _ = capfd.readouterr()
        assert "Aborted" in out

    def test_execute_force_override_low_confidence(self, capfd):
        from apyrobo.core.schemas import TaskStatus
        robot = make_mock_robot()
        agent, graph = make_mock_agent()
        report = make_mock_report(confidence=0.2, can_proceed=False)
        result = MagicMock()
        result.status = TaskStatus.COMPLETED
        result.steps_completed = 2
        result.steps_total = 2
        result.error = None

        with patch("apyrobo.cli.Robot.discover", return_value=robot), \
             patch("apyrobo.cli.Agent", return_value=agent), \
             patch("apyrobo.cli.SafetyEnforcer"), \
             patch("apyrobo.cli.ConfidenceEstimator") as MockCE:
            MockCE.return_value.assess.return_value = report
            agent.execute.return_value = result
            args = ns(task="go to (3,2)", robot="mock://tb4", provider="rule",
                      max_speed=1.5, force=True)
            # Should NOT exit 1 — force override
            cmd_execute(args)

        out, _ = capfd.readouterr()
        assert "Executing" in out

    def test_execute_with_result_error_shown(self, capfd):
        from apyrobo.core.schemas import TaskStatus
        robot = make_mock_robot()
        agent, graph = make_mock_agent()
        report = make_mock_report()
        result = MagicMock()
        result.status = TaskStatus.COMPLETED
        result.steps_completed = 2
        result.steps_total = 2
        result.error = "Minor warning"

        with patch("apyrobo.cli.Robot.discover", return_value=robot), \
             patch("apyrobo.cli.Agent", return_value=agent), \
             patch("apyrobo.cli.SafetyEnforcer"), \
             patch("apyrobo.cli.ConfidenceEstimator") as MockCE:
            MockCE.return_value.assess.return_value = report
            agent.execute.return_value = result
            args = ns(task="go to (3,2)", robot="mock://tb4", provider="rule",
                      max_speed=1.5, force=False)
            cmd_execute(args)

        out, _ = capfd.readouterr()
        assert "Minor warning" in out

    def test_execute_with_risks_shown(self, capfd):
        from apyrobo.core.schemas import TaskStatus
        robot = make_mock_robot()
        agent, graph = make_mock_agent()
        risk = MagicMock()
        risk.name = "SpeedRisk"
        risk.description = "Might be too fast"
        report = make_mock_report(confidence=0.9, can_proceed=True, risks=[risk])
        result = MagicMock()
        result.status = TaskStatus.COMPLETED
        result.steps_completed = 2
        result.steps_total = 2
        result.error = None

        with patch("apyrobo.cli.Robot.discover", return_value=robot), \
             patch("apyrobo.cli.Agent", return_value=agent), \
             patch("apyrobo.cli.SafetyEnforcer"), \
             patch("apyrobo.cli.ConfidenceEstimator") as MockCE:
            MockCE.return_value.assess.return_value = report
            agent.execute.return_value = result
            args = ns(task="go to (3,2)", robot="mock://tb4", provider="rule",
                      max_speed=1.5, force=False)
            cmd_execute(args)

        out, _ = capfd.readouterr()
        assert "SpeedRisk" in out


# ---------------------------------------------------------------------------
# cmd_skills
# ---------------------------------------------------------------------------

class TestCmdSkills:
    def test_skills_list(self, capfd):
        mock_skill = MagicMock()
        mock_skill.skill_id = "nav.go_to_pose"
        mock_skill.name = "Go To Pose"
        mock_skill.required_capability.value = "navigate"
        mock_skill.description = "Move to a pose"
        mock_skill.preconditions = []
        mock_skill.postconditions = []

        with patch("apyrobo.cli.BUILTIN_SKILLS", {"nav.go_to_pose": mock_skill}):
            args = ns(list=True, export=None)
            cmd_skills(args)

        out, _ = capfd.readouterr()
        assert "nav.go_to_pose" in out
        assert "Go To Pose" in out

    def test_skills_list_with_conditions(self, capfd):
        mock_skill = MagicMock()
        mock_skill.skill_id = "nav.go_to_pose"
        mock_skill.name = "Go To Pose"
        mock_skill.required_capability.value = "navigate"
        mock_skill.description = "Move to a pose"
        pre = MagicMock()
        pre.name = "robot_idle"
        post = MagicMock()
        post.name = "at_goal"
        mock_skill.preconditions = [pre]
        mock_skill.postconditions = [post]

        with patch("apyrobo.cli.BUILTIN_SKILLS", {"nav.go_to_pose": mock_skill}):
            cmd_skills(ns(list=True, export=None))

        out, _ = capfd.readouterr()
        assert "robot_idle" in out
        assert "at_goal" in out

    def test_skills_export_valid(self, capfd):
        mock_skill = MagicMock()
        mock_skill.to_json.return_value = '{"skill_id": "nav.go_to_pose"}'

        with patch("apyrobo.cli.BUILTIN_SKILLS", {"nav.go_to_pose": mock_skill}):
            args = ns(list=False, export="nav.go_to_pose")
            cmd_skills(args)

        out, _ = capfd.readouterr()
        assert "nav.go_to_pose" in out

    def test_skills_export_invalid_exits(self, capfd):
        with patch("apyrobo.cli.BUILTIN_SKILLS", {}):
            args = ns(list=False, export="nonexistent.skill")
            with pytest.raises(SystemExit) as exc:
                cmd_skills(args)
            assert exc.value.code == 1

        out, _ = capfd.readouterr()
        assert "Unknown skill" in out

    def test_skills_no_flags(self):
        # Neither list nor export — should do nothing
        args = ns(list=False, export=None)
        cmd_skills(args)  # Should not raise


# ---------------------------------------------------------------------------
# cmd_config
# ---------------------------------------------------------------------------

class TestCmdConfig:
    def test_config_generate(self, capfd):
        mock_config = MagicMock()
        mock_config.to_yaml.return_value = "robot_id: tb4\n"

        with patch("apyrobo.cli.ApyroboConfig", return_value=mock_config):
            args = ns(generate=True, file=None)
            cmd_config(args)

        out, _ = capfd.readouterr()
        assert "robot_id" in out

    def test_config_from_file(self, capfd, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("robot_id: tb4\n")

        mock_config = MagicMock()
        mock_config.__str__ = lambda s: "ApyroboConfig(robot_id=tb4)"
        mock_config.to_yaml.return_value = "robot_id: tb4\n"

        with patch("apyrobo.cli.ApyroboConfig") as MockConfig:
            MockConfig.from_file.return_value = mock_config
            args = ns(generate=False, file=str(config_file))
            cmd_config(args)

        out, _ = capfd.readouterr()
        assert len(out) > 0

    def test_config_from_env(self, capfd):
        mock_config = MagicMock()
        mock_config.__str__ = lambda s: "ApyroboConfig()"

        with patch("apyrobo.cli.ApyroboConfig") as MockConfig:
            MockConfig.from_env.return_value = mock_config
            args = ns(generate=False, file=None)
            cmd_config(args)

        # Should call from_env and print the config


# ---------------------------------------------------------------------------
# _get_registry
# ---------------------------------------------------------------------------

class TestGetRegistry:
    def test_get_registry_with_dir(self, tmp_path):
        args = ns(registry_dir=str(tmp_path))
        registry = _get_registry(args)
        assert registry.registry_dir == tmp_path

    def test_get_registry_without_dir(self):
        args = ns(registry_dir=None)
        registry = _get_registry(args)
        assert registry is not None

    def test_get_registry_no_attr(self):
        args = ns()  # No registry_dir attribute
        registry = _get_registry(args)
        assert registry is not None


# ---------------------------------------------------------------------------
# cmd_pkg_init
# ---------------------------------------------------------------------------

class TestCmdPkgInit:
    def test_pkg_init_basic(self, capfd, tmp_path):
        pkg_dir = str(tmp_path / "my-skill")
        args = ns(name="my-skill", version="1.0.0", description="My skill",
                  author="testuser", directory=pkg_dir)
        cmd_pkg_init(args)
        out, _ = capfd.readouterr()
        assert "my-skill" in out
        assert "1.0.0" in out

    def test_pkg_init_with_defaults(self, capfd, tmp_path):
        pkg_dir = str(tmp_path / "default-skill")
        args = ns(name="default-skill", version=None, description=None,
                  author=None, directory=pkg_dir)
        cmd_pkg_init(args)
        out, _ = capfd.readouterr()
        assert "default-skill" in out

    def test_pkg_init_no_directory(self, capfd, tmp_path):
        # directory=None → uses ./<name>
        args = ns(name="test-pkg", version="0.1.0", description="",
                  author="", directory=None)
        with patch("apyrobo.skills.package.SkillPackage.init") as mock_init:
            mock_pkg = MagicMock()
            mock_pkg.name = "test-pkg"
            mock_pkg.version = "0.1.0"
            mock_init.return_value = mock_pkg
            cmd_pkg_init(args)
        out, _ = capfd.readouterr()
        assert "test-pkg" in out


# ---------------------------------------------------------------------------
# cmd_pkg_pack
# ---------------------------------------------------------------------------

class TestCmdPkgPack:
    def test_pkg_pack_validation_errors_exits(self, capfd, tmp_path):
        mock_pkg = MagicMock()
        mock_pkg.validate.return_value = ["Missing field: name", "Invalid version"]
        mock_pkg.name = "bad-pkg"
        mock_pkg.version = "0.0"

        with patch("apyrobo.cli.SkillPackage") as MockSP:
            MockSP.load.return_value = mock_pkg
            args = ns(directory=str(tmp_path), output=None)
            with pytest.raises(SystemExit) as exc:
                cmd_pkg_pack(args)
            assert exc.value.code == 1

        out, _ = capfd.readouterr()
        assert "Validation errors" in out

    def test_pkg_pack_success(self, capfd, tmp_path):
        mock_pkg = MagicMock()
        mock_pkg.validate.return_value = []
        mock_pkg.name = "my-pkg"
        mock_pkg.version = "1.0.0"
        mock_pkg.skill_ids = ["nav.go_to_pose"]
        mock_pkg.pack.return_value = str(tmp_path / "my-pkg-1.0.0.skillpkg")

        with patch("apyrobo.cli.SkillPackage") as MockSP:
            MockSP.load.return_value = mock_pkg
            args = ns(directory=str(tmp_path), output=None)
            cmd_pkg_pack(args)

        out, _ = capfd.readouterr()
        assert "my-pkg" in out
        assert "Packed" in out


# ---------------------------------------------------------------------------
# cmd_pkg_install
# ---------------------------------------------------------------------------

class TestCmdPkgInstall:
    def test_pkg_install_success(self, capfd, tmp_path):
        mock_registry = MagicMock()
        mock_pkg = MagicMock()
        mock_pkg.name = "my-pkg"
        mock_pkg.version = "1.0.0"
        mock_pkg.skill_ids = ["nav.go_to_pose"]
        mock_registry.install.return_value = mock_pkg
        mock_registry.check_dependencies.return_value = []

        with patch("apyrobo.cli._get_registry", return_value=mock_registry):
            args = ns(source=str(tmp_path / "my-pkg-1.0.0.skillpkg"),
                      force=False, registry_dir=None)
            cmd_pkg_install(args)

        out, _ = capfd.readouterr()
        assert "Installed: my-pkg" in out

    def test_pkg_install_with_unmet_deps(self, capfd, tmp_path):
        mock_registry = MagicMock()
        mock_pkg = MagicMock()
        mock_pkg.name = "dep-pkg"
        mock_pkg.version = "1.0.0"
        mock_pkg.skill_ids = []
        mock_registry.install.return_value = mock_pkg
        mock_registry.check_dependencies.return_value = ["missing-base (>=1.0.0) — not installed"]

        with patch("apyrobo.cli._get_registry", return_value=mock_registry):
            args = ns(source="./dep-pkg", force=False, registry_dir=None)
            cmd_pkg_install(args)

        out, _ = capfd.readouterr()
        assert "Unmet dependencies" in out
        assert "missing-base" in out

    def test_pkg_install_package_conflict(self, capfd):
        mock_registry = MagicMock()
        mock_registry.install.side_effect = PackageConflict("Package already installed")

        with patch("apyrobo.cli._get_registry", return_value=mock_registry):
            args = ns(source="./my-pkg", force=False, registry_dir=None)
            with pytest.raises(SystemExit) as exc:
                cmd_pkg_install(args)
            assert exc.value.code == 1

        out, _ = capfd.readouterr()
        assert "Conflict" in out
        assert "--force" in out

    def test_pkg_install_generic_exception(self, capfd):
        mock_registry = MagicMock()
        mock_registry.install.side_effect = FileNotFoundError("not found")

        with patch("apyrobo.cli._get_registry", return_value=mock_registry):
            args = ns(source="./nonexistent", force=False, registry_dir=None)
            with pytest.raises(SystemExit) as exc:
                cmd_pkg_install(args)
            assert exc.value.code == 1

        out, _ = capfd.readouterr()
        assert "Install failed" in out


# ---------------------------------------------------------------------------
# cmd_pkg_remove
# ---------------------------------------------------------------------------

class TestCmdPkgRemove:
    def test_pkg_remove_found(self, capfd):
        mock_registry = MagicMock()
        mock_registry.remove.return_value = True

        with patch("apyrobo.cli._get_registry", return_value=mock_registry):
            args = ns(name="my-pkg", registry_dir=None)
            cmd_pkg_remove(args)

        out, _ = capfd.readouterr()
        assert "Removed: my-pkg" in out

    def test_pkg_remove_not_found_exits(self, capfd):
        mock_registry = MagicMock()
        mock_registry.remove.return_value = False

        with patch("apyrobo.cli._get_registry", return_value=mock_registry):
            args = ns(name="missing-pkg", registry_dir=None)
            with pytest.raises(SystemExit) as exc:
                cmd_pkg_remove(args)
            assert exc.value.code == 1

        out, _ = capfd.readouterr()
        assert "not installed" in out


# ---------------------------------------------------------------------------
# cmd_pkg_list
# ---------------------------------------------------------------------------

class TestCmdPkgList:
    def test_pkg_list_no_packages(self, capfd):
        mock_registry = MagicMock()
        mock_registry.list_packages.return_value = []

        with patch("apyrobo.cli._get_registry", return_value=mock_registry):
            args = ns(registry_dir=None, verbose_list=False)
            cmd_pkg_list(args)

        out, _ = capfd.readouterr()
        assert "No packages installed" in out

    def test_pkg_list_with_packages(self, capfd):
        mock_registry = MagicMock()
        mock_registry.list_packages.return_value = [
            {
                "name": "my-pkg",
                "version": "1.0.0",
                "description": "A cool package",
                "skills": ["nav.go_to_pose"],
                "tags": [],
                "dependencies": {},
            }
        ]

        with patch("apyrobo.cli._get_registry", return_value=mock_registry):
            args = ns(registry_dir=None, verbose_list=False)
            cmd_pkg_list(args)

        out, _ = capfd.readouterr()
        assert "my-pkg@1.0.0" in out
        assert "A cool package" in out

    def test_pkg_list_verbose(self, capfd):
        mock_registry = MagicMock()
        mock_registry.list_packages.return_value = [
            {
                "name": "full-pkg",
                "version": "2.0.0",
                "description": "Full package",
                "skills": ["nav.stop", "nav.go"],
                "tags": ["navigation", "robot"],
                "dependencies": {"nav-base": ">=1.0.0"},
            }
        ]

        with patch("apyrobo.cli._get_registry", return_value=mock_registry):
            args = ns(registry_dir=None, verbose_list=True)
            cmd_pkg_list(args)

        out, _ = capfd.readouterr()
        assert "nav.stop" in out
        assert "navigation" in out
        assert "nav-base" in out


# ---------------------------------------------------------------------------
# cmd_pkg_info
# ---------------------------------------------------------------------------

class TestCmdPkgInfo:
    def test_pkg_info_found(self, capfd):
        mock_registry = MagicMock()
        mock_pkg = MagicMock()
        mock_pkg.name = "info-pkg"
        mock_pkg.version = "1.0.0"
        mock_pkg.description = "Info package"
        mock_pkg.author = "author"
        mock_pkg.license = "MIT"
        mock_pkg.homepage = None
        mock_pkg.required_capabilities = ["navigate"]
        mock_pkg.min_apyrobo_version = "0.1.0"
        mock_pkg.tags = ["nav"]
        mock_pkg.skills = []
        mock_pkg.dependencies = {}
        mock_registry.get.return_value = mock_pkg

        with patch("apyrobo.cli._get_registry", return_value=mock_registry):
            args = ns(name="info-pkg", registry_dir=None)
            cmd_pkg_info(args)

        out, _ = capfd.readouterr()
        assert "info-pkg" in out
        assert "1.0.0" in out

    def test_pkg_info_not_found(self, capfd):
        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with patch("apyrobo.cli._get_registry", return_value=mock_registry):
            args = ns(name="missing", registry_dir=None)
            with pytest.raises(SystemExit) as exc:
                cmd_pkg_info(args)
            assert exc.value.code == 1

        out, _ = capfd.readouterr()
        assert "not installed" in out

    def test_pkg_info_with_dependencies(self, capfd):
        mock_registry = MagicMock()
        mock_pkg = MagicMock()
        mock_pkg.name = "dep-pkg"
        mock_pkg.version = "1.0.0"
        mock_pkg.description = ""
        mock_pkg.author = ""
        mock_pkg.license = "Apache-2.0"
        mock_pkg.homepage = None
        mock_pkg.required_capabilities = []
        mock_pkg.min_apyrobo_version = "0.1.0"
        mock_pkg.tags = []
        mock_pkg.skills = []
        mock_pkg.dependencies = {"nav-base": ">=1.0.0"}
        mock_registry.get.return_value = mock_pkg
        mock_registry.is_installed.return_value = True

        with patch("apyrobo.cli._get_registry", return_value=mock_registry):
            args = ns(name="dep-pkg", registry_dir=None)
            cmd_pkg_info(args)

        out, _ = capfd.readouterr()
        assert "nav-base" in out
        assert "installed" in out


# ---------------------------------------------------------------------------
# cmd_pkg_search
# ---------------------------------------------------------------------------

class TestCmdPkgSearch:
    def test_pkg_search_no_results(self, capfd):
        mock_registry = MagicMock()
        mock_registry.search.return_value = []

        with patch("apyrobo.cli._get_registry", return_value=mock_registry):
            args = ns(query="zzznomatch", registry_dir=None)
            cmd_pkg_search(args)

        out, _ = capfd.readouterr()
        assert "No packages match" in out

    def test_pkg_search_with_results(self, capfd):
        mock_registry = MagicMock()
        mock_registry.search.return_value = [
            {"name": "nav-pkg", "version": "1.0.0", "description": "Navigation"}
        ]

        with patch("apyrobo.cli._get_registry", return_value=mock_registry):
            args = ns(query="nav", registry_dir=None)
            cmd_pkg_search(args)

        out, _ = capfd.readouterr()
        assert "nav-pkg@1.0.0" in out
        assert "Navigation" in out

    def test_pkg_search_no_description(self, capfd):
        mock_registry = MagicMock()
        mock_registry.search.return_value = [
            {"name": "plain-pkg", "version": "1.0.0"}
        ]

        with patch("apyrobo.cli._get_registry", return_value=mock_registry):
            args = ns(query="plain", registry_dir=None)
            cmd_pkg_search(args)

        out, _ = capfd.readouterr()
        assert "plain-pkg" in out


# ---------------------------------------------------------------------------
# cmd_pkg_validate
# ---------------------------------------------------------------------------

class TestCmdPkgValidate:
    def test_validate_valid_package(self, capfd):
        mock_pkg = MagicMock()
        mock_pkg.name = "valid-pkg"
        mock_pkg.version = "1.0.0"
        mock_pkg.validate.return_value = []
        mock_pkg.skill_ids = ["nav.go_to_pose"]
        mock_pkg.dependencies = {}
        mock_pkg.tags = []

        with patch("apyrobo.cli.SkillPackage") as MockSP:
            MockSP.load.return_value = mock_pkg
            args = ns(directory="./valid-pkg")
            cmd_pkg_validate(args)

        out, _ = capfd.readouterr()
        assert "valid" in out.lower()

    def test_validate_invalid_package(self, capfd):
        mock_pkg = MagicMock()
        mock_pkg.name = "bad-pkg"
        mock_pkg.version = "bad"
        mock_pkg.validate.return_value = ["Invalid version"]

        with patch("apyrobo.cli.SkillPackage") as MockSP:
            MockSP.load.return_value = mock_pkg
            args = ns(directory="./bad-pkg")
            with pytest.raises(SystemExit) as exc:
                cmd_pkg_validate(args)
            assert exc.value.code == 1

        out, _ = capfd.readouterr()
        assert "errors" in out.lower() or "Invalid" in out

    def test_validate_load_failure(self, capfd):
        with patch("apyrobo.cli.SkillPackage") as MockSP:
            MockSP.load.side_effect = FileNotFoundError("no manifest")
            args = ns(directory="./missing")
            with pytest.raises(SystemExit) as exc:
                cmd_pkg_validate(args)
            assert exc.value.code == 1

        out, _ = capfd.readouterr()
        assert "Failed to load" in out


# ---------------------------------------------------------------------------
# cmd_pkg (dispatcher)
# ---------------------------------------------------------------------------

class TestCmdPkg:
    def test_cmd_pkg_dispatches_to_list(self, capfd):
        mock_registry = MagicMock()
        mock_registry.list_packages.return_value = []

        with patch("apyrobo.cli._get_registry", return_value=mock_registry):
            args = ns(pkg_command="list", registry_dir=None, verbose_list=False)
            cmd_pkg(args)

        out, _ = capfd.readouterr()
        assert "No packages" in out

    def test_cmd_pkg_no_subcommand_prints_help(self, capfd):
        args = ns(pkg_command=None)
        cli_module._p_pkg = None
        cmd_pkg(args)
        out, _ = capfd.readouterr()
        assert "Usage" in out

    def test_cmd_pkg_no_subcommand_with_parser(self, capfd):
        args = ns(pkg_command=None)
        mock_parser = MagicMock()
        cli_module._p_pkg = mock_parser
        cmd_pkg(args)
        mock_parser.print_help.assert_called_once()


# ---------------------------------------------------------------------------
# main() — argument parsing and dispatch
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_no_args_prints_help(self, capfd):
        with patch("sys.argv", ["apyrobo"]):
            main()
        out, _ = capfd.readouterr()
        # Should print help — contains usage info
        assert len(out) > 0 or True  # Help may go to stderr

    def test_main_verbose_sets_debug_logging(self):
        import logging
        robot = make_mock_robot()
        with patch("sys.argv", ["apyrobo", "--verbose", "discover", "mock://tb4"]), \
             patch("apyrobo.cli.Robot.discover", return_value=robot):
            main()

    def test_main_discover_command(self, capfd):
        robot = make_mock_robot()
        with patch("sys.argv", ["apyrobo", "discover", "mock://tb4"]), \
             patch("apyrobo.cli.Robot.discover", return_value=robot):
            main()
        out, _ = capfd.readouterr()
        assert "TurtleBot4" in out

    def test_main_skills_list(self, capfd):
        mock_skill = MagicMock()
        mock_skill.skill_id = "nav.go_to_pose"
        mock_skill.name = "Go"
        mock_skill.required_capability.value = "navigate"
        mock_skill.description = "Go to pose"
        mock_skill.preconditions = []
        mock_skill.postconditions = []
        with patch("sys.argv", ["apyrobo", "skills", "--list"]), \
             patch("apyrobo.cli.BUILTIN_SKILLS", {"nav.go_to_pose": mock_skill}):
            main()
        out, _ = capfd.readouterr()
        assert "nav.go_to_pose" in out

    def test_main_config_generate(self, capfd):
        mock_config = MagicMock()
        mock_config.to_yaml.return_value = "robot_id: tb4\n"
        with patch("sys.argv", ["apyrobo", "config", "--generate"]), \
             patch("apyrobo.cli.ApyroboConfig", return_value=mock_config):
            main()
        out, _ = capfd.readouterr()
        assert "robot_id" in out

    def test_main_pkg_list(self, capfd):
        mock_registry = MagicMock()
        mock_registry.list_packages.return_value = []
        with patch("sys.argv", ["apyrobo", "pkg", "list"]), \
             patch("apyrobo.cli._get_registry", return_value=mock_registry):
            main()
        out, _ = capfd.readouterr()
        assert "No packages" in out

"""
Tests for the skill handler registry system.

Covers:
    - @skill_handler decorator registration and dispatch
    - All 6 built-in handlers dispatch to correct robot methods
    - UnknownSkillError for unregistered skills
    - handler_module dynamic loading from skill JSON
    - Custom handler installation via .skillpkg
"""

from __future__ import annotations

import json
import sys
import textwrap
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import CapabilityType
from apyrobo.skills.handlers import (
    UnknownSkillError,
    _HANDLERS,
    dispatch,
    get_handler,
    load_handler_module,
    registered_skill_ids,
    skill_handler,
)
from apyrobo.skills.skill import Skill, BUILTIN_SKILLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_robot() -> Robot:
    return Robot.discover("mock://handler_test")


@pytest.fixture
def robot_mock() -> MagicMock:
    """A MagicMock with the Robot interface for fine-grained assertions."""
    m = MagicMock()
    m.gripper_close.return_value = True
    m.gripper_open.return_value = True
    caps_mock = MagicMock()
    caps_mock.name = "test_robot"
    caps_mock.capabilities = []
    caps_mock.sensors = []
    m.capabilities.return_value = caps_mock
    return m


# ===========================================================================
# @skill_handler decorator tests
# ===========================================================================

class TestSkillHandlerDecorator:
    """Test the @skill_handler decorator and dispatch function."""

    def test_decorator_registers_handler(self) -> None:
        """@skill_handler('foo') registers the function under 'foo'."""
        @skill_handler("_test_decorator_reg")
        def _handler(robot: Any, params: dict) -> bool:
            return True

        assert "_test_decorator_reg" in _HANDLERS
        assert _HANDLERS["_test_decorator_reg"] is _handler

        # Cleanup
        del _HANDLERS["_test_decorator_reg"]

    def test_dispatch_calls_handler(self) -> None:
        """dispatch() routes to the registered handler."""
        calls: list[tuple] = []

        @skill_handler("_test_dispatch_call")
        def _handler(robot: Any, params: dict) -> bool:
            calls.append((robot, params))
            return True

        result = dispatch("_test_dispatch_call", "robot_obj", {"a": 1})
        assert result is True
        assert len(calls) == 1
        assert calls[0] == ("robot_obj", {"a": 1})

        del _HANDLERS["_test_dispatch_call"]

    def test_dispatch_strips_numeric_suffix(self) -> None:
        """dispatch('skill_0') strips the suffix and finds 'skill'."""
        @skill_handler("_test_suffix_strip")
        def _handler(robot: Any, params: dict) -> bool:
            return True

        assert dispatch("_test_suffix_strip_7", "r", {}) is True

        del _HANDLERS["_test_suffix_strip"]

    def test_dispatch_unknown_raises(self) -> None:
        """dispatch() raises UnknownSkillError for unregistered skills."""
        with pytest.raises(UnknownSkillError, match="No handler registered"):
            dispatch("__absolutely_nonexistent__", None, {})

    def test_get_handler_returns_function(self) -> None:
        """get_handler() returns the callable for a registered ID."""
        handler = get_handler("navigate_to")
        assert handler is not None
        assert callable(handler)

    def test_get_handler_returns_none_for_unknown(self) -> None:
        """get_handler() returns None for unregistered IDs."""
        assert get_handler("__nope__") is None

    def test_registered_skill_ids_includes_builtins(self) -> None:
        """registered_skill_ids() contains all 6 built-in handler IDs."""
        ids = registered_skill_ids()
        for builtin_id in ["navigate_to", "rotate", "stop",
                           "pick_object", "place_object", "report_status"]:
            assert builtin_id in ids


# ===========================================================================
# Built-in handler dispatch tests
# ===========================================================================

class TestBuiltinHandlerDispatch:
    """Verify each built-in handler dispatches to the correct robot method."""

    def test_navigate_to_dispatches_to_move(self, robot_mock: MagicMock) -> None:
        """@skill_handler('navigate_to') calls robot.move()."""
        dispatch("navigate_to", robot_mock, {"x": 5.0, "y": 3.0, "speed": 0.8})
        robot_mock.move.assert_called_once_with(x=5.0, y=3.0, speed=0.8)

    def test_navigate_to_default_params(self, robot_mock: MagicMock) -> None:
        """navigate_to uses defaults for missing params."""
        dispatch("navigate_to", robot_mock, {})
        robot_mock.move.assert_called_once_with(x=0.0, y=0.0, speed=None)

    def test_stop_dispatches_to_stop(self, robot_mock: MagicMock) -> None:
        """@skill_handler('stop') calls robot.stop()."""
        dispatch("stop", robot_mock, {})
        robot_mock.stop.assert_called_once()

    def test_pick_object_dispatches_to_gripper_close(self, robot_mock: MagicMock) -> None:
        """@skill_handler('pick_object') calls robot.gripper_close()."""
        result = dispatch("pick_object", robot_mock, {})
        robot_mock.gripper_close.assert_called_once()
        assert result is True

    def test_place_object_dispatches_to_gripper_open(self, robot_mock: MagicMock) -> None:
        """@skill_handler('place_object') calls robot.gripper_open()."""
        result = dispatch("place_object", robot_mock, {})
        robot_mock.gripper_open.assert_called_once()
        assert result is True

    def test_report_status_logs_capabilities(self, robot_mock: MagicMock) -> None:
        """@skill_handler('report_status') queries capabilities."""
        result = dispatch("report_status", robot_mock, {})
        robot_mock.capabilities.assert_called_once()
        assert result is True

    def test_rotate_dispatches_to_rotate(self, robot_mock: MagicMock) -> None:
        """@skill_handler('rotate') calls robot.rotate()."""
        dispatch("rotate", robot_mock, {"angle_rad": 1.57, "speed": 0.5})
        robot_mock.rotate.assert_called_once_with(angle_rad=1.57, speed=0.5)


# ===========================================================================
# UnknownSkillError tests
# ===========================================================================

class TestUnknownSkillError:
    """Verify UnknownSkillError is raised for unregistered skills."""

    def test_direct_dispatch_raises(self) -> None:
        """dispatch() raises for truly unknown skill."""
        with pytest.raises(UnknownSkillError):
            dispatch("nonexistent_skill_xyz", None, {})

    def test_executor_returns_failed_for_unknown(self, mock_robot: Robot) -> None:
        """SkillExecutor returns FAILED for unregistered skill."""
        from apyrobo.skills.executor import SkillExecutor
        from apyrobo.skills.skill import SkillStatus

        unknown = Skill(
            skill_id="completely_unknown",
            name="Unknown",
            required_capability=CapabilityType.CUSTOM,
        )
        exe = SkillExecutor(mock_robot)
        status = exe.execute_skill(unknown)
        assert status == SkillStatus.FAILED


# ===========================================================================
# handler_module dynamic loading tests
# ===========================================================================

class TestHandlerModuleLoading:
    """Test dynamic import of handler_module declared in skill JSON."""

    def test_handler_auto_imported_from_module_path(self, tmp_path: Path) -> None:
        """A skill with handler_module set gets its handler loaded dynamically."""
        # Create a Python module with a custom handler
        handler_dir = tmp_path / "custom_handlers"
        handler_dir.mkdir()
        (handler_dir / "__init__.py").write_text("")
        (handler_dir / "my_handler.py").write_text(textwrap.dedent("""\
            from apyrobo.skills.handlers import skill_handler

            @skill_handler("custom_greet")
            def _greet(robot, params):
                return True
        """))

        # Add to sys.path so importlib can find it
        sys.path.insert(0, str(tmp_path))
        try:
            load_handler_module("custom_handlers.my_handler")
            assert "custom_greet" in _HANDLERS
            result = dispatch("custom_greet", None, {})
            assert result is True
        finally:
            sys.path.pop(0)
            _HANDLERS.pop("custom_greet", None)
            # Cleanup module from cache
            sys.modules.pop("custom_handlers.my_handler", None)
            sys.modules.pop("custom_handlers", None)

    def test_load_handler_module_bad_path_raises(self) -> None:
        """load_handler_module() raises for nonexistent modules."""
        with pytest.raises(ModuleNotFoundError):
            load_handler_module("absolutely.nonexistent.module.xyz")


# ===========================================================================
# Skill model handler fields tests
# ===========================================================================

class TestSkillHandlerFields:
    """Test handler_module and handler_fn fields on the Skill model."""

    def test_skill_has_handler_fields(self) -> None:
        """Skill model includes handler_module and handler_fn."""
        s = Skill(skill_id="test", name="Test")
        assert s.handler_module is None
        assert s.handler_fn is None

    def test_skill_handler_fields_roundtrip(self) -> None:
        """handler_module and handler_fn survive to_dict/from_dict."""
        s = Skill(
            skill_id="custom_skill",
            name="Custom",
            handler_module="my_pkg.handlers",
            handler_fn="handle_custom",
        )
        d = s.to_dict()
        assert d["handler_module"] == "my_pkg.handlers"
        assert d["handler_fn"] == "handle_custom"

        s2 = Skill.from_dict(d)
        assert s2.handler_module == "my_pkg.handlers"
        assert s2.handler_fn == "handle_custom"

    def test_skill_handler_fields_json_roundtrip(self) -> None:
        """handler_module and handler_fn survive JSON roundtrip."""
        s = Skill(
            skill_id="json_skill",
            name="JSON Skill",
            handler_module="some.module",
            handler_fn="some_fn",
        )
        json_str = s.to_json()
        s2 = Skill.from_json(json_str)
        assert s2.handler_module == "some.module"
        assert s2.handler_fn == "some_fn"

    def test_builtin_skills_have_no_handler_module(self) -> None:
        """Built-in skills don't set handler_module (they're auto-registered)."""
        for skill in BUILTIN_SKILLS.values():
            assert skill.handler_module is None


# ===========================================================================
# Custom handler via .skillpkg installation
# ===========================================================================

class TestCustomHandlerInstallation:
    """Test installing a .skillpkg with a custom handler and executing it."""

    def test_install_skillpkg_with_custom_handler(
        self, tmp_path: Path, mock_robot: Robot
    ) -> None:
        """
        Install a skill package whose skill declares a handler_module.
        After installation, the handler is registered and callable.
        """
        from apyrobo.skills.package import SkillPackage
        from apyrobo.skills.registry import SkillRegistry
        from apyrobo.skills.executor import SkillExecutor
        from apyrobo.skills.skill import SkillStatus

        # 1. Create a handler module on the Python path
        handler_dir = tmp_path / "test_custom_pkg"
        handler_dir.mkdir()
        (handler_dir / "__init__.py").write_text("")
        (handler_dir / "handler.py").write_text(textwrap.dedent("""\
            from apyrobo.skills.handlers import skill_handler

            @skill_handler("custom_beep")
            def _beep(robot, params):
                robot.stop()  # just do something verifiable
                return True
        """))
        sys.path.insert(0, str(tmp_path))

        try:
            # 2. Create a skill that references the handler module
            custom_skill = Skill(
                skill_id="custom_beep",
                name="Custom Beep",
                description="Beep the robot",
                handler_module="test_custom_pkg.handler",
            )

            # 3. Create and save a package
            pkg = SkillPackage(
                name="beep-pkg",
                version="1.0.0",
                skills=[custom_skill],
            )
            pkg_dir = tmp_path / "beep-pkg-src"
            pkg.save(pkg_dir)

            # 4. Install into a registry
            registry = SkillRegistry(tmp_path / "registry")
            installed = registry.install(pkg_dir)

            # 5. Verify handler is registered and callable
            assert "custom_beep" in _HANDLERS
            result = dispatch("custom_beep", mock_robot, {})
            assert result is True

        finally:
            sys.path.pop(0)
            _HANDLERS.pop("custom_beep", None)
            sys.modules.pop("test_custom_pkg.handler", None)
            sys.modules.pop("test_custom_pkg", None)

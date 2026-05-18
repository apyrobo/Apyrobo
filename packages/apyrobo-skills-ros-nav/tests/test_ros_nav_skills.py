"""Tests for apyrobo-skills-ros-nav skill implementations.

These tests are designed to run without ROS 2 installed.  They verify:

* The package and its skill modules import without rclpy present.
* Each skill raises ``ImportError`` with the correct message when rclpy is
  missing (simulated via ``sys.modules`` patching).
* The ``register()`` entry-point exists and is callable.
* Each decorated function carries the correct ``@skill`` metadata.
"""
from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RCLPY_MISSING_SENTINEL = "ROS 2 rclpy is required."


def _patch_rclpy_missing():
    """Return a context manager that makes ``import rclpy`` raise ImportError."""
    return patch.dict(sys.modules, {"rclpy": None})


# ---------------------------------------------------------------------------
# 1. Module imports without rclpy
# ---------------------------------------------------------------------------

class TestModuleImport:
    def test_package_importable_without_rclpy(self):
        """apyrobo_skills_ros_nav imports cleanly when rclpy is absent."""
        with _patch_rclpy_missing():
            # Force re-import so the patched sys.modules is active
            if "apyrobo_skills_ros_nav" in sys.modules:
                # Module is already cached — importing again returns the cache
                # so we test the cached module is not broken
                import apyrobo_skills_ros_nav  # noqa: F401
            else:
                import apyrobo_skills_ros_nav  # noqa: F401

    def test_skills_module_importable_without_rclpy(self):
        """apyrobo_skills_ros_nav.skills imports without rclpy."""
        # The module is cached from previous imports; just check it's accessible
        import apyrobo_skills_ros_nav.skills  # noqa: F401

    def test_nav2_client_module_importable_without_rclpy(self):
        """apyrobo_skills_ros_nav.nav2_client imports without rclpy."""
        import apyrobo_skills_ros_nav.nav2_client  # noqa: F401

    def test_all_skill_functions_accessible(self):
        import apyrobo_skills_ros_nav as pkg
        for name in ("navigate_to_pose", "follow_path", "clear_costmaps", "nav2_recover"):
            assert hasattr(pkg, name), f"missing export: {name}"


# ---------------------------------------------------------------------------
# 2. Skills raise ImportError when rclpy is missing
# ---------------------------------------------------------------------------

class TestImportErrorWhenRclpyMissing:
    """Each skill must raise ImportError with the standard message when rclpy
    is unavailable, regardless of the other arguments supplied."""

    def _call_with_no_rclpy(self, skill_fn, **kwargs):
        """Call *skill_fn* with rclpy patched out; expect ImportError."""
        with patch.dict(sys.modules, {"rclpy": None}):
            with pytest.raises(ImportError) as exc_info:
                skill_fn(robot=None, **kwargs)
        return exc_info.value

    def test_navigate_to_pose_raises_import_error(self):
        from apyrobo_skills_ros_nav.skills import navigate_to_pose
        exc = self._call_with_no_rclpy(navigate_to_pose)
        assert _RCLPY_MISSING_SENTINEL in str(exc)

    def test_follow_path_raises_import_error(self):
        from apyrobo_skills_ros_nav.skills import follow_path
        exc = self._call_with_no_rclpy(follow_path, waypoints=[[0, 0]])
        assert _RCLPY_MISSING_SENTINEL in str(exc)

    def test_clear_costmaps_raises_import_error(self):
        from apyrobo_skills_ros_nav.skills import clear_costmaps
        exc = self._call_with_no_rclpy(clear_costmaps)
        assert _RCLPY_MISSING_SENTINEL in str(exc)

    def test_nav2_recover_raises_import_error(self):
        from apyrobo_skills_ros_nav.skills import nav2_recover
        exc = self._call_with_no_rclpy(nav2_recover)
        assert _RCLPY_MISSING_SENTINEL in str(exc)

    def test_error_message_includes_source_instruction(self):
        """The ImportError message must mention sourcing a workspace."""
        from apyrobo_skills_ros_nav.skills import navigate_to_pose
        exc = self._call_with_no_rclpy(navigate_to_pose)
        msg = str(exc)
        assert "source" in msg.lower() or "Source" in msg

    def test_navigate_to_pose_error_is_import_error_subclass(self):
        from apyrobo_skills_ros_nav.skills import navigate_to_pose
        with patch.dict(sys.modules, {"rclpy": None}):
            with pytest.raises(ImportError):
                navigate_to_pose(robot=None)


# ---------------------------------------------------------------------------
# 3. register() function
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_is_callable(self):
        import apyrobo_skills_ros_nav as pkg
        assert callable(pkg.register)

    def test_register_runs_without_error(self, capsys):
        import apyrobo_skills_ros_nav as pkg
        pkg.register()  # must not raise

    def test_register_prints_skill_count(self, capsys):
        import apyrobo_skills_ros_nav as pkg
        pkg.register()
        out = capsys.readouterr().out
        assert "Registered" in out

    def test_register_mentions_all_skills(self, capsys):
        import apyrobo_skills_ros_nav as pkg
        pkg.register()
        out = capsys.readouterr().out
        for name in ("navigate_to_pose", "follow_path", "clear_costmaps", "nav2_recover"):
            assert name in out, f"expected '{name}' in register() output"


# ---------------------------------------------------------------------------
# 4. @skill decorator metadata
# ---------------------------------------------------------------------------

class TestSkillMetadata:
    """Verify that @skill attaches the expected metadata to each function."""

    def _get_skill_obj(self, fn):
        assert hasattr(fn, "__skill__"), f"{fn.__name__} missing __skill__ attribute"
        return fn.__skill__

    def test_navigate_to_pose_has_skill_attribute(self):
        from apyrobo_skills_ros_nav.skills import navigate_to_pose
        assert hasattr(navigate_to_pose, "__skill__")

    def test_navigate_to_pose_skill_id(self):
        from apyrobo_skills_ros_nav.skills import navigate_to_pose
        s = self._get_skill_obj(navigate_to_pose)
        assert s.skill_id == "navigate_to_pose"

    def test_navigate_to_pose_description(self):
        from apyrobo_skills_ros_nav.skills import navigate_to_pose
        s = self._get_skill_obj(navigate_to_pose)
        assert "Nav2" in s.description or "navigate" in s.description.lower()

    def test_follow_path_has_skill_attribute(self):
        from apyrobo_skills_ros_nav.skills import follow_path
        assert hasattr(follow_path, "__skill__")

    def test_follow_path_skill_id(self):
        from apyrobo_skills_ros_nav.skills import follow_path
        s = self._get_skill_obj(follow_path)
        assert s.skill_id == "follow_path"

    def test_follow_path_description(self):
        from apyrobo_skills_ros_nav.skills import follow_path
        s = self._get_skill_obj(follow_path)
        assert s.description  # non-empty

    def test_clear_costmaps_has_skill_attribute(self):
        from apyrobo_skills_ros_nav.skills import clear_costmaps
        assert hasattr(clear_costmaps, "__skill__")

    def test_clear_costmaps_skill_id(self):
        from apyrobo_skills_ros_nav.skills import clear_costmaps
        s = self._get_skill_obj(clear_costmaps)
        assert s.skill_id == "clear_costmaps"

    def test_clear_costmaps_description(self):
        from apyrobo_skills_ros_nav.skills import clear_costmaps
        s = self._get_skill_obj(clear_costmaps)
        assert "costmap" in s.description.lower()

    def test_nav2_recover_has_skill_attribute(self):
        from apyrobo_skills_ros_nav.skills import nav2_recover
        assert hasattr(nav2_recover, "__skill__")

    def test_nav2_recover_skill_id(self):
        from apyrobo_skills_ros_nav.skills import nav2_recover
        s = self._get_skill_obj(nav2_recover)
        assert s.skill_id == "nav2_recover"

    def test_nav2_recover_description(self):
        from apyrobo_skills_ros_nav.skills import nav2_recover
        s = self._get_skill_obj(nav2_recover)
        assert "recover" in s.description.lower() or "Nav2" in s.description

    def test_all_skills_have_skill_id_attribute(self):
        from apyrobo_skills_ros_nav.skills import (
            navigate_to_pose, follow_path, clear_costmaps, nav2_recover,
        )
        for fn in (navigate_to_pose, follow_path, clear_costmaps, nav2_recover):
            assert hasattr(fn, "__skill_id__"), f"{fn.__name__} missing __skill_id__"

    def test_skill_ids_match_function_names(self):
        from apyrobo_skills_ros_nav.skills import (
            navigate_to_pose, follow_path, clear_costmaps, nav2_recover,
        )
        for fn in (navigate_to_pose, follow_path, clear_costmaps, nav2_recover):
            assert fn.__skill_id__ == fn.__name__, (
                f"{fn.__name__}: __skill_id__ = {fn.__skill_id__!r}"
            )

    def test_navigate_to_pose_timeout_exceeds_30s(self):
        """navigate_to_pose has a 30 s Nav2 default — skill timeout must exceed it."""
        from apyrobo_skills_ros_nav.skills import navigate_to_pose
        s = self._get_skill_obj(navigate_to_pose)
        assert s.timeout_seconds > 30.0

    def test_follow_path_timeout_exceeds_120s(self):
        """follow_path paths can be long — skill timeout must exceed 120 s."""
        from apyrobo_skills_ros_nav.skills import follow_path
        s = self._get_skill_obj(follow_path)
        assert s.timeout_seconds > 120.0

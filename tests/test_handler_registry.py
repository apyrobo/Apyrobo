"""
Tests for apyrobo/skills/handlers.py — HandlerRegistry class and global API.
"""

from __future__ import annotations

import pytest

from apyrobo.skills.handlers import (
    HandlerRegistry,
    UnknownSkillError,
    dispatch,
    get_handler,
    registered_skill_ids,
    skill_handler,
    _DEFAULT_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    """Fresh isolated registry for each test."""
    return HandlerRegistry()


@pytest.fixture
def mock_robot():
    class _Robot:
        last_move = None
        def move(self, x, y, speed=None):
            self.last_move = (x, y, speed)
    return _Robot()


# ---------------------------------------------------------------------------
# HandlerRegistry — registration
# ---------------------------------------------------------------------------

class TestHandlerRegistryRegistration:
    def test_register_via_decorator(self, registry):
        @registry.register("navigate_to")
        def _nav(robot, params):
            return True

        assert "navigate_to" in registry
        assert registry.get("navigate_to") is _nav

    def test_register_via_add(self, registry):
        def _stop(robot, params):
            robot.stop()
            return True

        registry.add("stop", _stop)
        assert registry.get("stop") is _stop

    def test_remove_existing(self, registry):
        registry.add("pick", lambda r, p: True)
        removed = registry.remove("pick")
        assert removed is True
        assert "pick" not in registry

    def test_remove_nonexistent(self, registry):
        removed = registry.remove("no_such_skill")
        assert removed is False

    def test_clear(self, registry):
        registry.add("a", lambda r, p: True)
        registry.add("b", lambda r, p: True)
        registry.clear()
        assert len(registry) == 0

    def test_len(self, registry):
        assert len(registry) == 0
        registry.add("x", lambda r, p: True)
        assert len(registry) == 1

    def test_contains(self, registry):
        registry.add("y", lambda r, p: True)
        assert "y" in registry
        assert "z" not in registry

    def test_skill_ids_sorted(self, registry):
        registry.add("c_skill", lambda r, p: True)
        registry.add("a_skill", lambda r, p: True)
        registry.add("b_skill", lambda r, p: True)
        assert registry.skill_ids() == ["a_skill", "b_skill", "c_skill"]

    def test_repr(self, registry):
        registry.add("x", lambda r, p: True)
        assert "HandlerRegistry" in repr(registry)
        assert "1" in repr(registry)


# ---------------------------------------------------------------------------
# HandlerRegistry — dispatch
# ---------------------------------------------------------------------------

class TestHandlerRegistryDispatch:
    def test_dispatch_exact_id(self, registry, mock_robot):
        @registry.register("navigate_to")
        def _nav(robot, params):
            robot.move(params["x"], params["y"])
            return True

        result = registry.dispatch("navigate_to", mock_robot, {"x": 1.0, "y": 2.0})
        assert result is True
        assert mock_robot.last_move == (1.0, 2.0, None)

    def test_dispatch_strips_numeric_suffix(self, registry, mock_robot):
        @registry.register("navigate_to")
        def _nav(robot, params):
            return True

        # Skill IDs like navigate_to_0, navigate_to_1 should resolve
        result = registry.dispatch("navigate_to_0", mock_robot, {})
        assert result is True

    def test_dispatch_strips_longer_suffix(self, registry, mock_robot):
        @registry.register("pick_object")
        def _pick(robot, params):
            return True

        result = registry.dispatch("pick_object_3", mock_robot, {})
        assert result is True

    def test_dispatch_unknown_raises(self, registry, mock_robot):
        with pytest.raises(UnknownSkillError):
            registry.dispatch("unknown_skill", mock_robot, {})

    def test_dispatch_unknown_with_suffix_raises(self, registry, mock_robot):
        with pytest.raises(UnknownSkillError):
            registry.dispatch("unknown_skill_0", mock_robot, {})

    def test_resolve_returns_none_for_missing(self, registry):
        assert registry.resolve("no_such") is None

    def test_resolve_exact_before_suffix_strip(self, registry):
        def _a(robot, params):
            return True
        def _b(robot, params):
            return False

        registry.add("navigate_to_0", _a)  # exact match
        registry.add("navigate_to", _b)    # base match

        # Exact match should win
        assert registry.resolve("navigate_to_0") is _a


# ---------------------------------------------------------------------------
# HandlerRegistry — module loading
# ---------------------------------------------------------------------------

class TestHandlerRegistryModuleLoading:
    def test_load_nonexistent_module_raises(self, registry):
        with pytest.raises(Exception):
            registry.load_module("apyrobo.nonexistent_module_xyz")

    def test_load_valid_module(self, registry):
        # Loading a real module should not raise
        registry.load_module("apyrobo.skills.builtins")


# ---------------------------------------------------------------------------
# Global API — backward compatibility
# ---------------------------------------------------------------------------

class TestGlobalAPI:
    def test_skill_handler_decorator_registers_globally(self):
        # Save state
        before = set(registered_skill_ids())

        @skill_handler("_test_global_handler_xyz")
        def _handler(robot, params):
            return True

        after = set(registered_skill_ids())
        assert "_test_global_handler_xyz" in after - before

        # Cleanup
        _DEFAULT_REGISTRY.remove("_test_global_handler_xyz")

    def test_get_handler(self):
        @skill_handler("_test_get_handler_xyz")
        def _h(robot, params):
            return True

        assert get_handler("_test_get_handler_xyz") is _h

        # Cleanup
        _DEFAULT_REGISTRY.remove("_test_get_handler_xyz")

    def test_registered_skill_ids_returns_list(self):
        ids = registered_skill_ids()
        assert isinstance(ids, list)

    def test_global_dispatch_with_suffix(self):
        @skill_handler("_test_dispatch_suffix_xyz")
        def _h(robot, params):
            return True

        class _R:
            pass

        result = dispatch("_test_dispatch_suffix_xyz_0", _R(), {})
        assert result is True

        _DEFAULT_REGISTRY.remove("_test_dispatch_suffix_xyz")

    def test_global_dispatch_unknown_raises(self):
        with pytest.raises(UnknownSkillError):
            dispatch("_test_no_such_handler_xyz_abc", object(), {})


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_handler_can_return_false(self, registry, mock_robot):
        @registry.register("failing_skill")
        def _fail(robot, params):
            return False

        result = registry.dispatch("failing_skill", mock_robot, {})
        assert result is False

    def test_handler_overwrite(self, registry, mock_robot):
        registry.add("my_skill", lambda r, p: False)
        registry.add("my_skill", lambda r, p: True)  # overwrite

        result = registry.dispatch("my_skill", mock_robot, {})
        assert result is True

    def test_empty_skill_id_no_digit_suffix(self, registry, mock_robot):
        registry.add("abc", lambda r, p: True)
        # "abc" last char is 'c', not a digit — no suffix stripping
        result = registry.dispatch("abc", mock_robot, {})
        assert result is True

    def test_multiple_registries_isolated(self):
        r1 = HandlerRegistry()
        r2 = HandlerRegistry()
        r1.add("shared", lambda robot, p: True)

        assert "shared" in r1
        assert "shared" not in r2

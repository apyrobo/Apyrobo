"""Tests for the apyrobo plugin system."""

import pytest
from apyrobo.plugins.base import ApyroboPlugin
from apyrobo.plugins.loader import PluginLoader
from apyrobo.plugins.registry import PluginRegistry


# ── Fixtures ──────────────────────────────────────────────────────────────────

class _GoodPlugin(ApyroboPlugin):
    name = "good_plugin"
    version = "1.0.0"
    description = "A test plugin."

    def initialize(self, config: dict) -> None:
        self._initialized = True
        self._config = config

    def teardown(self) -> None:
        self._initialized = False

    def skills(self) -> list:
        return ["skill_a"]

    def adapters(self) -> list:
        return []


class _BadPlugin(ApyroboPlugin):
    name = "bad_plugin"
    version = "0.1.0"
    description = "A plugin that fails on initialize."

    def initialize(self, config: dict) -> None:
        raise RuntimeError("init failure")

    def teardown(self) -> None:
        pass

    def skills(self) -> list:
        return []

    def adapters(self) -> list:
        return []


# ── Base class tests ───────────────────────────────────────────────────────────

class TestApyroboPlugin:
    def test_abstract_not_instantiable(self):
        with pytest.raises(TypeError):
            ApyroboPlugin()

    def test_concrete_subclass(self):
        plugin = _GoodPlugin()
        assert plugin.name == "good_plugin"
        assert plugin.version == "1.0.0"

    def test_initialize(self):
        plugin = _GoodPlugin()
        plugin.initialize({"key": "val"})
        assert plugin._initialized is True
        assert plugin._config == {"key": "val"}

    def test_teardown(self):
        plugin = _GoodPlugin()
        plugin.initialize({})
        plugin.teardown()
        assert plugin._initialized is False

    def test_skills(self):
        plugin = _GoodPlugin()
        assert "skill_a" in plugin.skills()

    def test_adapters_empty(self):
        plugin = _GoodPlugin()
        assert plugin.adapters() == []


# ── Loader tests ───────────────────────────────────────────────────────────────

class TestPluginLoader:
    def test_discover_returns_list(self):
        loader = PluginLoader()
        result = loader.discover()
        assert isinstance(result, list)

    def test_discover_returns_list_of_types(self):
        loader = PluginLoader()
        result = loader.discover()
        assert isinstance(result, list)
        assert all(isinstance(c, type) for c in result)

    def test_load_from_path_invalid(self):
        loader = PluginLoader()
        with pytest.raises((ImportError, FileNotFoundError, AttributeError)):
            loader.load_from_path("/nonexistent/plugin.py")


# ── Registry tests ─────────────────────────────────────────────────────────────

class TestPluginRegistry:
    def setup_method(self):
        self.registry = PluginRegistry()

    def test_register_and_get(self):
        plugin = _GoodPlugin()
        plugin.initialize({})
        self.registry.register(plugin)
        retrieved = self.registry.get("good_plugin")
        assert retrieved is plugin

    def test_get_missing_returns_none(self):
        assert self.registry.get("nonexistent") is None

    def test_list_plugins(self):
        plugin = _GoodPlugin()
        plugin.initialize({})
        self.registry.register(plugin)
        listing = self.registry.list_plugins()
        assert any(p["name"] == "good_plugin" for p in listing)

    def test_initialize_all_success(self):
        plugin = _GoodPlugin()
        self.registry.register(plugin)
        results = self.registry.initialize_all({})
        assert results.get("good_plugin") == "ok"

    def test_initialize_all_failure(self):
        # Pre-register and simulate failure by patching initialize
        plugin = _GoodPlugin()
        self.registry.register(plugin)
        original = plugin.initialize
        plugin.initialize = lambda cfg: (_ for _ in ()).throw(RuntimeError("fail"))
        results = self.registry.initialize_all({})
        assert results.get("good_plugin") != "ok"
        plugin.initialize = original

    def test_teardown_all(self):
        plugin = _GoodPlugin()
        self.registry.register(plugin)
        self.registry.teardown_all()
        assert plugin._initialized is False

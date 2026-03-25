"""Plugin registry — manages loaded plugin instances."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apyrobo.plugins.base import ApyroboPlugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Manages a collection of :class:`~apyrobo.plugins.base.ApyroboPlugin` instances.

    Typical usage::

        registry = PluginRegistry()
        loader = PluginLoader()
        for cls in loader.discover():
            registry.register(cls())
        results = registry.initialize_all(config={"debug": True})
    """

    def __init__(self) -> None:
        self._plugins: dict[str, ApyroboPlugin] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, plugin: ApyroboPlugin) -> None:
        """Add *plugin* to the registry.

        Args:
            plugin: An already-instantiated plugin.

        Raises:
            ValueError: If a plugin with the same name is already registered,
                or if the plugin fails its own :meth:`~ApyroboPlugin.validate`.
        """
        plugin.validate()
        if plugin.name in self._plugins:
            raise ValueError(
                f"A plugin named {plugin.name!r} is already registered. "
                "Use a unique name or unregister the existing plugin first."
            )
        self._plugins[plugin.name] = plugin
        logger.debug("Registered plugin %r v%s", plugin.name, plugin.version)

    def unregister(self, name: str) -> None:
        """Remove a plugin by name (calls :meth:`~ApyroboPlugin.teardown` first).

        Args:
            name: The plugin's :attr:`~ApyroboPlugin.name`.

        Raises:
            KeyError: If no plugin with *name* is registered.
        """
        plugin = self._plugins.pop(name)
        try:
            plugin.teardown()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error during teardown of plugin %r: %s", name, exc)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ApyroboPlugin | None:
        """Return the plugin registered under *name*, or ``None``."""
        return self._plugins.get(name)

    def list_plugins(self) -> list[dict]:
        """Return a summary list of all registered plugins.

        Each entry is a dict with keys ``name``, ``version``, ``description``,
        and ``status`` (``"registered"`` for all current entries).
        """
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "status": "registered",
            }
            for p in self._plugins.values()
        ]

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def initialize_all(self, config: dict | None = None) -> dict[str, str]:
        """Call :meth:`~ApyroboPlugin.initialize` on every registered plugin.

        Args:
            config: Configuration dict passed to each plugin's ``initialize``.

        Returns:
            Mapping of plugin name → ``"ok"`` or an error message string.
        """
        cfg = config or {}
        results: dict[str, str] = {}
        for name, plugin in self._plugins.items():
            try:
                plugin.initialize(cfg)
                results[name] = "ok"
                logger.debug("Initialized plugin %r", name)
            except Exception as exc:  # noqa: BLE001
                results[name] = str(exc)
                logger.error("Failed to initialize plugin %r: %s", name, exc)
        return results

    def teardown_all(self) -> None:
        """Call :meth:`~ApyroboPlugin.teardown` on every registered plugin."""
        for name, plugin in list(self._plugins.items()):
            try:
                plugin.teardown()
                logger.debug("Tore down plugin %r", name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error during teardown of plugin %r: %s", name, exc)

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._plugins)

    def __contains__(self, name: object) -> bool:
        return name in self._plugins

    def __repr__(self) -> str:  # pragma: no cover
        names = ", ".join(self._plugins)
        return f"<PluginRegistry [{names}]>"

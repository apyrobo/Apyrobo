"""Base class for all APYROBO plugins."""

from __future__ import annotations

import abc
from typing import Any


class ApyroboPlugin(abc.ABC):
    """Base class that all APYROBO plugins must subclass.

    Third-party packages expose plugins via the ``apyrobo.plugins`` entry-point
    group.  Each plugin is a class (not an instance) registered there; the
    :class:`~apyrobo.plugins.registry.PluginRegistry` instantiates it and calls
    :meth:`initialize` before use.

    Minimal example::

        class MyPlugin(ApyroboPlugin):
            name = "my_plugin"
            version = "1.0.0"
            description = "Does something useful."

            def initialize(self, config: dict) -> None:
                self._config = config

            def skills(self) -> list:
                return [my_skill_function]

            def adapters(self) -> list:
                return []
    """

    #: Unique plugin identifier — **must** be overridden by each subclass.
    name: str = ""
    #: SemVer string for this plugin release.
    version: str = "0.0.0"
    #: One-line human-readable description.
    description: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def initialize(self, config: dict) -> None:
        """Called once after the plugin is loaded.

        Args:
            config: Arbitrary configuration dictionary passed from the
                application (e.g. loaded from ``apyrobo.yaml``).
        """

    def teardown(self) -> None:
        """Called when the plugin is unloaded or the application shuts down."""

    # ------------------------------------------------------------------
    # Extension points
    # ------------------------------------------------------------------

    def skills(self) -> list:
        """Return skill callables to register with :class:`~apyrobo.skills.registry.SkillRegistry`.

        Returns:
            A list of callables decorated with ``@skill`` or plain functions.
        """
        return []

    def adapters(self) -> list:
        """Return adapter classes to register with the capability adapter layer.

        Returns:
            A list of :class:`~apyrobo.core.adapters.CapabilityAdapter` subclasses.
        """
        return []

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`ValueError` if the plugin is mis-configured."""
        if not self.name:
            raise ValueError(
                f"{type(self).__name__} must define a non-empty `name` class attribute."
            )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ApyroboPlugin name={self.name!r} version={self.version!r}>"

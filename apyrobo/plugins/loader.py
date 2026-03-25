"""Plugin discovery and dynamic loading."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apyrobo.plugins.base import ApyroboPlugin

logger = logging.getLogger(__name__)

_ENTRY_POINT_GROUP = "apyrobo.plugins"


class PluginLoader:
    """Discovers and loads :class:`~apyrobo.plugins.base.ApyroboPlugin` subclasses.

    Two discovery strategies are provided:

    * :meth:`discover` — scans the ``apyrobo.plugins`` setuptools entry-point
      group (the standard approach for installed packages).
    * :meth:`load_from_path` — dynamically imports a ``.py`` file by absolute
      path (useful for development / testing).
    """

    # ------------------------------------------------------------------
    # Entry-point discovery
    # ------------------------------------------------------------------

    def discover(self) -> list[type[ApyroboPlugin]]:
        """Scan the ``apyrobo.plugins`` entry-point group and return plugin classes.

        Returns:
            List of :class:`~apyrobo.plugins.base.ApyroboPlugin` subclasses found
            across all installed packages.  Classes that fail to import are
            logged and skipped.
        """
        try:
            from importlib.metadata import entry_points
        except ImportError:  # Python < 3.9 fallback
            from importlib_metadata import entry_points  # type: ignore[no-redef]

        plugins: list[type[ApyroboPlugin]] = []
        import sys
        if sys.version_info >= (3, 10):
            eps = entry_points(group=_ENTRY_POINT_GROUP)
        else:
            eps = entry_points().get(_ENTRY_POINT_GROUP, [])

        for ep in eps:
            try:
                cls = ep.load()
                if self._is_valid_plugin_class(cls):
                    plugins.append(cls)
                    logger.debug("Discovered plugin %r from entry point %r", cls, ep.name)
                else:
                    logger.warning(
                        "Entry point %r loaded %r which is not an ApyroboPlugin subclass — skipping",
                        ep.name,
                        cls,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load plugin from entry point %r: %s", ep.name, exc)

        return plugins

    # ------------------------------------------------------------------
    # Path-based loading
    # ------------------------------------------------------------------

    def load_from_path(self, path: str) -> type[ApyroboPlugin]:
        """Dynamically import a plugin class from a ``.py`` file.

        The file must define **exactly one** non-abstract
        :class:`~apyrobo.plugins.base.ApyroboPlugin` subclass at module level.

        Args:
            path: Absolute (or relative) path to the Python source file.

        Returns:
            The plugin class found in the file.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If zero or more than one plugin class is found.
            ImportError: If the module cannot be imported.
        """
        from apyrobo.plugins.base import ApyroboPlugin as _Base

        resolved = Path(path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Plugin file not found: {resolved}")

        module_name = f"_apyrobo_plugin_{resolved.stem}_{id(resolved)}"
        spec = importlib.util.spec_from_file_location(module_name, resolved)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {resolved}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        candidates = [
            obj
            for obj in vars(module).values()
            if (
                isinstance(obj, type)
                and issubclass(obj, _Base)
                and obj is not _Base
                and not getattr(obj, "__abstractmethods__", None)
            )
        ]

        if not candidates:
            raise ValueError(f"No ApyroboPlugin subclass found in {resolved}")
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple ApyroboPlugin subclasses found in {resolved}: "
                + ", ".join(c.__name__ for c in candidates)
                + " — export exactly one."
            )

        cls = candidates[0]
        logger.debug("Loaded plugin %r from %s", cls.__name__, resolved)
        return cls

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_plugin_class(obj: object) -> bool:
        from apyrobo.plugins.base import ApyroboPlugin as _Base

        return (
            isinstance(obj, type)
            and issubclass(obj, _Base)
            and obj is not _Base
        )

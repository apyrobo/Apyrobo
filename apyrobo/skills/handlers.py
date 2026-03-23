"""
Skill Handler Registry — dynamic dispatch for skill execution.

Replaces the hardcoded if/elif chain in executor._dispatch_skill() with
a registry-based lookup.  Handlers are registered via the @skill_handler
decorator.

Two usage patterns are supported:

1. Global registry (convenience API — backward compatible):

    from apyrobo.skills.handlers import skill_handler, dispatch

    @skill_handler('navigate_to')
    def _navigate(robot, params):
        robot.move(x=params.get('x', 0), y=params.get('y', 0))
        return True

    result = dispatch('navigate_to_0', robot, params)

2. HandlerRegistry class (for isolation, e.g. per-robot or per-test):

    from apyrobo.skills.handlers import HandlerRegistry

    registry = HandlerRegistry()

    @registry.register('navigate_to')
    def _navigate(robot, params):
        robot.move(x=params.get('x', 0), y=params.get('y', 0))
        return True

    result = registry.dispatch('navigate_to', robot, params)
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Type alias
HandlerFn = Callable[..., bool]


class UnknownSkillError(Exception):
    """Raised when no handler is registered for a skill ID."""


# ---------------------------------------------------------------------------
# HandlerRegistry class
# ---------------------------------------------------------------------------

class HandlerRegistry:
    """
    Registry that maps skill IDs to handler callables.

    Handlers are registered via the ``register`` decorator or ``add`` method.
    Dispatch strips trailing numeric suffixes so skill-graph node IDs like
    ``navigate_to_0`` resolve to the ``navigate_to`` handler.

    Usage::

        registry = HandlerRegistry()

        @registry.register('navigate_to')
        def _navigate(robot, params):
            robot.move(x=params.get('x', 0), y=params.get('y', 0))
            return True

        ok = registry.dispatch('navigate_to_0', robot, params)
    """

    def __init__(self) -> None:
        self._handlers: dict[str, HandlerFn] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, skill_id: str) -> Callable[[HandlerFn], HandlerFn]:
        """
        Decorator that registers a handler for *skill_id*.

        The decorated function must accept ``(robot, params)`` and return bool.
        """
        def decorator(fn: HandlerFn) -> HandlerFn:
            self._handlers[skill_id] = fn
            logger.debug("Registered handler for %r", skill_id)
            return fn
        return decorator

    def add(self, skill_id: str, fn: HandlerFn) -> None:
        """Register *fn* as the handler for *skill_id* (imperative form)."""
        self._handlers[skill_id] = fn
        logger.debug("Registered handler for %r", skill_id)

    def remove(self, skill_id: str) -> bool:
        """Unregister the handler for *skill_id*. Returns True if it existed."""
        existed = skill_id in self._handlers
        self._handlers.pop(skill_id, None)
        return existed

    def clear(self) -> None:
        """Remove all registered handlers."""
        self._handlers.clear()

    # ------------------------------------------------------------------
    # Lookup and dispatch
    # ------------------------------------------------------------------

    def get(self, skill_id: str) -> HandlerFn | None:
        """Return the handler for *skill_id* (no suffix stripping), or None."""
        return self._handlers.get(skill_id)

    def resolve(self, skill_id: str) -> HandlerFn | None:
        """
        Return the handler for *skill_id*, stripping trailing numeric suffix.

        ``navigate_to_0`` → looks up ``navigate_to``.
        """
        handler = self._handlers.get(skill_id)
        if handler:
            return handler
        # Strip trailing numeric suffix (agent-generated unique IDs)
        if skill_id and skill_id[-1].isdigit():
            base_id = skill_id.rsplit("_", 1)[0]
            return self._handlers.get(base_id)
        return None

    def dispatch(self, skill_id: str, robot: Any, params: dict[str, Any]) -> bool:
        """
        Look up and invoke the handler for *skill_id*.

        Strips trailing numeric suffixes. Raises ``UnknownSkillError`` if
        no handler is found.
        """
        handler = self.resolve(skill_id)
        if handler is None:
            base_id = skill_id.rsplit("_", 1)[0] if skill_id[-1:].isdigit() else skill_id
            raise UnknownSkillError(f"No handler registered for {base_id!r}")
        return handler(robot, params)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def skill_ids(self) -> list[str]:
        """Return all registered skill IDs, sorted."""
        return sorted(self._handlers.keys())

    def __contains__(self, skill_id: str) -> bool:
        return skill_id in self._handlers

    def __len__(self) -> int:
        return len(self._handlers)

    def __repr__(self) -> str:
        return f"<HandlerRegistry handlers={len(self._handlers)}>"

    # ------------------------------------------------------------------
    # Module loading
    # ------------------------------------------------------------------

    def load_module(self, module_path: str) -> None:
        """
        Dynamically import *module_path* so its ``@registry.register``
        decorators execute and populate this registry.
        """
        try:
            importlib.import_module(module_path)
            logger.info("Loaded handler module: %s", module_path)
        except Exception:
            logger.exception("Failed to load handler module: %s", module_path)
            raise


# ---------------------------------------------------------------------------
# Global registry — backward-compatible convenience API
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY = HandlerRegistry()

# Keep a direct reference for fast module-level access (avoids attribute lookup)
_HANDLERS = _DEFAULT_REGISTRY._handlers


def skill_handler(skill_id: str) -> Callable:
    """
    Decorator that registers a function in the **global** handler registry.

    The decorated function must accept ``(robot, params)`` and return bool.
    """
    return _DEFAULT_REGISTRY.register(skill_id)


def dispatch(skill_id: str, robot: Any, params: dict[str, Any]) -> bool:
    """
    Look up and call the handler for *skill_id* in the global registry.

    Strips a trailing numeric suffix (e.g. ``navigate_to_0`` → ``navigate_to``)
    to support agent-generated unique IDs in skill graphs.

    Raises UnknownSkillError if no handler is found.
    """
    return _DEFAULT_REGISTRY.dispatch(skill_id, robot, params)


def get_handler(skill_id: str) -> HandlerFn | None:
    """Return the handler for *skill_id* (without suffix stripping), or None."""
    return _DEFAULT_REGISTRY.get(skill_id)


def registered_skill_ids() -> list[str]:
    """Return all registered handler skill IDs in the global registry."""
    return _DEFAULT_REGISTRY.skill_ids()


def load_handler_module(module_path: str) -> None:
    """
    Dynamically import a module so that its ``@skill_handler`` decorators
    execute and register handlers in the global registry.
    """
    _DEFAULT_REGISTRY.load_module(module_path)

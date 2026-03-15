"""
Skill Handler Registry — dynamic dispatch for skill execution.

Replaces the hardcoded if/elif chain in executor._dispatch_skill() with
a registry-based lookup.  Handlers are registered via the @skill_handler
decorator.

Usage:
    from apyrobo.skills.handlers import skill_handler, dispatch

    @skill_handler('navigate_to')
    def _navigate(robot, params):
        robot.move(x=params.get('x', 0), y=params.get('y', 0))
        return True

    # Later, in executor:
    result = dispatch('navigate_to_0', robot, params)
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Global handler registry: skill_id -> handler callable
_HANDLERS: dict[str, Callable[..., bool]] = {}


class UnknownSkillError(Exception):
    """Raised when no handler is registered for a skill ID."""


def skill_handler(skill_id: str) -> Callable:
    """
    Decorator that registers a function as the handler for *skill_id*.

    The decorated function must accept (robot, params) and return bool.
    """
    def decorator(fn: Callable[..., bool]) -> Callable[..., bool]:
        _HANDLERS[skill_id] = fn
        return fn
    return decorator


def dispatch(skill_id: str, robot: Any, params: dict[str, Any]) -> bool:
    """
    Look up and call the handler for *skill_id*.

    Strips a trailing numeric suffix (e.g. ``navigate_to_0`` → ``navigate_to``)
    to support agent-generated unique IDs in skill graphs.

    Raises UnknownSkillError if no handler is found.
    """
    base_id = skill_id.rsplit("_", 1)[0] if skill_id[-1:].isdigit() else skill_id
    handler = _HANDLERS.get(base_id)
    if handler:
        return handler(robot, params)
    raise UnknownSkillError(f"No handler registered for {base_id!r}")


def get_handler(skill_id: str) -> Callable[..., bool] | None:
    """Return the handler for *skill_id* (without suffix stripping), or None."""
    return _HANDLERS.get(skill_id)


def registered_skill_ids() -> list[str]:
    """Return all registered handler skill IDs."""
    return list(_HANDLERS.keys())


def load_handler_module(module_path: str) -> None:
    """
    Dynamically import a module so that its @skill_handler decorators
    execute and register handlers.
    """
    try:
        importlib.import_module(module_path)
        logger.info("Loaded handler module: %s", module_path)
    except Exception:
        logger.exception("Failed to load handler module: %s", module_path)
        raise

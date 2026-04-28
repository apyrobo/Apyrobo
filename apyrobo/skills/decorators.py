"""
@skill decorator — register Python functions as APYROBO skills.

Usage (bare decorator)::

    @skill
    def pick_cup(object_id: str, speed: float = 0.5) -> bool:
        ...

Usage (with keyword arguments)::

    @skill(description="Pick up a cup", capability="pick", timeout=30.0, retries=2)
    def pick_cup(object_id: str, speed: float = 0.5) -> bool:
        ...

The decorated function stays callable normally. ``pick_cup.__skill__`` holds
the ``Skill`` object; ``pick_cup.__skill_id__`` holds the skill ID string.

After decorating, load all skills into a library with::

    lib = SkillLibrary.from_decorated()
    agent = Agent(provider="rule", library=lib)
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable

from apyrobo.core.schemas import CapabilityType
from apyrobo.skills.skill import Skill

logger = logging.getLogger(__name__)

# skill_id -> (Skill, original_fn)
_DECORATED_SKILLS: dict[str, tuple[Skill, Callable[..., Any]]] = {}

# Map Python types (or their string names under PEP 563) to schema defaults.
_TYPE_DEFAULTS: dict[type, Any] = {
    str: "",
    float: 0.0,
    int: 0,
    bool: False,
    bytes: b"",
}
_TYPE_DEFAULTS_BY_NAME: dict[str, Any] = {
    t.__name__: v for t, v in _TYPE_DEFAULTS.items()
}


def _build_parameters(fn: Callable[..., Any]) -> dict[str, Any]:
    """Derive a skill parameter dict from a function's signature.

    Handles both eager annotations (type objects) and lazy string annotations
    produced by ``from __future__ import annotations`` (PEP 563).
    """
    sig = inspect.signature(fn)
    params: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls", "robot", "executor"):
            continue
        ann = param.annotation
        if param.default is not inspect.Parameter.empty:
            default: Any = param.default
        elif ann is inspect.Parameter.empty:
            default = None
        elif ann in _TYPE_DEFAULTS:  # actual type object
            default = _TYPE_DEFAULTS[ann]
        elif isinstance(ann, str) and ann in _TYPE_DEFAULTS_BY_NAME:  # PEP 563 string
            default = _TYPE_DEFAULTS_BY_NAME[ann]
        else:
            default = None
        params[pname] = default
    return params


def skill(
    _fn: Callable[..., Any] | None = None,
    *,
    description: str = "",
    capability: str = "custom",
    timeout: float = 60.0,
    retries: int = 0,
    skill_id: str | None = None,
    name: str | None = None,
) -> Any:
    """Register a function as an APYROBO skill.

    Works as a bare ``@skill`` decorator or with keyword arguments:
    ``@skill(description=..., capability=..., timeout=..., retries=...,
    skill_id=..., name=...)``.

    Args:
        description: Human-readable description of what the skill does.
        capability: Required robot capability (e.g. ``"pick"``, ``"navigate"``).
        timeout: Maximum execution time in seconds.
        retries: How many times to retry on failure.
        skill_id: Override the skill ID (defaults to the function name).
        name: Override the display name (defaults to title-cased function name).
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        sid = skill_id or fn.__name__
        display_name = name or sid.replace("_", " ").title()
        parameters = _build_parameters(fn)

        try:
            cap_type = CapabilityType(capability)
        except ValueError:
            logger.warning(
                "@skill: unknown capability %r for %s, defaulting to 'custom'",
                capability, sid,
            )
            cap_type = CapabilityType.CUSTOM

        skill_obj = Skill(
            skill_id=sid,
            name=display_name,
            description=description,
            required_capability=cap_type,
            parameters=parameters,
            timeout_seconds=timeout,
            retry_count=retries,
        )

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        wrapper.__skill__ = skill_obj  # type: ignore[attr-defined]
        wrapper.__skill_id__ = sid     # type: ignore[attr-defined]

        _DECORATED_SKILLS[sid] = (skill_obj, fn)
        logger.debug("@skill registered: %s", sid)
        return wrapper

    if _fn is not None:
        # Called as bare @skill (no parentheses)
        return decorator(_fn)
    return decorator


def get_decorated_skills() -> dict[str, tuple[Skill, Callable[..., Any]]]:
    """Return a snapshot of all currently registered decorated skills."""
    return dict(_DECORATED_SKILLS)


def clear_decorated_skills() -> None:
    """Remove all entries from the decorated-skills registry (useful in tests)."""
    _DECORATED_SKILLS.clear()

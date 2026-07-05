"""Load the packaged copies of the normative spec schemas.

``apyrobo/conformance/schemas/*.json`` are byte-identical copies of
``spec/schemas/*.json`` — shipped inside the wheel so an installed package
can validate targets without a repo checkout. The drift test in
``tests/test_spec_schemas.py`` fails if the copies diverge from the spec.
"""
from __future__ import annotations

import json
from importlib import resources
from typing import Any

try:
    import jsonschema

    JSONSCHEMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    JSONSCHEMA_AVAILABLE = False


def load_schema(name: str) -> dict[str, Any]:
    """Return the packaged JSON Schema *name* (e.g. 'robot-capability')."""
    path = resources.files("apyrobo.conformance") / "schemas" / f"{name}.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def validation_error(instance: Any, schema_name: str) -> str | None:
    """Validate *instance* against a packaged schema.

    Returns None when valid, the validator message when not.
    Raises ImportError when jsonschema is not installed — callers decide
    whether that means "skip" or "install apyrobo[conformance]".
    """
    if not JSONSCHEMA_AVAILABLE:
        raise ImportError(
            "jsonschema is required for schema validation checks. "
            "Install with: pip install 'apyrobo[conformance]'"
        )
    validator = jsonschema.Draft202012Validator(load_schema(schema_name))
    errors = sorted(validator.iter_errors(instance), key=lambda e: e.json_path)
    if not errors:
        return None
    first = errors[0]
    return f"{first.json_path}: {first.message}"

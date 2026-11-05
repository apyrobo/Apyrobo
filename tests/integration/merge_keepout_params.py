#!/usr/bin/env python3
"""Paste ``apyrobo mcp keepout``'s config into a Nav2 params file, as a user would.

Usage: merge_keepout_params.py NAV2_PARAMS PRINTED_CONFIG OUTPUT

Nested keys merge and the printed values win. The Nav2-in-Gazebo e2e boots on
the result, so it exercises the printed config rather than a hand-kept copy.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


def merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """``base`` with ``extra`` laid over it, recursing into mappings."""
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    base, printed, output = (Path(a) for a in argv)
    params = merge(yaml.safe_load(base.read_text()), yaml.safe_load(printed.read_text()))
    output.write_text(yaml.safe_dump(params, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""The README's "Connecting to a Robot" table must match the adapter registry.

Every scheme it lists has to resolve to a registered adapter, and a scheme is
called an "in-memory stand-in" in the README exactly when its adapter class
says so in its docstring. This keeps the table from overselling a stand-in or
underselling a real bridge (mujoco:// was listed as a stand-in long after it
became a live physics bridge).
"""

from __future__ import annotations

import functools
import json
import re
import subprocess
import sys
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"

# Read the registry in a fresh interpreter: other tests import optional
# modules (e.g. apyrobo.core.isaac_adapter) that re-register a scheme, and the
# README describes what a plain `import apyrobo` gives a user.
_SNAPSHOT = """
import json, apyrobo
from apyrobo.core.adapters import _ADAPTER_REGISTRY as R
print(json.dumps({s: "in-memory stand-in" in (c.__doc__ or "").lower()
                  for s, c in R.items()}))
"""


@functools.lru_cache(maxsize=1)
def _registry() -> dict[str, bool]:
    """scheme -> True when its adapter class calls itself an in-memory stand-in."""
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _SNAPSHOT],
        capture_output=True, text=True, check=True, timeout=120,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def _scheme_rows() -> dict[str, str]:
    text = README.read_text(encoding="utf-8")
    section = text.split("## Connecting to a Robot", 1)[1].split("\n## ", 1)[0]
    rows: dict[str, str] = {}
    for line in section.splitlines():
        match = re.match(r"\|\s*`([a-z0-9_]+)://`\s*\|(.*)", line)
        if match:
            rows[match.group(1)] = match.group(2)
    return rows


def test_table_lists_schemes():
    rows = _scheme_rows()
    assert {"mock", "ros2", "mujoco", "vda5050"} <= set(rows)


def test_every_listed_scheme_is_registered():
    missing = [s for s in _scheme_rows() if s not in _registry()]
    assert not missing, f"README lists unregistered schemes: {missing}"


def test_stand_in_labels_match_adapters():
    wrong = []
    for scheme, row in _scheme_rows().items():
        says_stand_in = "stand-in" in row.lower()
        if says_stand_in != _registry()[scheme]:
            wrong.append(scheme)
    assert not wrong, f"README stand-in label disagrees with the adapter: {wrong}"


def test_registered_stand_ins_are_listed():
    rows = _scheme_rows()
    unlisted = [
        s for s, stand_in in _registry().items()
        if stand_in and s not in rows
    ]
    assert not unlisted, f"stand-in schemes missing from the README table: {unlisted}"

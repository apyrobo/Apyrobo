"""Importing the Isaac SDK adapter must not change what ``isaac://`` means.

``isaac://`` is the in-memory stand-in from ``apyrobo.sim.adapters``. The SDK
adapter in ``apyrobo.core.isaac_adapter`` used to register itself under the
same scheme on import, so an unrelated import silently swapped the adapter a
user got. It now registers ``isaac_sim://`` instead. Each check runs in a fresh
interpreter so other tests' imports cannot hide the bug.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

_PROBE = """
import json, sys, importlib
import apyrobo
from apyrobo.core.adapters import _ADAPTER_REGISTRY as R

def snap():
    return {s: (R[s].__module__ if s in R else None) for s in ("isaac", "isaac_sim")}

order = sys.argv[1]
if order == "core_first":
    importlib.import_module("apyrobo.core.isaac_adapter")
    importlib.import_module("apyrobo.sim.adapters")
    after = snap()
else:
    importlib.import_module("apyrobo.sim.adapters")
    importlib.import_module("apyrobo.core.isaac_adapter")
    after = snap()

from apyrobo.core.robot import Robot
backend = Robot.discover("isaac://scene").capabilities().metadata.get("backend")
print(json.dumps({"after": after, "backend": backend}))
"""


def _probe(order: str) -> dict:
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _PROBE, order],
        capture_output=True, text=True, check=True, timeout=120,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


@pytest.mark.parametrize("order", ["core_first", "sim_first"])
def test_import_order_does_not_change_isaac_scheme(order: str) -> None:
    result = _probe(order)
    assert result["after"]["isaac"] == "apyrobo.sim.adapters"
    assert result["backend"] == "isaac_sim"  # the stand-in's metadata


@pytest.mark.parametrize("order", ["core_first", "sim_first"])
def test_sdk_adapter_is_opt_in_under_isaac_sim(order: str) -> None:
    result = _probe(order)
    assert result["after"]["isaac_sim"] == "apyrobo.core.isaac_adapter"


def test_plain_import_does_not_register_isaac_sim() -> None:
    code = (
        "import json, apyrobo\n"
        "from apyrobo.core.adapters import _ADAPTER_REGISTRY as R\n"
        "print(json.dumps('isaac_sim' in R))\n"
    )
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        capture_output=True, text=True, check=True, timeout=120,
    ).stdout
    assert json.loads(out.strip().splitlines()[-1]) is False

"""The adapter scaffold and the authoring guide must not turn a bad speed into full speed.

``min(speed or cap, cap)`` drives at the cap for ``speed=0`` (0 is falsy),
passes a negative speed through unclamped, and returns ``nan`` for NaN.
Both ``apyrobo init <scheme> --adapter`` and docs/adapter_authoring.md used it.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import re
import sys
from pathlib import Path

import pytest

from apyrobo.cli import cmd_init
from apyrobo.core.adapters import _ADAPTER_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
SCHEME = "speedcheck"


@pytest.fixture
def adapter(tmp_path):
    out = tmp_path / SCHEME
    cmd_init(argparse.Namespace(
        name=SCHEME, adapter=True, description="", author="",
        directory=str(out), force=False,
    ))
    module_name = f"apyrobo_adapter_{SCHEME}"
    sys.path.insert(0, str(out / "src"))
    try:
        module = importlib.import_module(module_name)
        yield module.SpeedcheckAdapter("bot", max_speed=1.0)
    finally:
        sys.modules.pop(module_name, None)
        sys.modules.pop(module_name + ".adapter", None)
        _ADAPTER_REGISTRY.pop(SCHEME, None)
        sys.path[:] = [p for p in sys.path if str(tmp_path) not in p]


def _logged_speed(caplog) -> float:
    for record in reversed(caplog.records):
        match = re.search(r"at (\S+) m/s", record.getMessage())
        if match:
            return float(match.group(1))
    raise AssertionError("move() logged no speed")


@pytest.mark.parametrize("bad", [0, 0.0, -0.5, -5.0, float("nan"), float("inf"), float("-inf")])
def test_scaffold_refuses_a_speed_that_is_not_positive_and_finite(adapter, bad):
    with pytest.raises(ValueError, match="positive number of m/s"):
        adapter.move(1.0, 1.0, speed=bad)
    assert adapter.get_position() == (0.0, 0.0)


@pytest.mark.parametrize("speed, expected", [(None, 1.0), (0.3, 0.3), (1.0, 1.0), (5.0, 1.0)])
def test_scaffold_drives_at_the_asked_speed_capped(adapter, caplog, speed, expected):
    with caplog.at_level(logging.INFO):
        adapter.move(2.0, 3.0, speed=speed)
    assert _logged_speed(caplog) == pytest.approx(expected)
    assert adapter.get_position() == (2.0, 3.0)


def test_guide_move_example_refuses_bad_speeds():
    text = (ROOT / "docs" / "adapter_authoring.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    move_block = next(b for b in blocks if b.startswith("def move("))
    assert "speed or self._max_speed" not in move_block

    import math

    sent: list[float] = []

    class Client:
        def goto(self, x, y, speed):
            sent.append(speed)

    namespace: dict = {"math": math}
    exec(compile(move_block, "adapter_authoring.md", "exec"), namespace)

    class Adapter:
        _max_speed = 1.5
        _client = Client()

        def _require_connected(self, op):
            pass

        move = namespace["move"]

    a = Adapter()
    for bad in (0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            a.move(1.0, 1.0, speed=bad)
    a.move(1.0, 1.0)
    a.move(1.0, 1.0, speed=0.4)
    a.move(1.0, 1.0, speed=9.0)
    assert sent == [1.5, 0.4, 1.5]

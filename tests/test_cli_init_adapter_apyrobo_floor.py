"""`apyrobo init <scheme> --adapter` must not declare an apyrobo floor its own
tests cannot run on.

The generated tests/test_adapter.py imports ``apyrobo.conformance`` and the
generated CI and next steps run ``apyrobo conformance``. Neither ships in
apyrobo 4.0.0, yet the scaffold declared ``apyrobo>=4.0.0``: in a fresh
environment pip resolved 4.0.0 and pytest stopped at collection with
``ModuleNotFoundError: No module named 'apyrobo.conformance'``. The floor is
now the release that wrote the scaffold.
"""
from __future__ import annotations

import argparse
import json
import re

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

import apyrobo.cli as cli
from apyrobo.__version__ import __version__


def _ns(name: str, out: str) -> argparse.Namespace:
    return argparse.Namespace(
        name=name, adapter=True, description="", author="",
        directory=out, force=False,
    )


def _apyrobo_requirement(out) -> Requirement:
    # tomllib is 3.11+ and CI also runs 3.10: read the one dependencies line.
    text = (out / "pyproject.toml").read_text()
    (line,) = re.findall(r"^dependencies = (\[.*\])$", text, re.M)
    reqs = [Requirement(d) for d in json.loads(line)]
    (req,) = [r for r in reqs if r.name == "apyrobo"]
    return req


def test_floor_is_the_release_that_wrote_the_scaffold(tmp_path):
    out = tmp_path / "floorbot"
    cli.cmd_init(_ns("floorbot", str(out)))
    req = _apyrobo_requirement(out)
    assert str(req.specifier) == f">={Version(__version__).base_version}"
    # The running apyrobo satisfies it, so `pip install -e .` next to it works.
    assert req.specifier.contains(Version(__version__).base_version)


def test_floor_excludes_4_0_0_which_has_no_conformance(tmp_path):
    out = tmp_path / "floorbot"
    cli.cmd_init(_ns("floorbot", str(out)))
    # The scaffold's own tests need apyrobo.conformance.
    assert "from apyrobo.conformance import" in (
        out / "tests" / "test_adapter.py"
    ).read_text()
    assert not _apyrobo_requirement(out).specifier.contains("4.0.0")


@pytest.mark.parametrize(
    ("running", "floor"),
    [("4.2.0", "4.2.0"), ("4.2.0rc1", "4.2.0"), ("4.3.0.dev2", "4.3.0"),
     ("4.2.0+local", "4.2.0")],
)
def test_floor_drops_pre_dev_and_local_parts(monkeypatch, tmp_path, running, floor):
    monkeypatch.setattr(cli, "_apyrobo_version", lambda: running)
    out = tmp_path / "floorbot"
    cli.cmd_init(_ns("floorbot", str(out)))
    assert str(_apyrobo_requirement(out).specifier) == f">={floor}"

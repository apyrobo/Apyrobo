"""``apyrobo mcp config`` refuses a blank ``--name`` or one with outer spaces.

``apyrobo mcp config claude-code --name ""`` used to print the default name
``apyrobo`` as if ``--name`` had been left out, and ``--name "  "`` printed a
server named by two spaces.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys

import pytest

from apyrobo import mcp_clients


def _config(client: str = "claude-code", **kw) -> tuple[int, str, str]:
    ns = argparse.Namespace(client=client, launcher="apyrobo", name="apyrobo",
                            cfg_ssh=None, cfg_ssh_setup=None)
    for key, value in kw.items():
        setattr(ns, key, value)
    out, err = io.StringIO(), io.StringIO()
    code = mcp_clients.main(ns, stdout=out, stderr=err,
                            which=lambda name: f"/usr/bin/{name}")
    return code, out.getvalue(), err.getvalue()


@pytest.mark.parametrize("value", ["", "  ", "\t"])
def test_blank_name_is_refused(value):
    code, out, err = _config(name=value)
    assert code == 1
    assert out == ""
    assert "--name is empty" in err
    assert "leave --name out" in err


@pytest.mark.parametrize("value", [" robot", "robot ", " robot "])
def test_name_with_outer_spaces_is_refused(value):
    code, out, err = _config(name=value)
    assert code == 1
    assert out == ""
    assert "starts or ends with a space" in err
    assert "--name 'robot'" in err


def test_missing_name_uses_the_default():
    code, out, _ = _config(client="claude-desktop", name=None)
    assert code == 0
    assert list(json.loads(out)["mcpServers"]) == ["apyrobo"]


def test_inner_space_is_kept():
    code, out, _ = _config(name="my robot")
    assert code == 0
    assert "mcp add 'my robot' --" in out


def test_cli_refuses_empty_name():
    proc = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "mcp", "config", "claude-desktop",
         "--launcher", "apyrobo", "--name", ""],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert "--name is empty" in proc.stderr

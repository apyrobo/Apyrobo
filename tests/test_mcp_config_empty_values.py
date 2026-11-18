"""``apyrobo mcp config`` refuses an empty ``--robot``, ``--policy-file``,
``--audit-log`` or ``--provider``.

``apyrobo mcp config claude-code --policy-file "$POLICY"`` with ``POLICY``
unset used to print a config without the policy file, so the server started
with the default limits and none of the file's zones, and nothing said so.
The same held for an empty ``--audit-log`` (no log), ``--robot`` (mock://)
and ``--provider`` (the rule planner).
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys

import pytest

from apyrobo import mcp_clients

FLAGS = {
    "--robot": "robot",
    "--policy-file": "policy_file",
    "--audit-log": "audit_log",
    "--provider": "provider",
}


def _config(**kw) -> tuple[int, str, str]:
    ns = argparse.Namespace(client="claude-code", launcher="apyrobo", name=None,
                            cfg_ssh=None, cfg_ssh_setup=None)
    for key, value in kw.items():
        setattr(ns, key, value)
    out, err = io.StringIO(), io.StringIO()
    code = mcp_clients.main(ns, stdout=out, stderr=err,
                            which=lambda name: f"/usr/bin/{name}")
    return code, out.getvalue(), err.getvalue()


@pytest.mark.parametrize("flag", sorted(FLAGS))
@pytest.mark.parametrize("value", ["", "  "])
@pytest.mark.parametrize("side", ["cfg_", ""])
def test_empty_value_is_refused(flag, value, side):
    code, out, err = _config(**{side + FLAGS[flag]: value})
    assert code == 1
    assert out == ""
    assert f"{flag} is empty" in err
    assert f"leave {flag} out" in err


@pytest.mark.parametrize("flag", sorted(FLAGS))
def test_empty_value_is_refused_with_ssh(flag):
    code, out, err = _config(cfg_ssh="pi@robot", **{"cfg_" + FLAGS[flag]: ""})
    assert code == 1
    assert out == ""
    assert f"{flag} is empty" in err


def test_given_values_still_pass(tmp_path):
    policy = tmp_path / "p.json"
    policy.write_text('{"max_speed": 0.5}')
    log = tmp_path / "audit.jsonl"
    code, out, err = _config(cfg_robot="mock://bot", cfg_policy_file=str(policy),
                             cfg_audit_log=str(log), cfg_provider="rule")
    assert code == 0, err
    assert "--robot mock://bot" in out
    assert f"--policy-file {policy}" in out
    assert f"--audit-log {log}" in out


def test_flags_left_out_still_pass():
    code, out, err = _config()
    assert code == 0, err
    assert "--policy-file" not in out
    assert "--audit-log" not in out


def test_cli_refuses_empty_policy_file():
    proc = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "mcp", "--policy-file", "",
         "config", "cursor"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert "--policy-file is empty" in proc.stderr

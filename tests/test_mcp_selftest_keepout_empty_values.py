"""``apyrobo mcp selftest`` and ``apyrobo mcp keepout`` refuse empty values.

``apyrobo mcp selftest --audit-log "$LOG"`` with ``LOG`` unset used to pass
3/3 checks with no audit-log check at all, and an empty ``--provider`` was
checked as the rule planner. ``apyrobo mcp keepout --out "$DIR"`` wrote the
mask into the current folder, and an empty ``--policy-file`` drew a mask
without the file's zones. Each now stops with the flag named.
"""

from __future__ import annotations

import io
import subprocess
import sys
from argparse import Namespace

import pytest

from apyrobo import mcp_keepout, mcp_selftest

SELFTEST_FLAGS = {
    "--robot": "robot",
    "--policy-file": "policy_file",
    "--audit-log": "audit_log",
    "--provider": "provider",
}


def _selftest_args(**kw) -> Namespace:
    base = dict(robot="mock://turtlebot4", policy=None, policy_file=None, max_speed=None,
                no_go=None, allow_area=None, confirm_moves=False, audit_log=None,
                provider="rule", st_robot=None, st_policy=None, st_policy_file=None,
                st_max_speed=None, st_no_go=None, st_allow_area=None,
                st_confirm_moves=False, st_audit_log=None, st_provider=None)
    base.update(kw)
    return Namespace(**base)


def _selftest(**kw) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = mcp_selftest.main(_selftest_args(**kw), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


@pytest.mark.parametrize("flag", sorted(SELFTEST_FLAGS))
@pytest.mark.parametrize("value", ["", "  "])
@pytest.mark.parametrize("side", ["st_", ""])
def test_selftest_refuses_an_empty_value(flag, value, side):
    code, out, err = _selftest(**{side + SELFTEST_FLAGS[flag]: value})
    assert code == 1
    assert out == ""
    assert f"{flag} is empty" in err
    assert f"leave {flag} out" in err


def test_selftest_value_after_selftest_wins_over_the_one_before(tmp_path):
    log = tmp_path / "audit.jsonl"
    code, out, err = _selftest(audit_log="", st_audit_log=str(log))
    assert code == 0, err
    assert "audit log" in out


def test_selftest_given_values_still_pass(tmp_path):
    policy = tmp_path / "p.json"
    policy.write_text('{"max_speed": 0.5}')
    log = tmp_path / "audit.jsonl"
    code, out, err = _selftest(st_policy_file=str(policy), st_audit_log=str(log),
                               st_provider="rule", st_robot="mock://bot")
    assert code == 0, err
    assert "4/4 checks passed" in out


def test_selftest_flags_left_out_still_pass():
    code, out, err = _selftest()
    assert code == 0, err
    assert "3/3 checks passed" in out


def _keepout_args(**kw) -> Namespace:
    base = dict(policy_file=None, no_go=None, ko_policy_file=None,
                ko_no_go=["0,0,1,1"], ko_resolution=0.05, ko_out=".")
    base.update(kw)
    return Namespace(**base)


@pytest.mark.parametrize("value", ["", "  "])
def test_keepout_refuses_an_empty_out(value, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert mcp_keepout.main(_keepout_args(ko_out=value)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--out is empty" in captured.err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", ["", "  "])
@pytest.mark.parametrize("key", ["ko_policy_file", "policy_file"])
def test_keepout_refuses_an_empty_policy_file(key, value, tmp_path, capsys):
    out_dir = tmp_path / "mask"
    assert mcp_keepout.main(_keepout_args(ko_out=str(out_dir), **{key: value})) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--policy-file is empty" in captured.err
    assert not out_dir.exists()


def test_keepout_given_values_still_write(tmp_path, capsys):
    out_dir = tmp_path / "mask"
    assert mcp_keepout.main(_keepout_args(ko_out=str(out_dir))) == 0
    assert (out_dir / "keepout_mask.yaml").exists()


def test_cli_selftest_refuses_empty_audit_log():
    proc = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "mcp", "selftest", "--audit-log", ""],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert "--audit-log is empty" in proc.stderr

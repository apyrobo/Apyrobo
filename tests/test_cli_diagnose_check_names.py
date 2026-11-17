"""apyrobo diagnose: environment checks carry stable names and fix hints,
and a failed robot connection is said on stderr, not only inside the JSON."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from apyrobo import cli as cli_mod
from apyrobo.cli import _CheckResult, cmd_diagnose, run_doctor_checks

_EXPECTED = {"python", "apyrobo", "rclpy", "mock_adapter", "llm", "docker", "skill_registry"}


def _report(capsys, **kwargs) -> tuple[dict, str]:
    args = SimpleNamespace(**{"robot": None, "out": "-", "timeout": 10.0, **kwargs})
    cmd_diagnose(args)
    captured = capsys.readouterr()
    return json.loads(captured.out), captured.err


def test_doctor_checks_have_stable_names():
    names = [r.name for r in run_doctor_checks()]
    assert set(names) >= _EXPECTED
    assert len(names) == len(set(names))


def test_diagnose_report_uses_check_names_not_first_word(capsys):
    report, _ = _report(capsys)
    names = {c["name"] for c in report["checks"]}
    assert names >= _EXPECTED
    # Before the fix the names were the first word of each message.
    assert not names & {"No", "Skill", "Mock", "Docker", "LLM", "Python"}


def test_diagnose_report_includes_hint_for_warnings(capsys):
    fake = [
        _CheckResult("warn", "Docker not responding", hint="Start Docker", name="docker"),
        _CheckResult("pass", "Python 3.12.0", hint="unused", name="python"),
    ]
    with patch.object(cli_mod, "run_doctor_checks", lambda: fake):
        report, _ = _report(capsys)
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["docker"]["hint"] == "Start Docker"
    assert "hint" not in by_name["python"]


def test_failed_connection_is_reported_on_stderr(capsys):
    def refused(uri, timeout):
        return None, 0.1, "connection refused"

    with patch.object(cli_mod, "_connect_with_timeout", refused):
        report, err = _report(capsys, robot="mock://gone")
    assert report["robot"]["connected"] is False
    assert "Could not connect to mock://gone: connection refused" in err


def test_bad_uri_is_reported_on_stderr(capsys):
    report, err = _report(capsys, robot="nope://")
    assert report["robot"]["connected"] is False
    assert "Could not connect to nope://" in err


def test_successful_connection_prints_nothing_on_stderr(capsys):
    report, err = _report(capsys, robot="mock://turtlebot4")
    assert report["robot"]["connected"] is True
    assert "Could not connect" not in err

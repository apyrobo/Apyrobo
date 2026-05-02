"""Tests for `apyrobo diagnose` CLI command."""
from __future__ import annotations

import json
import sys
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from apyrobo.cli import cmd_diagnose, _LogCapture, _collect_system_info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs) -> SimpleNamespace:
    defaults = {"robot": None, "out": "-", "timeout": 10.0}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _run_diagnose_to_str(**kwargs) -> dict:
    """Run cmd_diagnose with --out - and capture stdout as parsed JSON."""
    args = _make_args(**kwargs)
    buf = StringIO()
    with patch("sys.stdout", buf):
        cmd_diagnose(args)
    return json.loads(buf.getvalue())


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------

class TestDiagnoseReportStructure:
    def test_top_level_keys_present(self):
        report = _run_diagnose_to_str()
        for key in ("generated_at", "apyrobo_version", "system", "robot",
                    "health", "recent_tasks", "log_entries", "checks"):
            assert key in report, f"missing key: {key}"

    def test_system_info_keys(self):
        report = _run_diagnose_to_str()
        sys_info = report["system"]
        assert "python" in sys_info
        assert "os" in sys_info
        assert "ros_domain_id" in sys_info

    def test_version_string(self):
        report = _run_diagnose_to_str()
        assert isinstance(report["apyrobo_version"], str)

    def test_checks_is_list(self):
        report = _run_diagnose_to_str()
        assert isinstance(report["checks"], list)

    def test_robot_none_when_no_uri(self):
        report = _run_diagnose_to_str()
        assert report["robot"] is None

    def test_health_none_when_no_uri(self):
        report = _run_diagnose_to_str()
        assert report["health"] is None

    def test_recent_tasks_empty_when_no_uri(self):
        report = _run_diagnose_to_str()
        assert report["recent_tasks"] == []


# ---------------------------------------------------------------------------
# --out - (stdout)
# ---------------------------------------------------------------------------

class TestDiagnoseOutStdout:
    def test_out_dash_prints_json(self):
        report = _run_diagnose_to_str(out="-")
        assert "generated_at" in report

    def test_output_is_valid_json(self):
        args = _make_args(out="-")
        buf = StringIO()
        with patch("sys.stdout", buf):
            cmd_diagnose(args)
        # Should parse without error
        json.loads(buf.getvalue())


# ---------------------------------------------------------------------------
# --out <file>
# ---------------------------------------------------------------------------

class TestDiagnoseOutFile:
    def test_writes_json_to_file(self, tmp_path):
        out_file = tmp_path / "diag.json"
        args = _make_args(out=str(out_file))
        with patch("builtins.print"):  # suppress "written to" message
            cmd_diagnose(args)
        assert out_file.exists()
        report = json.loads(out_file.read_text())
        assert "generated_at" in report

    def test_prints_confirmation_message(self, tmp_path, capsys):
        out_file = tmp_path / "diag.json"
        args = _make_args(out=str(out_file))
        cmd_diagnose(args)
        captured = capsys.readouterr()
        assert str(out_file) in captured.out


# ---------------------------------------------------------------------------
# Robot connection
# ---------------------------------------------------------------------------

class TestDiagnoseRobotConnection:
    def test_robot_connected_true_on_success(self):
        report = _run_diagnose_to_str(robot="mock://turtlebot4")
        assert report["robot"] is not None
        assert report["robot"]["connected"] is True
        assert report["robot"]["uri"] == "mock://turtlebot4"

    def test_robot_checks_added_on_success(self):
        report = _run_diagnose_to_str(robot="mock://turtlebot4")
        names = [c["name"] for c in report["checks"]]
        assert "position" in names
        assert "battery" in names

    def test_robot_connected_false_on_failure(self):
        args = _make_args(robot="mock://nonexistent-never", out="-")
        # Patch _connect_with_timeout to simulate connection failure
        from apyrobo import cli as cli_mod
        def _fake_connect(uri, timeout):
            return None, 0.1, "connection refused"

        buf = StringIO()
        with patch.object(cli_mod, "_connect_with_timeout", _fake_connect):
            with patch("sys.stdout", buf):
                cmd_diagnose(args)
        report = json.loads(buf.getvalue())
        assert report["robot"]["connected"] is False
        assert "error" in report["robot"]

    def test_partial_report_even_when_robot_fails(self):
        args = _make_args(robot="mock://never", out="-")
        from apyrobo import cli as cli_mod
        buf = StringIO()
        with patch.object(cli_mod, "_connect_with_timeout", lambda *a: (None, 0.0, "err")):
            with patch("sys.stdout", buf):
                cmd_diagnose(args)
        report = json.loads(buf.getvalue())
        # These always-present keys must survive a failed robot connection
        assert "generated_at" in report
        assert "checks" in report
        assert isinstance(report["checks"], list)


# ---------------------------------------------------------------------------
# Log capture
# ---------------------------------------------------------------------------

class TestLogCapture:
    def test_log_capture_buffers_warnings(self):
        import logging
        lc = _LogCapture(maxlen=5)
        record = logging.LogRecord(
            name="test.logger", level=logging.WARNING,
            pathname="", lineno=0, msg="something bad", args=(), exc_info=None,
        )
        lc.emit(record)
        entries = lc.entries()
        assert len(entries) == 1
        assert entries[0]["level"] == "WARNING"
        assert entries[0]["logger"] == "test.logger"

    def test_log_capture_respects_maxlen(self):
        import logging
        lc = _LogCapture(maxlen=3)
        for i in range(5):
            record = logging.LogRecord(
                name="t", level=logging.WARNING,
                pathname="", lineno=0, msg=f"msg {i}", args=(), exc_info=None,
            )
            lc.emit(record)
        assert len(lc.entries()) == 3

    def test_log_entries_in_report(self, monkeypatch):
        import logging
        report = _run_diagnose_to_str()
        assert isinstance(report["log_entries"], list)

    def test_log_capture_has_timestamp(self):
        import logging
        lc = _LogCapture()
        record = logging.LogRecord(
            name="t", level=logging.ERROR,
            pathname="", lineno=0, msg="err", args=(), exc_info=None,
        )
        lc.emit(record)
        entry = lc.entries()[0]
        assert "timestamp" in entry
        assert "T" in entry["timestamp"]  # ISO-8601


# ---------------------------------------------------------------------------
# _collect_system_info
# ---------------------------------------------------------------------------

class TestCollectSystemInfo:
    def test_returns_dict_with_required_keys(self):
        info = _collect_system_info()
        assert "python" in info
        assert "os" in info
        assert "ros_domain_id" in info

    def test_python_version_format(self):
        info = _collect_system_info()
        parts = info["python"].split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

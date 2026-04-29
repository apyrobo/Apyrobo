"""Tests for `apyrobo doctor` / `apyrobo diagnose`."""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from unittest.mock import MagicMock

import pytest

import apyrobo.cli as _cli
from apyrobo.cli import (
    _CheckResult,
    _check_apyrobo_install,
    _check_docker,
    _check_llm_api_key,
    _check_mock_adapter,
    _check_python_version,
    _check_rclpy,
    _check_ros_domain_id,
    _check_skill_registry,
    cmd_doctor,
    run_doctor_checks,
)


def _args() -> argparse.Namespace:
    return argparse.Namespace()


# ---------------------------------------------------------------------------
# Python version check
# ---------------------------------------------------------------------------

class TestCheckPythonVersion:
    def test_modern_python_passes(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 12, 0, "final", 0))
        result = _check_python_version()
        assert result.status == "pass"
        assert "3.12.0" in result.message

    def test_old_python_fails(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 9, 7, "final", 0))
        result = _check_python_version()
        assert result.status == "fail"
        assert "3.9.7" in result.message
        assert result.hint is not None

    def test_minimum_boundary_passes(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 10, 0, "final", 0))
        result = _check_python_version()
        assert result.status == "pass"

    def test_below_minimum_fails(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 8, 0, "final", 0))
        result = _check_python_version()
        assert result.status == "fail"


# ---------------------------------------------------------------------------
# apyrobo install check
# ---------------------------------------------------------------------------

class TestCheckApyroboInstall:
    def test_passes_with_version(self):
        result = _check_apyrobo_install()
        assert result.status == "pass"
        assert "apyrobo" in result.message
        assert "1.0.0" in result.message


# ---------------------------------------------------------------------------
# rclpy check
# ---------------------------------------------------------------------------

class TestCheckRclpy:
    def test_missing_warns(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "rclpy", None)
        result, ok = _check_rclpy()
        assert result.status == "warn"
        assert ok is False
        assert result.hint is not None
        assert "Docker" in result.hint

    def test_present_passes(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "rclpy", MagicMock())
        result, ok = _check_rclpy()
        assert result.status == "pass"
        assert ok is True


# ---------------------------------------------------------------------------
# ROS_DOMAIN_ID check
# ---------------------------------------------------------------------------

class TestCheckRosDomainId:
    def test_set_passes(self, monkeypatch):
        monkeypatch.setenv("ROS_DOMAIN_ID", "42")
        result = _check_ros_domain_id()
        assert result.status == "pass"
        assert "42" in result.message

    def test_missing_warns(self, monkeypatch):
        monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
        result = _check_ros_domain_id()
        assert result.status == "warn"
        assert result.hint is not None


# ---------------------------------------------------------------------------
# Mock adapter check
# ---------------------------------------------------------------------------

class TestCheckMockAdapter:
    def test_passes(self):
        result = _check_mock_adapter()
        assert result.status == "pass"

    def test_exception_fails(self, monkeypatch):
        mock_robot_cls = MagicMock()
        mock_robot_cls.discover.side_effect = RuntimeError("adapter broken")
        monkeypatch.setattr(_cli, "Robot", mock_robot_cls)
        result = _check_mock_adapter()
        assert result.status == "fail"
        assert "adapter broken" in result.message
        assert result.hint is not None


# ---------------------------------------------------------------------------
# LLM API key check
# ---------------------------------------------------------------------------

class TestCheckLlmApiKey:
    def test_anthropic_key_passes(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        result = _check_llm_api_key()
        assert result.status == "pass"
        assert "ANTHROPIC_API_KEY" in result.message

    def test_openai_key_passes(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        result = _check_llm_api_key()
        assert result.status == "pass"

    def test_no_keys_warns(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        result = _check_llm_api_key()
        assert result.status == "warn"
        assert "ANTHROPIC_API_KEY" in result.message
        assert result.hint is not None


# ---------------------------------------------------------------------------
# Docker check
# ---------------------------------------------------------------------------

class TestCheckDocker:
    def test_available_passes(self, monkeypatch):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_proc)
        result = _check_docker()
        assert result.status == "pass"

    def test_not_found_warns(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()))
        result = _check_docker()
        assert result.status == "warn"
        assert "Docker" in result.hint

    def test_timeout_warns(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired("docker", 3)),
        )
        result = _check_docker()
        assert result.status == "warn"

    def test_nonzero_exit_warns(self, monkeypatch):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_proc)
        result = _check_docker()
        assert result.status == "warn"


# ---------------------------------------------------------------------------
# Skill registry check
# ---------------------------------------------------------------------------

class TestCheckSkillRegistry:
    def test_reachable_passes(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: MagicMock())
        result = _check_skill_registry()
        assert result.status == "pass"
        assert "localhost:8080" in result.message

    def test_unreachable_warns(self, monkeypatch):
        def _raise(*a, **kw):
            raise urllib.error.URLError("connection refused")
        monkeypatch.setattr(urllib.request, "urlopen", _raise)
        result = _check_skill_registry()
        assert result.status == "warn"
        assert "localhost:8080" in result.message
        assert result.hint is not None


# ---------------------------------------------------------------------------
# run_doctor_checks integration
# ---------------------------------------------------------------------------

class TestRunDoctorChecks:
    def test_returns_list_of_results(self):
        results = run_doctor_checks()
        assert isinstance(results, list)
        assert len(results) >= 7
        assert all(isinstance(r, _CheckResult) for r in results)

    def test_rclpy_missing_skips_domain_id_check(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "rclpy", None)
        results = run_doctor_checks()
        assert not any("ROS_DOMAIN_ID" in r.message for r in results)

    def test_rclpy_present_includes_domain_id_check(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "rclpy", MagicMock())
        monkeypatch.setenv("ROS_DOMAIN_ID", "10")
        results = run_doctor_checks()
        assert any("ROS_DOMAIN_ID" in r.message for r in results)


# ---------------------------------------------------------------------------
# cmd_doctor — output format and exit codes
# ---------------------------------------------------------------------------

class TestCmdDoctor:
    def test_all_pass_exits_0(self, monkeypatch, capsys):
        monkeypatch.setattr(
            _cli, "run_doctor_checks",
            lambda: [_CheckResult("pass", "thing ok")],
        )
        cmd_doctor(_args())
        out = capsys.readouterr().out
        assert "1 passed" in out
        assert "0 failures" in out

    def test_failure_exits_1(self, monkeypatch, capsys):
        monkeypatch.setattr(
            _cli, "run_doctor_checks",
            lambda: [_CheckResult("fail", "something broken", hint="fix it")],
        )
        with pytest.raises(SystemExit) as exc:
            cmd_doctor(_args())
        assert exc.value.code == 1

    def test_warnings_only_exits_0(self, monkeypatch, capsys):
        monkeypatch.setattr(
            _cli, "run_doctor_checks",
            lambda: [_CheckResult("warn", "something iffy", hint="maybe fix")],
        )
        cmd_doctor(_args())  # must not raise
        out = capsys.readouterr().out
        assert "0 failures" in out
        assert "1 warnings" in out

    def test_output_contains_header(self, monkeypatch, capsys):
        monkeypatch.setattr(_cli, "run_doctor_checks", lambda: [])
        cmd_doctor(_args())
        out = capsys.readouterr().out
        assert "apyrobo doctor" in out
        assert "─" in out

    def test_hint_shown_for_warn(self, monkeypatch, capsys):
        monkeypatch.setattr(
            _cli, "run_doctor_checks",
            lambda: [_CheckResult("warn", "missing thing", hint="do this to fix")],
        )
        cmd_doctor(_args())
        assert "do this to fix" in capsys.readouterr().out

    def test_hint_shown_for_fail(self, monkeypatch, capsys):
        monkeypatch.setattr(
            _cli, "run_doctor_checks",
            lambda: [_CheckResult("fail", "broken", hint="urgent fix")],
        )
        with pytest.raises(SystemExit):
            cmd_doctor(_args())
        assert "urgent fix" in capsys.readouterr().out

    def test_summary_counts_correctly(self, monkeypatch, capsys):
        monkeypatch.setattr(
            _cli, "run_doctor_checks",
            lambda: [
                _CheckResult("pass", "ok1"),
                _CheckResult("pass", "ok2"),
                _CheckResult("warn", "iffy"),
                _CheckResult("pass", "ok3"),
            ],
        )
        cmd_doctor(_args())
        out = capsys.readouterr().out
        assert "3 passed" in out
        assert "1 warnings" in out
        assert "0 failures" in out

    def test_python_too_old_fails(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "version_info", (3, 8, 0, "final", 0))
        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: (_ for _ in ()).throw(Exception()))
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()))
        with pytest.raises(SystemExit) as exc:
            cmd_doctor(_args())
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "3.8.0" in out

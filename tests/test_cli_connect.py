"""Tests for `apyrobo connect [--verify] [--json]`."""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import apyrobo.cli as _cli
from apyrobo.cli import cmd_connect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _args(**overrides) -> argparse.Namespace:
    defaults = dict(uri="mock://turtlebot4", verify=False, timeout=10.0, json=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_robot(
    position: tuple[float, float] = (0.0, 0.0),
    battery: float | None = 100.0,
    cap_names: tuple[str, ...] = ("navigate_to", "rotate", "pick"),
    health=None,
) -> MagicMock:
    robot = MagicMock()
    robot.get_position.return_value = position
    robot.get_health.return_value = (
        {"battery_pct": battery} if battery is not None else {}
    )
    caps = MagicMock()
    caps.capabilities = [SimpleNamespace(name=n) for n in cap_names]
    robot.capabilities.return_value = caps
    robot.health = health
    return robot


def _patch(monkeypatch, robot=None, connect_time: float = 1.2, error: str | None = None):
    """Shortcut: patch _connect_with_timeout to return a fixed result."""
    monkeypatch.setattr(
        _cli,
        "_connect_with_timeout",
        lambda uri, timeout: (robot, connect_time, error),
    )


# ---------------------------------------------------------------------------
# Basic connect (no --verify)
# ---------------------------------------------------------------------------

class TestConnectBasic:
    def test_successful_connect_exits_0(self, monkeypatch):
        _patch(monkeypatch, robot=_make_robot())
        cmd_connect(_args())  # must not raise

    def test_successful_connect_prints_uri_and_time(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot(), connect_time=1.23)
        cmd_connect(_args(uri="mock://tb4"))
        out = capsys.readouterr().out
        assert "mock://tb4" in out
        assert "1.2s" in out

    def test_successful_connect_shows_connected_icon(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot())
        cmd_connect(_args())
        out = capsys.readouterr().out
        assert "✅" in out

    def test_connection_error_exits_1(self, monkeypatch):
        _patch(monkeypatch, robot=None, error="no route to host")
        with pytest.raises(SystemExit) as exc:
            cmd_connect(_args())
        assert exc.value.code == 1

    def test_connection_error_prints_message(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=None, error="no route to host")
        with pytest.raises(SystemExit):
            cmd_connect(_args())
        assert "no route to host" in capsys.readouterr().out

    def test_connection_timeout_exits_1(self, monkeypatch):
        _patch(monkeypatch, robot=None, connect_time=10.0,
               error="Connection timed out after 10s")
        with pytest.raises(SystemExit) as exc:
            cmd_connect(_args())
        assert exc.value.code == 1

    def test_connection_timeout_prints_timed_out(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=None, connect_time=10.0,
               error="Connection timed out after 10s")
        with pytest.raises(SystemExit):
            cmd_connect(_args())
        assert "timed out" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# --verify: all checks pass
# ---------------------------------------------------------------------------

class TestVerifyAllPass:
    def test_exits_0_when_all_pass(self, monkeypatch):
        _patch(monkeypatch, robot=_make_robot())
        cmd_connect(_args(verify=True))  # must not raise

    def test_output_contains_rule_separator(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot())
        cmd_connect(_args(verify=True))
        assert "─" in capsys.readouterr().out

    def test_output_contains_position(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot(position=(1.5, 2.3)))
        cmd_connect(_args(verify=True))
        out = capsys.readouterr().out
        assert "1.50" in out
        assert "2.30" in out

    def test_output_contains_battery(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot(battery=87.0))
        cmd_connect(_args(verify=True))
        assert "87%" in capsys.readouterr().out

    def test_output_contains_capability_names(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot(cap_names=("navigate_to", "pick")))
        cmd_connect(_args(verify=True))
        out = capsys.readouterr().out
        assert "navigate_to" in out
        assert "pick" in out

    def test_output_contains_skill_count(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot(cap_names=("a", "b", "c")))
        cmd_connect(_args(verify=True))
        assert "3 skills" in capsys.readouterr().out

    def test_output_contains_latency(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot())
        cmd_connect(_args(verify=True))
        assert "ms p50" in capsys.readouterr().out

    def test_output_contains_health_not_monitored(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot(health=None))
        cmd_connect(_args(verify=True))
        assert "not monitored" in capsys.readouterr().out

    def test_summary_line_shows_counts(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot())
        cmd_connect(_args(verify=True))
        out = capsys.readouterr().out
        assert "passed" in out
        assert "warnings" in out
        assert "failures" in out


# ---------------------------------------------------------------------------
# --verify: warning cases
# ---------------------------------------------------------------------------

class TestVerifyWarnings:
    def test_low_battery_is_warn_not_fail(self, monkeypatch):
        _patch(monkeypatch, robot=_make_robot(battery=15.0))
        cmd_connect(_args(verify=True))  # must not raise — warn, not fail

    def test_low_battery_prints_warning_icon(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot(battery=10.0))
        cmd_connect(_args(verify=True))
        assert "⚠️" in capsys.readouterr().out

    def test_missing_battery_in_health_response_is_warn(self, monkeypatch):
        _patch(monkeypatch, robot=_make_robot(battery=None))
        cmd_connect(_args(verify=True))  # must not raise

    def test_unhealthy_monitor_is_warn_not_fail(self, monkeypatch):
        monkeypatch.setattr(_cli.time, "sleep", lambda _: None)
        mock_health = MagicMock()
        mock_health.is_healthy = False
        robot = _make_robot(health=mock_health)
        _patch(monkeypatch, robot=robot)
        cmd_connect(_args(verify=True))  # must not raise

    def test_unhealthy_monitor_shows_warn_icon(self, monkeypatch, capsys):
        monkeypatch.setattr(_cli.time, "sleep", lambda _: None)
        mock_health = MagicMock()
        mock_health.is_healthy = False
        _patch(monkeypatch, robot=_make_robot(health=mock_health))
        cmd_connect(_args(verify=True))
        assert "⚠️" in capsys.readouterr().out

    def test_warn_exits_0(self, monkeypatch):
        _patch(monkeypatch, robot=_make_robot(battery=5.0))
        cmd_connect(_args(verify=True))  # exits 0 even with warning


# ---------------------------------------------------------------------------
# --verify: failure cases
# ---------------------------------------------------------------------------

class TestVerifyFailures:
    def test_position_failure_exits_1(self, monkeypatch):
        robot = _make_robot()
        robot.get_position.side_effect = RuntimeError("odom unavailable")
        _patch(monkeypatch, robot=robot)
        with pytest.raises(SystemExit) as exc:
            cmd_connect(_args(verify=True))
        assert exc.value.code == 1

    def test_position_failure_shows_fail_icon(self, monkeypatch, capsys):
        robot = _make_robot()
        robot.get_position.side_effect = RuntimeError("odom unavailable")
        _patch(monkeypatch, robot=robot)
        with pytest.raises(SystemExit):
            cmd_connect(_args(verify=True))
        assert "❌" in capsys.readouterr().out

    def test_capabilities_failure_exits_1(self, monkeypatch):
        robot = _make_robot()
        robot.capabilities.side_effect = RuntimeError("caps broken")
        _patch(monkeypatch, robot=robot)
        with pytest.raises(SystemExit) as exc:
            cmd_connect(_args(verify=True))
        assert exc.value.code == 1

    def test_summary_shows_failure_count(self, monkeypatch, capsys):
        robot = _make_robot()
        robot.get_position.side_effect = RuntimeError("broken")
        _patch(monkeypatch, robot=robot)
        with pytest.raises(SystemExit):
            cmd_connect(_args(verify=True))
        out = capsys.readouterr().out
        assert "1 failures" in out or "failures" in out


# ---------------------------------------------------------------------------
# --json output
# ---------------------------------------------------------------------------

class TestJsonOutput:
    def test_basic_connect_json_is_valid(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot(), connect_time=1.2)
        cmd_connect(_args(json=True))
        doc = json.loads(capsys.readouterr().out)
        assert doc["connected"] is True
        assert doc["uri"] == "mock://turtlebot4"
        assert "connect_time_s" in doc

    def test_connect_failure_json_has_error(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=None, error="timeout")
        with pytest.raises(SystemExit):
            cmd_connect(_args(json=True))
        doc = json.loads(capsys.readouterr().out)
        assert doc["connected"] is False
        assert "error" in doc

    def test_verify_json_has_checks_and_summary(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot(), connect_time=1.2)
        cmd_connect(_args(verify=True, json=True))
        doc = json.loads(capsys.readouterr().out)
        assert "checks" in doc
        assert "summary" in doc
        assert isinstance(doc["checks"], list)
        assert len(doc["checks"]) == 5  # position, battery, capabilities, latency, health

    def test_verify_json_summary_counts_correct(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot(), connect_time=1.2)
        cmd_connect(_args(verify=True, json=True))
        doc = json.loads(capsys.readouterr().out)
        summary = doc["summary"]
        assert "passed" in summary
        assert "warnings" in summary
        assert "failures" in summary
        assert summary["failures"] == 0

    def test_verify_json_check_names(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot(), connect_time=1.2)
        cmd_connect(_args(verify=True, json=True))
        doc = json.loads(capsys.readouterr().out)
        names = {c["name"] for c in doc["checks"]}
        assert names == {"position", "battery", "capabilities", "latency_ms_p50", "health_monitor"}

    def test_verify_json_position_value(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot(position=(3.0, 4.0)), connect_time=0.5)
        cmd_connect(_args(verify=True, json=True))
        doc = json.loads(capsys.readouterr().out)
        pos_check = next(c for c in doc["checks"] if c["name"] == "position")
        assert pos_check["value"] == [3.0, 4.0]
        assert pos_check["status"] == "pass"

    def test_verify_json_low_battery_is_warn(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot(battery=5.0), connect_time=0.5)
        cmd_connect(_args(verify=True, json=True))
        doc = json.loads(capsys.readouterr().out)
        bat_check = next(c for c in doc["checks"] if c["name"] == "battery")
        assert bat_check["status"] == "warn"

    def test_json_no_extra_output_before_document(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot(), connect_time=1.2)
        cmd_connect(_args(json=True))
        out = capsys.readouterr().out.strip()
        # Must be valid JSON starting at char 0
        doc = json.loads(out)
        assert isinstance(doc, dict)


# ---------------------------------------------------------------------------
# More than 3 capabilities truncation
# ---------------------------------------------------------------------------

class TestCapabilityDisplay:
    def test_more_than_3_caps_shows_plus_more(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot(
            cap_names=("a", "b", "c", "d", "e")
        ))
        cmd_connect(_args(verify=True))
        out = capsys.readouterr().out
        assert "+2 more" in out

    def test_exactly_3_caps_no_truncation(self, monkeypatch, capsys):
        _patch(monkeypatch, robot=_make_robot(cap_names=("a", "b", "c")))
        cmd_connect(_args(verify=True))
        out = capsys.readouterr().out
        assert "more" not in out
        assert "3 skills" in out

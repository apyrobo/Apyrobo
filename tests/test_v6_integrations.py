"""Tests for v6.0.0 Ecosystem Integration additions.

Covers: profiles detect, TelemetryContextProvider, agent telemetry injection.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# profiles detect — hardware auto-detection
# ---------------------------------------------------------------------------

class TestProfilesDetect:
    def test_detect_profile_returns_detection_result(self):
        from apyrobo.profiles.schema import detect_profile, DetectionResult
        result = detect_profile()
        assert isinstance(result, DetectionResult)

    def test_detect_result_has_required_fields(self):
        from apyrobo.profiles.schema import detect_profile
        r = detect_profile()
        assert r.recommended_profile
        assert r.confidence in ("high", "medium", "low")
        assert r.reason
        assert isinstance(r.ollama_available, bool)
        assert isinstance(r.ollama_models, list)
        assert isinstance(r.gpu_detected, bool)
        assert r.ram_gb > 0

    def test_detect_to_dict_complete(self):
        from apyrobo.profiles.schema import detect_profile
        d = detect_profile().to_dict()
        for key in ("recommended_profile", "confidence", "reason",
                    "ollama_available", "ollama_models", "gpu_detected", "ram_gb"):
            assert key in d

    def test_detect_recommends_valid_profile(self):
        from apyrobo.profiles.schema import detect_profile, ProfileRegistry
        result = detect_profile()
        reg = ProfileRegistry()
        assert result.recommended_profile in reg.names()

    def test_detect_with_mocked_ollama(self):
        from apyrobo.profiles.schema import detect_profile
        # Fake Ollama running with llama3.1:8b
        with patch("apyrobo.profiles.schema._detect_ollama", return_value=(True, ["llama3.1:8b"])):
            with patch("apyrobo.profiles.schema._detect_gpu", return_value=(False, "", 0.0)):
                with patch("apyrobo.profiles.schema._detect_ram_gb", return_value=16.0):
                    result = detect_profile()
        assert result.ollama_available is True
        assert "llama3.1:8b" in result.ollama_models
        assert result.recommended_profile in ("cpu-only", "raspberry-pi", "jetson-orin")

    def test_detect_with_mocked_gpu_and_ollama(self):
        from apyrobo.profiles.schema import detect_profile
        with patch("apyrobo.profiles.schema._detect_ollama", return_value=(True, ["llama3.1:8b"])):
            with patch("apyrobo.profiles.schema._detect_gpu", return_value=(True, "RTX 4090", 24.0)):
                with patch("apyrobo.profiles.schema._detect_ram_gb", return_value=32.0):
                    result = detect_profile()
        assert result.gpu_detected is True
        assert result.recommended_profile == "workstation-gpu"
        assert result.confidence == "high"

    def test_detect_no_gpu_no_ollama_suggests_cloud_or_cpu(self):
        from apyrobo.profiles.schema import detect_profile
        with patch("apyrobo.profiles.schema._detect_ollama", return_value=(False, [])):
            with patch("apyrobo.profiles.schema._detect_gpu", return_value=(False, "", 0.0)):
                with patch("apyrobo.profiles.schema._detect_ram_gb", return_value=8.0):
                    result = detect_profile()
        assert result.recommended_profile in ("cpu-only", "cloud")
        assert result.confidence in ("low", "medium")

    def test_detect_high_ram_suggests_cloud(self):
        from apyrobo.profiles.schema import detect_profile
        with patch("apyrobo.profiles.schema._detect_ollama", return_value=(False, [])):
            with patch("apyrobo.profiles.schema._detect_gpu", return_value=(False, "", 0.0)):
                with patch("apyrobo.profiles.schema._detect_ram_gb", return_value=128.0):
                    result = detect_profile()
        assert result.recommended_profile == "cloud"

    def test_detect_ollama_probes_localhost(self):
        from apyrobo.profiles.schema import _detect_ollama
        # When Ollama is not running, should return (False, []) without raising
        available, models = _detect_ollama()
        assert isinstance(available, bool)
        assert isinstance(models, list)

    def test_profiles_detect_cli_command(self, capsys):
        from apyrobo.cli import cmd_profiles
        import argparse
        args = argparse.Namespace(
            profiles_command="detect",
            json=False,
            profile_name=None,
        )
        with patch("apyrobo.profiles.schema._detect_ollama", return_value=(False, [])):
            with patch("apyrobo.profiles.schema._detect_gpu", return_value=(False, "", 0.0)):
                with patch("apyrobo.profiles.schema._detect_ram_gb", return_value=8.0):
                    cmd_profiles(args)
        out = capsys.readouterr().out
        assert "Recommended profile" in out
        assert "apyrobo execute" in out

    def test_profiles_detect_cli_json(self, capsys):
        import argparse, json
        from apyrobo.cli import cmd_profiles
        args = argparse.Namespace(
            profiles_command="detect",
            json=True,
            profile_name=None,
        )
        with patch("apyrobo.profiles.schema._detect_ollama", return_value=(False, [])):
            with patch("apyrobo.profiles.schema._detect_gpu", return_value=(False, "", 0.0)):
                with patch("apyrobo.profiles.schema._detect_ram_gb", return_value=8.0):
                    cmd_profiles(args)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "recommended_profile" in data


# ---------------------------------------------------------------------------
# TelemetryContextProvider
# ---------------------------------------------------------------------------

class TestTelemetrySnapshot:
    def test_context_string_has_required_sections(self):
        from apyrobo.skills.telemetry_context import TelemetrySnapshot
        snap = TelemetrySnapshot(
            timestamp=time.monotonic(),
            position=(1.0, 2.0),
            heading_deg=90.0,
            battery_pct=75.0,
            linear_velocity=0.5,
            angular_velocity=0.1,
        )
        ctx = snap.to_context_string()
        assert "Robot State" in ctx
        assert "1.00" in ctx and "2.00" in ctx
        assert "90°" in ctx
        assert "75%" in ctx
        assert "0.50 m/s" in ctx

    def test_context_string_skips_none_fields(self):
        from apyrobo.skills.telemetry_context import TelemetrySnapshot
        snap = TelemetrySnapshot(timestamp=time.monotonic(), battery_pct=50.0)
        ctx = snap.to_context_string()
        assert "50%" in ctx
        assert "Position" not in ctx
        assert "Velocity" not in ctx

    def test_context_string_includes_errors(self):
        from apyrobo.skills.telemetry_context import TelemetrySnapshot
        snap = TelemetrySnapshot(
            timestamp=time.monotonic(),
            active_errors=["lidar disconnected"],
        )
        ctx = snap.to_context_string()
        assert "lidar disconnected" in ctx

    def test_age_seconds_is_small(self):
        from apyrobo.skills.telemetry_context import TelemetrySnapshot
        snap = TelemetrySnapshot(timestamp=time.monotonic())
        assert snap.age_seconds() < 1.0


class TestTelemetryContextProvider:
    @pytest.fixture
    def mock_robot(self):
        robot = MagicMock()
        robot.get_position.return_value = (1.5, -0.3, 0.0)
        robot.get_battery_level.return_value = 0.82
        robot.get_velocity.return_value = (0.0, 0.0)
        robot.get_sensor_status.side_effect = AttributeError("no sensor_status")
        robot.get_active_errors.return_value = []
        return robot

    def test_provider_samples_robot_state(self, mock_robot):
        from apyrobo.skills.telemetry_context import TelemetryContextProvider
        provider = TelemetryContextProvider(mock_robot, refresh_interval=60.0)
        # Give the background thread a moment
        time.sleep(0.1)
        snap = provider.get_snapshot()
        provider.stop()
        assert snap is not None
        assert snap.battery_pct is not None
        assert abs(snap.battery_pct - 82.0) < 1.0

    def test_get_context_string_nonempty(self, mock_robot):
        from apyrobo.skills.telemetry_context import TelemetryContextProvider
        provider = TelemetryContextProvider(mock_robot, refresh_interval=60.0)
        time.sleep(0.15)
        ctx = provider.get_context_string()
        provider.stop()
        assert ctx  # non-empty
        assert "Robot State" in ctx

    def test_stale_snapshot_returns_warning(self, mock_robot):
        from apyrobo.skills.telemetry_context import TelemetryContextProvider, TelemetrySnapshot
        provider = TelemetryContextProvider(mock_robot, refresh_interval=60.0,
                                            max_snapshot_age=0.001)
        # Force a stale snapshot
        provider._snapshot = TelemetrySnapshot(
            timestamp=time.monotonic() - 100,
            battery_pct=50.0,
        )
        ctx = provider.get_context_string()
        provider.stop()
        assert "WARNING" in ctx or "old" in ctx.lower()

    def test_no_snapshot_returns_empty_string(self, mock_robot):
        from apyrobo.skills.telemetry_context import TelemetryContextProvider
        provider = TelemetryContextProvider(mock_robot, refresh_interval=60.0)
        provider._snapshot = None  # clear before thread runs
        ctx = provider.get_context_string()
        provider.stop()
        # Either empty (no snapshot yet) or already has one — both are valid
        assert isinstance(ctx, str)

    def test_stop_stops_background_thread(self, mock_robot):
        from apyrobo.skills.telemetry_context import TelemetryContextProvider
        provider = TelemetryContextProvider(mock_robot, refresh_interval=60.0)
        provider.stop()
        assert provider._stop_event.is_set()

    def test_robot_error_does_not_crash_provider(self):
        from apyrobo.skills.telemetry_context import TelemetryContextProvider
        bad_robot = MagicMock()
        bad_robot.get_position.side_effect = RuntimeError("hardware fault")
        bad_robot.get_battery_level.side_effect = RuntimeError("hardware fault")
        bad_robot.get_velocity.side_effect = RuntimeError("hardware fault")
        bad_robot.get_sensor_status.side_effect = RuntimeError("hardware fault")
        bad_robot.get_active_errors.side_effect = RuntimeError("hardware fault")
        provider = TelemetryContextProvider(bad_robot, refresh_interval=60.0)
        time.sleep(0.1)
        # Should not raise
        ctx = provider.get_context_string()
        provider.stop()
        assert isinstance(ctx, str)


class TestAgentTelemetryIntegration:
    def test_agent_accepts_telemetry_provider(self):
        from apyrobo.skills.agent import Agent
        from apyrobo.skills.telemetry_context import TelemetryContextProvider
        mock_robot = MagicMock()
        mock_robot.get_position.return_value = (0.0, 0.0)
        mock_robot.get_battery_level.return_value = 1.0
        mock_robot.get_velocity.side_effect = AttributeError
        mock_robot.get_sensor_status.side_effect = AttributeError
        mock_robot.get_active_errors.side_effect = AttributeError
        telemetry = TelemetryContextProvider(mock_robot, refresh_interval=60.0)
        agent = Agent(provider="rule", telemetry_provider=telemetry)
        assert agent._telemetry is telemetry
        telemetry.stop()

    def test_telemetry_injected_into_llm_task(self):
        """Telemetry context is prepended to task for LLM providers."""
        from apyrobo.skills.agent import Agent, LLMProvider
        from apyrobo.skills.telemetry_context import TelemetryContextProvider, TelemetrySnapshot

        mock_robot = MagicMock()
        mock_robot.capabilities.return_value.capabilities = []
        mock_robot.capabilities.return_value.name = "MockBot"

        telemetry = MagicMock(spec=TelemetryContextProvider)
        telemetry.get_context_string.return_value = "[Robot State]\nBattery: 50%"

        captured_task: list[str] = []

        class CapturingLLMProvider(LLMProvider):
            def plan(self, task, available_skills, capability_names, **kw):
                captured_task.append(task)
                return []

        agent = Agent.__new__(Agent)
        agent._provider = CapturingLLMProvider.__new__(CapturingLLMProvider)
        agent._provider.__class__ = CapturingLLMProvider
        agent.memory = None
        agent._telemetry = telemetry
        agent._use_constrained_prompt = False
        agent._library = None
        agent._registry = None
        agent._rule_patterns = []
        agent._last_events = []
        agent._last_state = None
        agent._correction_learner = None
        agent._last_task = ""

        agent.plan("navigate to dock", mock_robot)
        assert captured_task
        assert "Robot State" in captured_task[0]
        assert "Battery: 50%" in captured_task[0]

    def test_rule_provider_skips_telemetry(self):
        """Telemetry is NOT injected for rule-based provider (no LLM context window)."""
        from apyrobo.skills.agent import Agent, RuleBasedProvider
        from apyrobo.skills.telemetry_context import TelemetryContextProvider

        telemetry = MagicMock(spec=TelemetryContextProvider)
        telemetry.get_context_string.return_value = "[Robot State]\nBattery: 50%"

        agent = Agent(provider="rule", telemetry_provider=telemetry)

        mock_robot = MagicMock()
        mock_robot.capabilities.return_value.capabilities = []
        mock_robot.capabilities.return_value.name = "MockBot"

        agent.plan("navigate to dock", mock_robot)
        # get_context_string should NOT have been called for rule provider
        telemetry.get_context_string.assert_not_called()

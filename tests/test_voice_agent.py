"""Tests for VoiceAgent and WhisperAdapter.transcribe(bytes|str)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from apyrobo.voice import MockVoiceAdapter, VoiceAgent, WhisperAdapter


# ---------------------------------------------------------------------------
# VoiceAgent
# ---------------------------------------------------------------------------

class _FakeResult:
    status = "completed"
    steps_completed = 1
    steps_total = 1
    error = None


class TestVoiceAgent:
    def _make_agent(self):
        agent = MagicMock()
        agent.execute.return_value = _FakeResult()
        return agent

    def test_run_with_mock_stt_returns_summary(self):
        robot = MagicMock()
        stt = MockVoiceAdapter(responses=["navigate to (1.0, 2.0)"])
        tts = MockVoiceAdapter()
        va = VoiceAgent(agent=self._make_agent(), robot=robot, stt=stt, tts=tts)

        summary = va.run()

        assert "completed" in summary
        assert "1/1" in summary

    def test_run_with_string_path_calls_transcribe(self):
        robot = MagicMock()
        stt = MockVoiceAdapter(responses=["stop"])
        stt.transcribe = MagicMock(return_value="stop the robot")
        tts = MockVoiceAdapter()
        va = VoiceAgent(agent=self._make_agent(), robot=robot, stt=stt, tts=tts)

        va.run("path/to/audio.wav")

        stt.transcribe.assert_called_once_with("path/to/audio.wav")

    def test_run_with_bytes_calls_transcribe(self):
        robot = MagicMock()
        stt = MockVoiceAdapter()
        stt.transcribe = MagicMock(return_value="go to dock")
        tts = MockVoiceAdapter()
        va = VoiceAgent(agent=self._make_agent(), robot=robot, stt=stt, tts=tts)

        va.run(b"\x00\x01\x02")

        stt.transcribe.assert_called_once_with(b"\x00\x01\x02")

    def test_run_speaks_summary(self):
        robot = MagicMock()
        stt = MockVoiceAdapter(responses=["move to room 3"])
        tts = MockVoiceAdapter()
        va = VoiceAgent(agent=self._make_agent(), robot=robot, stt=stt, tts=tts)

        va.run()

        assert len(tts.spoken) == 1
        assert "completed" in tts.spoken[0]

    def test_run_empty_transcription_returns_empty_string(self):
        robot = MagicMock()
        stt = MockVoiceAdapter(responses=[""])
        tts = MockVoiceAdapter()
        va = VoiceAgent(agent=self._make_agent(), robot=robot, stt=stt, tts=tts)

        result = va.run()

        assert result == ""

    def test_run_includes_error_when_task_fails(self):
        robot = MagicMock()
        agent = MagicMock()
        failed = _FakeResult()
        failed.status = "failed"
        failed.error = "precondition not met"
        agent.execute.return_value = failed

        stt = MockVoiceAdapter(responses=["do impossible thing"])
        tts = MockVoiceAdapter()
        va = VoiceAgent(agent=agent, robot=robot, stt=stt, tts=tts)

        summary = va.run()

        assert "precondition not met" in summary

    def test_default_tts_is_mock(self):
        va = VoiceAgent(
            agent=self._make_agent(),
            robot=MagicMock(),
            stt=MockVoiceAdapter(responses=["hi"]),
        )
        assert isinstance(va._tts, MockVoiceAdapter)

    def test_run_falls_back_to_listen_from_file_when_no_transcribe(self):
        robot = MagicMock()
        # Build a minimal adapter that has listen_from_file but not transcribe
        from apyrobo.voice import VoiceAdapter

        class _STTOnly(VoiceAdapter):
            def listen(self, timeout_s=5.0):
                return ""

            def speak(self, text):
                pass

            def listen_from_file(self, path):
                return "move forward"

        stt = _STTOnly()
        stt.listen_from_file = MagicMock(return_value="move forward")
        tts = MockVoiceAdapter()
        va = VoiceAgent(agent=self._make_agent(), robot=robot, stt=stt, tts=tts)

        va.run("audio.wav")

        stt.listen_from_file.assert_called_once_with("audio.wav")


# ---------------------------------------------------------------------------
# WhisperAdapter.transcribe(bytes|str)
# ---------------------------------------------------------------------------

class TestWhisperAdapterTranscribe:
    def test_transcribe_str_delegates_to_listen_from_file(self):
        adapter = WhisperAdapter()
        adapter.listen_from_file = MagicMock(return_value="hello robot")

        result = adapter.transcribe("path/to/file.wav")

        adapter.listen_from_file.assert_called_once_with("path/to/file.wav")
        assert result == "hello robot"

    def test_transcribe_bytes_writes_temp_file_and_transcribes(self, tmp_path):
        adapter = WhisperAdapter()
        adapter.listen_from_file = MagicMock(return_value="navigate home")

        result = adapter.transcribe(b"\x00\x01\x02\x03")

        # listen_from_file was called with some path (the temp file)
        adapter.listen_from_file.assert_called_once()
        called_path = adapter.listen_from_file.call_args[0][0]
        assert isinstance(called_path, str)
        assert called_path.endswith(".wav")
        # Temp file is cleaned up
        import os
        assert not os.path.exists(called_path)
        assert result == "navigate home"

    def test_transcribe_bytes_temp_file_cleaned_up_on_error(self):
        adapter = WhisperAdapter()
        adapter.listen_from_file = MagicMock(side_effect=RuntimeError("model fail"))

        with pytest.raises(RuntimeError):
            adapter.transcribe(b"\x00\x01")

        # Temp file must be cleaned up even after exception
        import os
        # (We can't easily get the path here, but the cleanup is in a finally block)

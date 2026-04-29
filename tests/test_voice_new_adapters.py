"""
Tests for new voice adapter classes:
WhisperSTTAdapter, OpenAISTTAdapter, PiperTTSAdapter, OpenAITTSAdapter,
MockSTTAdapter, MockTTSAdapter, VoiceAgent.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_whisper_mock(transcribed: str = "hello robot") -> MagicMock:
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": transcribed}
    mock_whisper = MagicMock()
    mock_whisper.load_model.return_value = mock_model
    return mock_whisper


# ---------------------------------------------------------------------------
# MockSTTAdapter
# ---------------------------------------------------------------------------

class TestMockSTTAdapter:
    def test_default_response(self):
        from apyrobo.voice import MockSTTAdapter
        a = MockSTTAdapter()
        assert a.transcribe(b"audio") == "mock transcription"

    def test_custom_responses(self):
        from apyrobo.voice import MockSTTAdapter
        a = MockSTTAdapter(responses=["hello", "world"])
        assert a.transcribe(b"") == "hello"
        assert a.transcribe(b"") == "world"

    def test_cycles_through_responses(self):
        from apyrobo.voice import MockSTTAdapter
        a = MockSTTAdapter(responses=["a", "b"])
        assert a.transcribe(b"") == "a"
        assert a.transcribe(b"") == "b"
        assert a.transcribe(b"") == "a"

    def test_accepts_file_path(self):
        from apyrobo.voice import MockSTTAdapter
        a = MockSTTAdapter(responses=["from file"])
        assert a.transcribe("/tmp/fake.wav") == "from file"

    def test_is_available(self):
        from apyrobo.voice import MockSTTAdapter
        assert MockSTTAdapter().is_available() is True


# ---------------------------------------------------------------------------
# MockTTSAdapter
# ---------------------------------------------------------------------------

class TestMockTTSAdapter:
    def test_synthesize_returns_bytes(self):
        from apyrobo.voice import MockTTSAdapter
        a = MockTTSAdapter()
        assert a.synthesize("hello") == b""

    def test_synthesize_records_calls(self):
        from apyrobo.voice import MockTTSAdapter
        a = MockTTSAdapter()
        a.synthesize("one")
        a.synthesize("two")
        assert a.synthesized == ["one", "two"]

    def test_is_available(self):
        from apyrobo.voice import MockTTSAdapter
        assert MockTTSAdapter().is_available() is True


# ---------------------------------------------------------------------------
# WhisperSTTAdapter
# ---------------------------------------------------------------------------

class TestWhisperSTTAdapter:
    def test_instantiation(self):
        from apyrobo.voice import WhisperSTTAdapter
        a = WhisperSTTAdapter(model_size="tiny")
        assert a._model_size == "tiny"

    def test_is_available_with_whisper(self):
        from apyrobo.voice import WhisperSTTAdapter
        mock_whisper = MagicMock()
        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            a = WhisperSTTAdapter()
            assert a.is_available() is True

    def test_is_available_without_whisper(self):
        from apyrobo.voice import WhisperSTTAdapter
        with patch.dict(sys.modules, {"whisper": None}):
            a = WhisperSTTAdapter()
            assert a.is_available() is False

    def test_transcribe_from_bytes(self, tmp_path):
        from apyrobo.voice import WhisperSTTAdapter
        mock_whisper = _make_whisper_mock("transcribed text")
        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            a = WhisperSTTAdapter()
            result = a.transcribe(b"\x00\x01\x02\x03")
        assert result == "transcribed text"

    def test_transcribe_from_file_path(self, tmp_path):
        from apyrobo.voice import WhisperSTTAdapter
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio")
        mock_whisper = _make_whisper_mock(" from file ")
        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            a = WhisperSTTAdapter()
            result = a.transcribe(str(audio_file))
        assert result == "from file"

    def test_model_loaded_lazily(self):
        from apyrobo.voice import WhisperSTTAdapter
        mock_whisper = _make_whisper_mock()
        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            a = WhisperSTTAdapter()
            assert a._model is None
            a.transcribe(b"\x00")
            assert a._model is not None

    def test_model_loaded_once(self):
        from apyrobo.voice import WhisperSTTAdapter
        mock_whisper = _make_whisper_mock()
        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            a = WhisperSTTAdapter()
            a.transcribe(b"\x00")
            a.transcribe(b"\x00")
        mock_whisper.load_model.assert_called_once()


# ---------------------------------------------------------------------------
# OpenAISTTAdapter
# ---------------------------------------------------------------------------

class TestOpenAISTTAdapter:
    def _mock_openai_transcription(self, text: str = "api result") -> MagicMock:
        mock_transcript = MagicMock()
        mock_transcript.text = text
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_transcript
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        return mock_openai

    def test_instantiation(self):
        from apyrobo.voice import OpenAISTTAdapter
        a = OpenAISTTAdapter(model="whisper-1", api_key="sk-test")
        assert a._model == "whisper-1"
        assert a._api_key == "sk-test"

    def test_is_available_with_openai(self):
        from apyrobo.voice import OpenAISTTAdapter
        mock_openai = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            a = OpenAISTTAdapter()
            assert a.is_available() is True

    def test_is_available_without_openai(self):
        from apyrobo.voice import OpenAISTTAdapter
        with patch.dict(sys.modules, {"openai": None}):
            a = OpenAISTTAdapter()
            assert a.is_available() is False

    def test_transcribe_from_bytes(self):
        from apyrobo.voice import OpenAISTTAdapter
        mock_openai = self._mock_openai_transcription("hello from api")
        with patch.dict(sys.modules, {"openai": mock_openai}):
            a = OpenAISTTAdapter()
            result = a.transcribe(b"\x00\x01\x02\x03")
        assert result == "hello from api"

    def test_transcribe_from_file_path(self, tmp_path):
        from apyrobo.voice import OpenAISTTAdapter
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio")
        mock_openai = self._mock_openai_transcription(" from file ")
        with patch.dict(sys.modules, {"openai": mock_openai}):
            a = OpenAISTTAdapter()
            result = a.transcribe(str(audio_file))
        assert result == "from file"

    def test_client_lazy_init(self):
        from apyrobo.voice import OpenAISTTAdapter
        mock_openai = self._mock_openai_transcription()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            a = OpenAISTTAdapter()
            assert a._client is None
            a.transcribe(b"audio")
            assert a._client is not None


# ---------------------------------------------------------------------------
# PiperTTSAdapter
# ---------------------------------------------------------------------------

class TestPiperTTSAdapter:
    def test_instantiation(self):
        from apyrobo.voice import PiperTTSAdapter
        a = PiperTTSAdapter(voice="en_US-lessac-medium")
        assert a._voice == "en_US-lessac-medium"

    def test_is_available_when_piper_on_path(self):
        from apyrobo.voice import PiperTTSAdapter
        with patch("shutil.which", return_value="/usr/bin/piper"):
            assert PiperTTSAdapter().is_available() is True

    def test_is_available_when_piper_missing(self):
        from apyrobo.voice import PiperTTSAdapter
        with patch("shutil.which", return_value=None):
            assert PiperTTSAdapter().is_available() is False

    def test_synthesize_returns_bytes_on_success(self):
        from apyrobo.voice import PiperTTSAdapter
        import subprocess
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"\x00\x01\x02\x03"
        with patch("subprocess.run", return_value=mock_result):
            a = PiperTTSAdapter()
            result = a.synthesize("hello world")
        assert result == b"\x00\x01\x02\x03"

    def test_synthesize_returns_empty_on_failure(self):
        from apyrobo.voice import PiperTTSAdapter
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = b""
        with patch("subprocess.run", return_value=mock_result):
            a = PiperTTSAdapter()
            result = a.synthesize("hello")
        assert result == b""

    def test_synthesize_returns_empty_when_not_found(self):
        from apyrobo.voice import PiperTTSAdapter
        with patch("subprocess.run", side_effect=FileNotFoundError):
            a = PiperTTSAdapter()
            result = a.synthesize("hello")
        assert result == b""

    def test_voice_model_passed_to_cmd(self):
        from apyrobo.voice import PiperTTSAdapter
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"audio"
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            a = PiperTTSAdapter(voice="en_US-lessac-medium")
            a.synthesize("test")
        cmd = mock_run.call_args[0][0]
        assert "--model" in cmd
        assert "en_US-lessac-medium" in cmd


# ---------------------------------------------------------------------------
# OpenAITTSAdapter
# ---------------------------------------------------------------------------

class TestOpenAITTSAdapter:
    def _mock_openai_tts(self, audio_bytes: bytes = b"mp3data") -> MagicMock:
        mock_response = MagicMock()
        mock_response.content = audio_bytes
        mock_client = MagicMock()
        mock_client.audio.speech.create.return_value = mock_response
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        return mock_openai

    def test_instantiation(self):
        from apyrobo.voice import OpenAITTSAdapter
        a = OpenAITTSAdapter(model="tts-1-hd", voice="nova")
        assert a._model == "tts-1-hd"
        assert a._voice == "nova"

    def test_is_available_with_openai(self):
        from apyrobo.voice import OpenAITTSAdapter
        mock_openai = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            assert OpenAITTSAdapter().is_available() is True

    def test_is_available_without_openai(self):
        from apyrobo.voice import OpenAITTSAdapter
        with patch.dict(sys.modules, {"openai": None}):
            assert OpenAITTSAdapter().is_available() is False

    def test_synthesize_returns_bytes(self):
        from apyrobo.voice import OpenAITTSAdapter
        mock_openai = self._mock_openai_tts(b"mp3audio")
        with patch.dict(sys.modules, {"openai": mock_openai}):
            a = OpenAITTSAdapter()
            result = a.synthesize("hello robot")
        assert result == b"mp3audio"

    def test_synthesize_passes_model_and_voice(self):
        from apyrobo.voice import OpenAITTSAdapter
        mock_openai = self._mock_openai_tts()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            a = OpenAITTSAdapter(model="tts-1-hd", voice="nova", api_key="sk-x")
            a.synthesize("test text")
        mock_client = mock_openai.OpenAI.return_value
        call_kwargs = mock_client.audio.speech.create.call_args[1]
        assert call_kwargs["model"] == "tts-1-hd"
        assert call_kwargs["voice"] == "nova"
        assert call_kwargs["input"] == "test text"


# ---------------------------------------------------------------------------
# VoiceAgent
# ---------------------------------------------------------------------------

class TestVoiceAgent:
    def _make_agent_and_robot(self):
        from apyrobo.core.robot import Robot
        from apyrobo.skills.agent import Agent

        robot = Robot.discover("mock://turtlebot4")
        agent = Agent(provider="rule")
        return agent, robot

    def test_run_returns_summary_string(self):
        from apyrobo.voice import VoiceAgent, MockSTTAdapter, MockTTSAdapter

        agent, robot = self._make_agent_and_robot()
        stt = MockSTTAdapter(responses=["go to room 3"])
        tts = MockTTSAdapter()
        va = VoiceAgent(agent=agent, robot=robot, stt=stt, tts=tts)
        result = va.run(b"audio bytes")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_run_calls_stt_with_audio(self):
        from apyrobo.voice import VoiceAgent, MockSTTAdapter, MockTTSAdapter

        agent, robot = self._make_agent_and_robot()
        tts = MockTTSAdapter()

        # Subclass to track the call
        class TrackingSTT(MockSTTAdapter):
            def __init__(self):
                super().__init__(responses=["navigate"])
                self.received = []
            def transcribe(self, audio):
                self.received.append(audio)
                return super().transcribe(audio)

        tracking = TrackingSTT()
        va = VoiceAgent(agent=agent, robot=robot, stt=tracking, tts=tts)
        va.run(b"\xff\xfe")
        assert b"\xff\xfe" in tracking.received

    def test_run_calls_tts_with_summary(self):
        from apyrobo.voice import VoiceAgent, MockSTTAdapter, MockTTSAdapter

        agent, robot = self._make_agent_and_robot()
        stt = MockSTTAdapter(responses=["go to (1, 2)"])
        tts = MockTTSAdapter()
        va = VoiceAgent(agent=agent, robot=robot, stt=stt, tts=tts)
        summary = va.run(b"audio")
        assert len(tts.synthesized) == 1
        assert tts.synthesized[0] == summary

    def test_run_empty_transcription_returns_empty(self):
        from apyrobo.voice import VoiceAgent, MockSTTAdapter, MockTTSAdapter

        agent, robot = self._make_agent_and_robot()
        stt = MockSTTAdapter(responses=[""])
        tts = MockTTSAdapter()
        va = VoiceAgent(agent=agent, robot=robot, stt=stt, tts=tts)
        result = va.run(b"audio")
        assert result == ""
        assert tts.synthesized == []

    def test_run_accepts_file_path(self, tmp_path):
        from apyrobo.voice import VoiceAgent, MockSTTAdapter, MockTTSAdapter

        audio_file = tmp_path / "cmd.wav"
        audio_file.write_bytes(b"fake audio")
        agent, robot = self._make_agent_and_robot()
        stt = MockSTTAdapter(responses=["patrol"])
        tts = MockTTSAdapter()
        va = VoiceAgent(agent=agent, robot=robot, stt=stt, tts=tts)
        result = va.run(str(audio_file))
        assert isinstance(result, str)

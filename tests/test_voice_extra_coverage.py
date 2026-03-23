"""
Extended voice.py coverage tests.

Targets previously-uncovered lines:
  71-74, 78-95, 99-103, 107-126, 204-210, 214-243, 247-253, 257-278, 281-285

Covers:
- WhisperAdapter._load_model()         lines 71-74
- WhisperAdapter.listen()              lines 78-95
- WhisperAdapter.listen_from_file()    lines 99-103
- WhisperAdapter.speak()               lines 107-126 (both paths)
- WhisperAdapter.is_available()        lines 128-133
- PiperAdapter.speak()                 lines 155-176 (success + fail)
- OpenAIVoiceAdapter._get_client()     lines 204-210
- OpenAIVoiceAdapter.listen()          lines 214-243
- OpenAIVoiceAdapter.listen_from_file() lines 247-253
- OpenAIVoiceAdapter.speak()           lines 257-278
- OpenAIVoiceAdapter.is_available()    lines 281-285
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from apyrobo.voice import (
    OpenAIVoiceAdapter,
    PiperAdapter,
    WhisperAdapter,
)


# ===========================================================================
# WhisperAdapter._load_model (lines 71-74)
# ===========================================================================

class TestWhisperAdapterLoadModel:
    """_load_model() lazily imports whisper and caches the model."""

    def _make_adapter(self, model_size: str = "base", device=None) -> WhisperAdapter:
        return WhisperAdapter(model_size=model_size, device=device)

    def test_load_model_calls_whisper_load(self) -> None:
        """_load_model calls whisper.load_model with the configured size."""
        adapter = self._make_adapter(model_size="tiny")
        mock_model = MagicMock()
        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = mock_model

        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            result = adapter._load_model()

        mock_whisper.load_model.assert_called_once_with("tiny", device=None)
        assert result is mock_model

    def test_load_model_with_device(self) -> None:
        adapter = self._make_adapter(model_size="small", device="cuda")
        mock_model = MagicMock()
        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = mock_model

        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            result = adapter._load_model()

        mock_whisper.load_model.assert_called_once_with("small", device="cuda")
        assert result is mock_model

    def test_load_model_caches_result(self) -> None:
        """Second call returns the cached model without re-importing."""
        adapter = self._make_adapter()
        mock_model = MagicMock()
        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = mock_model

        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            r1 = adapter._load_model()
            r2 = adapter._load_model()

        assert r1 is r2
        # whisper.load_model only called once
        assert mock_whisper.load_model.call_count == 1

    def test_load_model_uses_cached_if_already_set(self) -> None:
        """If _model is already set, load_model is never called."""
        adapter = self._make_adapter()
        pre_loaded = MagicMock()
        adapter._model = pre_loaded

        mock_whisper = MagicMock()
        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            result = adapter._load_model()

        mock_whisper.load_model.assert_not_called()
        assert result is pre_loaded


# ===========================================================================
# WhisperAdapter.listen (lines 78-95)
# ===========================================================================

class TestWhisperAdapterListen:
    """listen() records audio with sounddevice and transcribes with Whisper."""

    def _make_adapter(self) -> WhisperAdapter:
        return WhisperAdapter(model_size="base")

    def test_listen_records_and_transcribes(self) -> None:
        adapter = self._make_adapter()

        mock_sd = MagicMock()
        mock_np = MagicMock()
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": " hello world "}
        mock_whisper.load_model.return_value = mock_model

        # sd.rec returns a mock array; flatten() returns a flat array
        mock_audio = MagicMock()
        mock_audio.flatten.return_value = MagicMock()
        mock_sd.rec.return_value = mock_audio

        with patch.dict("sys.modules", {
            "whisper": mock_whisper,
            "sounddevice": mock_sd,
            "numpy": mock_np,
        }):
            text = adapter.listen(timeout_s=3.0)

        assert text == "hello world"
        mock_sd.rec.assert_called_once()
        mock_sd.wait.assert_called_once()
        mock_model.transcribe.assert_called_once()

    def test_listen_passes_fp16_false(self) -> None:
        """Whisper is called with fp16=False for CPU compatibility."""
        adapter = self._make_adapter()
        mock_sd = MagicMock()
        mock_np = MagicMock()
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "test"}
        mock_whisper.load_model.return_value = mock_model
        mock_audio = MagicMock()
        mock_audio.flatten.return_value = MagicMock()
        mock_sd.rec.return_value = mock_audio

        with patch.dict("sys.modules", {
            "whisper": mock_whisper,
            "sounddevice": mock_sd,
            "numpy": mock_np,
        }):
            adapter.listen()

        _, kwargs = mock_model.transcribe.call_args
        assert kwargs.get("fp16") is False

    def test_listen_uses_16000_sample_rate(self) -> None:
        adapter = self._make_adapter()
        mock_sd = MagicMock()
        mock_np = MagicMock()
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "ok"}
        mock_whisper.load_model.return_value = mock_model
        mock_audio = MagicMock()
        mock_audio.flatten.return_value = MagicMock()
        mock_sd.rec.return_value = mock_audio

        with patch.dict("sys.modules", {
            "whisper": mock_whisper,
            "sounddevice": mock_sd,
            "numpy": mock_np,
        }):
            adapter.listen(timeout_s=2.0)

        call_kwargs = mock_sd.rec.call_args
        # second positional/keyword arg is samplerate
        assert call_kwargs[1].get("samplerate") == 16000 or \
               call_kwargs[0][1] == 16000  # handle positional

    def test_listen_returns_stripped_text(self) -> None:
        adapter = self._make_adapter()
        mock_sd = MagicMock()
        mock_np = MagicMock()
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "   spaces everywhere   "}
        mock_whisper.load_model.return_value = mock_model
        mock_audio = MagicMock()
        mock_audio.flatten.return_value = MagicMock()
        mock_sd.rec.return_value = mock_audio

        with patch.dict("sys.modules", {
            "whisper": mock_whisper,
            "sounddevice": mock_sd,
            "numpy": mock_np,
        }):
            text = adapter.listen()

        assert text == "spaces everywhere"

    def test_listen_empty_result(self) -> None:
        adapter = self._make_adapter()
        mock_sd = MagicMock()
        mock_np = MagicMock()
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {}  # no "text" key
        mock_whisper.load_model.return_value = mock_model
        mock_audio = MagicMock()
        mock_audio.flatten.return_value = MagicMock()
        mock_sd.rec.return_value = mock_audio

        with patch.dict("sys.modules", {
            "whisper": mock_whisper,
            "sounddevice": mock_sd,
            "numpy": mock_np,
        }):
            text = adapter.listen()

        assert text == ""


# ===========================================================================
# WhisperAdapter.listen_from_file (lines 99-103)
# ===========================================================================

class TestWhisperAdapterListenFromFile:
    def test_transcribes_file(self, tmp_path) -> None:
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio")

        adapter = WhisperAdapter()
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": " from file "}
        mock_whisper.load_model.return_value = mock_model

        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            text = adapter.listen_from_file(str(audio_file))

        assert text == "from file"
        mock_model.transcribe.assert_called_once_with(str(audio_file), fp16=False)

    def test_listen_from_file_strips_result(self, tmp_path) -> None:
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"x")

        adapter = WhisperAdapter()
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  trimmed  "}
        mock_whisper.load_model.return_value = mock_model

        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            text = adapter.listen_from_file(str(f))

        assert text == "trimmed"

    def test_listen_from_file_missing_text_key(self, tmp_path) -> None:
        f = tmp_path / "audio.wav"
        f.write_bytes(b"x")

        adapter = WhisperAdapter()
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {}
        mock_whisper.load_model.return_value = mock_model

        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            text = adapter.listen_from_file(str(f))

        assert text == ""


# ===========================================================================
# WhisperAdapter.speak (lines 107-126)
# ===========================================================================

class TestWhisperAdapterSpeak:
    """speak() pipes text through piper then aplay."""

    def test_speak_success_path(self) -> None:
        """piper returns 0 with stdout → aplay is called."""
        import subprocess as _real_subprocess
        adapter = WhisperAdapter()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = b"raw audio data"

        mock_subprocess = MagicMock()
        mock_subprocess.run.return_value = mock_proc
        with patch.dict("sys.modules", {"subprocess": mock_subprocess}):
            adapter.speak("hello robot")

        # Two subprocess.run calls: piper then aplay
        assert mock_subprocess.run.call_count == 2
        first_call = mock_subprocess.run.call_args_list[0]
        assert first_call[0][0][0] == "piper"

    def test_speak_piper_nonzero_logs_warning(self, caplog) -> None:
        """When piper returns non-zero, aplay is not called and warning is logged."""
        import logging
        adapter = WhisperAdapter()
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = b""

        mock_subprocess = MagicMock()
        mock_subprocess.run.return_value = mock_proc
        with patch.dict("sys.modules", {"subprocess": mock_subprocess}):
            with caplog.at_level(logging.WARNING):
                adapter.speak("test")

        # Only one call (piper) — aplay not triggered
        assert mock_subprocess.run.call_count == 1

    def test_speak_piper_empty_stdout_no_aplay(self) -> None:
        """piper returncode=0 but empty stdout → aplay not called."""
        adapter = WhisperAdapter()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = b""  # empty

        mock_subprocess = MagicMock()
        mock_subprocess.run.return_value = mock_proc
        with patch.dict("sys.modules", {"subprocess": mock_subprocess}):
            adapter.speak("silent")

        assert mock_subprocess.run.call_count == 1

    def test_speak_file_not_found_raises(self) -> None:
        """FileNotFoundError (piper not installed) is re-raised."""
        adapter = WhisperAdapter()
        mock_subprocess = MagicMock()
        mock_subprocess.run.side_effect = FileNotFoundError("piper not found")
        with patch.dict("sys.modules", {"subprocess": mock_subprocess}):
            with pytest.raises(FileNotFoundError):
                adapter.speak("crash")

    def test_speak_passes_text_as_input(self) -> None:
        """Text is encoded and passed as stdin input to piper."""
        adapter = WhisperAdapter()
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = b""

        mock_subprocess = MagicMock()
        mock_subprocess.run.return_value = mock_proc
        with patch.dict("sys.modules", {"subprocess": mock_subprocess}):
            adapter.speak("hello world")

        first_call = mock_subprocess.run.call_args_list[0]
        assert first_call[1]["input"] == b"hello world"


# ===========================================================================
# WhisperAdapter.is_available (lines 128-133)
# ===========================================================================

class TestWhisperAdapterIsAvailable:
    def test_is_available_true_when_whisper_importable(self) -> None:
        mock_whisper = MagicMock()
        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            adapter = WhisperAdapter()
            assert adapter.is_available() is True

    def test_is_available_false_when_whisper_not_importable(self) -> None:
        """ImportError in whisper import → returns False."""
        adapter = WhisperAdapter()
        # Remove whisper from sys.modules and make import fail
        original = sys.modules.pop("whisper", None)
        try:
            with patch("builtins.__import__", side_effect=lambda name, *a, **kw: (
                (_ for _ in ()).throw(ImportError("no whisper")) if name == "whisper"
                else __import__(name, *a, **kw)
            )):
                result = adapter.is_available()
            assert result is False
        finally:
            if original is not None:
                sys.modules["whisper"] = original


# ===========================================================================
# PiperAdapter.speak (lines 155-176)
# ===========================================================================

class TestPiperAdapterSpeak:
    def test_speak_success_calls_aplay(self) -> None:
        """piper success → aplay is called with stdout."""
        adapter = PiperAdapter()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = b"audio bytes"

        mock_subprocess = MagicMock()
        mock_subprocess.run.return_value = mock_proc
        with patch.dict("sys.modules", {"subprocess": mock_subprocess}):
            adapter.speak("test text")

        assert mock_subprocess.run.call_count == 2
        piper_call = mock_subprocess.run.call_args_list[0]
        assert "piper" in piper_call[0][0]

    def test_speak_with_voice_model(self) -> None:
        """When voice is set, --model is added to piper command."""
        adapter = PiperAdapter(voice="en_US-lessac-medium")
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = b"audio"

        mock_subprocess = MagicMock()
        mock_subprocess.run.return_value = mock_proc
        with patch.dict("sys.modules", {"subprocess": mock_subprocess}):
            adapter.speak("hello")

        piper_args = mock_subprocess.run.call_args_list[0][0][0]
        assert "--model" in piper_args
        assert "en_US-lessac-medium" in piper_args

    def test_speak_without_voice_no_model_flag(self) -> None:
        """When voice is None, --model is NOT in piper command."""
        adapter = PiperAdapter(voice=None)
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = b"data"

        mock_subprocess = MagicMock()
        mock_subprocess.run.return_value = mock_proc
        with patch.dict("sys.modules", {"subprocess": mock_subprocess}):
            adapter.speak("hi")

        piper_args = mock_subprocess.run.call_args_list[0][0][0]
        assert "--model" not in piper_args

    def test_speak_failure_returncode_logs_warning(self, caplog) -> None:
        """Non-zero returncode from piper logs a warning; aplay not called."""
        import logging
        adapter = PiperAdapter()
        mock_proc = MagicMock()
        mock_proc.returncode = 2
        mock_proc.stdout = b""

        mock_subprocess = MagicMock()
        mock_subprocess.run.return_value = mock_proc
        with patch.dict("sys.modules", {"subprocess": mock_subprocess}):
            with caplog.at_level(logging.WARNING):
                adapter.speak("fail")

        assert mock_subprocess.run.call_count == 1

    def test_speak_empty_stdout_no_aplay(self) -> None:
        """returncode=0 but empty stdout → aplay not called."""
        adapter = PiperAdapter()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = b""

        mock_subprocess = MagicMock()
        mock_subprocess.run.return_value = mock_proc
        with patch.dict("sys.modules", {"subprocess": mock_subprocess}):
            adapter.speak("nothing")

        assert mock_subprocess.run.call_count == 1

    def test_speak_text_encoded_as_bytes(self) -> None:
        adapter = PiperAdapter()
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = b""

        mock_subprocess = MagicMock()
        mock_subprocess.run.return_value = mock_proc
        with patch.dict("sys.modules", {"subprocess": mock_subprocess}):
            adapter.speak("encoded test")

        first_call = mock_subprocess.run.call_args_list[0]
        assert first_call[1]["input"] == b"encoded test"


# ===========================================================================
# OpenAIVoiceAdapter._get_client (lines 204-210)
# ===========================================================================

class TestOpenAIVoiceAdapterGetClient:
    def test_get_client_creates_openai_client(self) -> None:
        adapter = OpenAIVoiceAdapter()
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = adapter._get_client()

        assert client is mock_client
        mock_openai.OpenAI.assert_called_once_with()

    def test_get_client_passes_api_key(self) -> None:
        adapter = OpenAIVoiceAdapter(api_key="sk-test-key")
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            adapter._get_client()

        mock_openai.OpenAI.assert_called_once_with(api_key="sk-test-key")

    def test_get_client_caches_result(self) -> None:
        """Second call returns the same client object."""
        adapter = OpenAIVoiceAdapter()
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            c1 = adapter._get_client()
            c2 = adapter._get_client()

        assert c1 is c2
        assert mock_openai.OpenAI.call_count == 1

    def test_get_client_no_api_key_no_kwarg(self) -> None:
        """Without api_key, OpenAI() is called without api_key kwarg."""
        adapter = OpenAIVoiceAdapter(api_key=None)
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            adapter._get_client()

        _, kwargs = mock_openai.OpenAI.call_args
        assert "api_key" not in kwargs


# ===========================================================================
# OpenAIVoiceAdapter.listen (lines 214-243)
# ===========================================================================

class TestOpenAIVoiceAdapterListen:
    def _make_adapter(self) -> OpenAIVoiceAdapter:
        return OpenAIVoiceAdapter(model="whisper-1")

    def test_listen_records_and_transcribes(self) -> None:
        adapter = self._make_adapter()

        mock_sd = MagicMock()
        mock_np = MagicMock()
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_transcript = MagicMock()
        mock_transcript.text = " hello from openai "
        mock_client.audio.transcriptions.create.return_value = mock_transcript

        # sd.rec returns a numpy-like array
        mock_audio = MagicMock()
        mock_audio.tobytes.return_value = b"\x00" * 100
        mock_sd.rec.return_value = mock_audio

        with patch.dict("sys.modules", {
            "openai": mock_openai,
            "sounddevice": mock_sd,
            "numpy": mock_np,
        }):
            text = adapter.listen(timeout_s=2.0)

        assert text == "hello from openai"
        mock_sd.rec.assert_called_once()
        mock_sd.wait.assert_called_once()
        mock_client.audio.transcriptions.create.assert_called_once()

    def test_listen_uses_int16_dtype(self) -> None:
        adapter = self._make_adapter()
        mock_sd = MagicMock()
        mock_np = MagicMock()
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_transcript = MagicMock()
        mock_transcript.text = "ok"
        mock_client.audio.transcriptions.create.return_value = mock_transcript

        mock_audio = MagicMock()
        mock_audio.tobytes.return_value = b"\x00" * 100
        mock_sd.rec.return_value = mock_audio

        with patch.dict("sys.modules", {
            "openai": mock_openai,
            "sounddevice": mock_sd,
            "numpy": mock_np,
        }):
            adapter.listen()

        call_kwargs = mock_sd.rec.call_args[1]
        assert call_kwargs.get("dtype") == "int16"

    def test_listen_passes_model_to_api(self) -> None:
        adapter = OpenAIVoiceAdapter(model="whisper-2-custom")
        mock_sd = MagicMock()
        mock_np = MagicMock()
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_transcript = MagicMock()
        mock_transcript.text = "result"
        mock_client.audio.transcriptions.create.return_value = mock_transcript

        mock_audio = MagicMock()
        mock_audio.tobytes.return_value = b"\x00" * 100
        mock_sd.rec.return_value = mock_audio

        with patch.dict("sys.modules", {
            "openai": mock_openai,
            "sounddevice": mock_sd,
            "numpy": mock_np,
        }):
            adapter.listen()

        create_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        assert create_kwargs["model"] == "whisper-2-custom"


# ===========================================================================
# OpenAIVoiceAdapter.listen_from_file (lines 247-253)
# ===========================================================================

class TestOpenAIVoiceAdapterListenFromFile:
    def test_listen_from_file_opens_and_transcribes(self, tmp_path) -> None:
        audio_file = tmp_path / "speech.wav"
        audio_file.write_bytes(b"fake wav")

        adapter = OpenAIVoiceAdapter()
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_transcript = MagicMock()
        mock_transcript.text = "  transcribed text  "
        mock_client.audio.transcriptions.create.return_value = mock_transcript

        with patch.dict("sys.modules", {"openai": mock_openai}):
            text = adapter.listen_from_file(str(audio_file))

        assert text == "transcribed text"
        mock_client.audio.transcriptions.create.assert_called_once()

    def test_listen_from_file_passes_model(self, tmp_path) -> None:
        f = tmp_path / "a.wav"
        f.write_bytes(b"x")

        adapter = OpenAIVoiceAdapter(model="whisper-large")
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_transcript = MagicMock()
        mock_transcript.text = "ok"
        mock_client.audio.transcriptions.create.return_value = mock_transcript

        with patch.dict("sys.modules", {"openai": mock_openai}):
            adapter.listen_from_file(str(f))

        create_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        assert create_kwargs["model"] == "whisper-large"


# ===========================================================================
# OpenAIVoiceAdapter.speak (lines 257-278)
# ===========================================================================

class TestOpenAIVoiceAdapterSpeak:
    def test_speak_writes_tmp_file_and_plays(self, tmp_path) -> None:
        adapter = OpenAIVoiceAdapter(tts_model="tts-1", tts_voice="nova")
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = b"mp3 audio data"
        mock_client.audio.speech.create.return_value = mock_response

        mock_subprocess = MagicMock()
        mock_subprocess.run.return_value = MagicMock()
        mock_os = MagicMock()

        with patch.dict("sys.modules", {
            "openai": mock_openai,
            "subprocess": mock_subprocess,
            "os": mock_os,
        }):
            adapter.speak("say this")

        mock_client.audio.speech.create.assert_called_once_with(
            model="tts-1",
            voice="nova",
            input="say this",
        )
        # ffplay should be called
        mock_subprocess.run.assert_called_once()
        ffplay_args = mock_subprocess.run.call_args[0][0]
        assert "ffplay" in ffplay_args

        # os.unlink should clean up temp file
        mock_os.unlink.assert_called_once()

    def test_speak_ffplay_not_found_still_cleans_up(self) -> None:
        """FileNotFoundError from ffplay is caught; temp file is still deleted."""
        adapter = OpenAIVoiceAdapter()
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = b"audio"
        mock_client.audio.speech.create.return_value = mock_response

        mock_subprocess = MagicMock()
        mock_subprocess.run.side_effect = FileNotFoundError("ffplay not found")
        mock_os = MagicMock()

        with patch.dict("sys.modules", {
            "openai": mock_openai,
            "subprocess": mock_subprocess,
            "os": mock_os,
        }):
            # Should NOT raise — FileNotFoundError is caught in speak()
            adapter.speak("silent test")

        # os.unlink must still be called (finally block)
        mock_os.unlink.assert_called_once()

    def test_speak_passes_correct_tts_voice(self) -> None:
        adapter = OpenAIVoiceAdapter(tts_voice="shimmer")
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = b"x"
        mock_client.audio.speech.create.return_value = mock_response

        mock_subprocess = MagicMock()
        mock_os = MagicMock()

        with patch.dict("sys.modules", {
            "openai": mock_openai,
            "subprocess": mock_subprocess,
            "os": mock_os,
        }):
            adapter.speak("custom voice")

        create_kwargs = mock_client.audio.speech.create.call_args[1]
        assert create_kwargs["voice"] == "shimmer"


# ===========================================================================
# OpenAIVoiceAdapter.is_available (lines 281-285)
# ===========================================================================

class TestOpenAIVoiceAdapterIsAvailable:
    def test_is_available_true_when_openai_importable(self) -> None:
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            adapter = OpenAIVoiceAdapter()
            assert adapter.is_available() is True

    def test_is_available_false_when_openai_not_importable(self) -> None:
        """ImportError → returns False."""
        adapter = OpenAIVoiceAdapter()
        original = sys.modules.pop("openai", None)
        try:
            with patch("builtins.__import__", side_effect=lambda name, *a, **kw: (
                (_ for _ in ()).throw(ImportError("no openai")) if name == "openai"
                else __import__(name, *a, **kw)
            )):
                result = adapter.is_available()
            assert result is False
        finally:
            if original is not None:
                sys.modules["openai"] = original

"""Tests for config-based VoiceAdapter (SpeechAdapter / StubVoiceAdapter)."""
from __future__ import annotations

import os
import pytest

from apyrobo.voice import (
    VoiceConfig,
    TranscriptionResult,
    SynthesisResult,
    SpeechAdapter,
    StubVoiceAdapter,
)


# ---------------------------------------------------------------------------
# StubVoiceAdapter
# ---------------------------------------------------------------------------

class TestStubVoiceAdapter:
    def test_transcribe_returns_transcription_result(self):
        adapter = StubVoiceAdapter()
        result = adapter.transcribe(b"\x00\x01")
        assert isinstance(result, TranscriptionResult)

    def test_transcribe_result_fields(self):
        adapter = StubVoiceAdapter()
        result = adapter.transcribe(b"audio")
        assert result.text == "hello robot"
        assert result.confidence == 1.0
        assert result.language == "en"
        assert result.duration_ms == 10.0

    def test_synthesize_returns_synthesis_result(self):
        adapter = StubVoiceAdapter()
        result = adapter.synthesize("hello")
        assert isinstance(result, SynthesisResult)

    def test_synthesize_result_fields(self):
        adapter = StubVoiceAdapter()
        result = adapter.synthesize("test text")
        assert result.text == "test text"
        assert result.audio_bytes == b"\x00\x01\x02\x03"
        assert result.duration_ms == 10.0

    def test_is_available_always_true(self):
        adapter = StubVoiceAdapter()
        assert adapter.is_available() is True

    def test_is_speech_adapter(self):
        adapter = StubVoiceAdapter()
        assert isinstance(adapter, SpeechAdapter)

    def test_transcribe_empty_audio(self):
        adapter = StubVoiceAdapter()
        result = adapter.transcribe(b"")
        assert result.text == "hello robot"

    def test_synthesize_empty_text(self):
        adapter = StubVoiceAdapter()
        result = adapter.synthesize("")
        assert result.text == ""
        assert isinstance(result.audio_bytes, bytes)


# ---------------------------------------------------------------------------
# SpeechAdapter with stub backend
# ---------------------------------------------------------------------------

class TestSpeechAdapterStubBackend:
    def test_is_available_stub(self):
        adapter = SpeechAdapter(VoiceConfig(stt_backend="stub", tts_backend="stub"))
        assert adapter.is_available() is True

    def test_transcribe_stub_returns_fixed_text(self):
        adapter = SpeechAdapter(VoiceConfig(stt_backend="stub"))
        result = adapter.transcribe(b"audio data")
        assert result.text == "stub transcription"
        assert isinstance(result, TranscriptionResult)

    def test_synthesize_stub_returns_empty_bytes(self):
        adapter = SpeechAdapter(VoiceConfig(tts_backend="stub"))
        result = adapter.synthesize("speak this")
        assert result.audio_bytes == b""
        assert result.text == "speak this"
        assert isinstance(result, SynthesisResult)

    def test_default_config_is_stub(self):
        adapter = SpeechAdapter()
        assert adapter.config.stt_backend == "stub"
        assert adapter.config.tts_backend == "stub"

    def test_transcribe_confidence_is_one(self):
        adapter = SpeechAdapter()
        result = adapter.transcribe(b"x")
        assert result.confidence == 1.0

    def test_transcribe_uses_config_language(self):
        adapter = SpeechAdapter(VoiceConfig(language="fr"))
        result = adapter.transcribe(b"audio")
        assert result.language == "fr"


# ---------------------------------------------------------------------------
# VoiceConfig dataclass
# ---------------------------------------------------------------------------

class TestVoiceConfig:
    def test_default_backends(self):
        cfg = VoiceConfig()
        assert cfg.stt_backend == "stub"
        assert cfg.tts_backend == "stub"

    def test_whisper_backend(self):
        cfg = VoiceConfig(stt_backend="whisper", whisper_model="small")
        assert cfg.stt_backend == "whisper"
        assert cfg.whisper_model == "small"

    def test_openai_backend(self):
        cfg = VoiceConfig(stt_backend="openai", tts_backend="openai", openai_api_key="sk-test")
        assert cfg.stt_backend == "openai"
        assert cfg.tts_backend == "openai"
        assert cfg.openai_api_key == "sk-test"

    def test_piper_tts_backend(self):
        cfg = VoiceConfig(tts_backend="piper")
        assert cfg.tts_backend == "piper"

    def test_language_default(self):
        cfg = VoiceConfig()
        assert cfg.language == "en"


# ---------------------------------------------------------------------------
# transcribe_file
# ---------------------------------------------------------------------------

class TestTranscribeFile:
    def test_transcribe_file_reads_path(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00\x01\x02\x03")
        adapter = SpeechAdapter()
        result = adapter.transcribe_file(str(audio_file))
        assert isinstance(result, TranscriptionResult)

    def test_transcribe_file_stub_returns_fixed_text(self, tmp_path):
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"fake audio")
        adapter = SpeechAdapter()
        result = adapter.transcribe_file(str(audio_file))
        assert result.text == "stub transcription"

    def test_stub_adapter_transcribe_file(self, tmp_path):
        audio_file = tmp_path / "input.wav"
        audio_file.write_bytes(b"data")
        adapter = StubVoiceAdapter()
        result = adapter.transcribe_file(str(audio_file))
        assert result.text == "hello robot"

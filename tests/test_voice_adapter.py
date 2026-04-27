"""
Tests for VoiceConfig, TranscriptionResult, SynthesisResult, StubVoiceAdapter.
"""

from __future__ import annotations

import pytest

from apyrobo.voice import (
    VoiceAdapter,
    VoiceConfig,
    TranscriptionResult,
    SynthesisResult,
    StubVoiceAdapter,
    MockVoiceAdapter,
)


class TestVoiceConfig:
    def test_defaults(self):
        cfg = VoiceConfig()
        assert cfg.backend == "stub"
        assert cfg.sample_rate == 16000
        assert cfg.timeout_s == 5.0
        assert cfg.language == "en"

    def test_custom_values(self):
        cfg = VoiceConfig(backend="whisper", model_name="large", timeout_s=10.0)
        assert cfg.backend == "whisper"
        assert cfg.model_name == "large"
        assert cfg.timeout_s == 10.0

    def test_tts_voice_field(self):
        cfg = VoiceConfig(tts_voice="nova")
        assert cfg.tts_voice == "nova"


class TestTranscriptionResult:
    def test_minimal_construction(self):
        result = TranscriptionResult(text="hello")
        assert result.text == "hello"
        assert result.confidence == 1.0
        assert result.duration_s == 0.0

    def test_full_construction(self):
        result = TranscriptionResult(
            text="navigate to room 5",
            confidence=0.92,
            duration_s=2.3,
            language="en",
            raw={"words": ["navigate", "to", "room", "5"]},
        )
        assert result.text == "navigate to room 5"
        assert result.confidence == 0.92
        assert result.duration_s == 2.3
        assert result.raw["words"] == ["navigate", "to", "room", "5"]

    def test_raw_defaults_to_empty_dict(self):
        result = TranscriptionResult(text="test")
        assert result.raw == {}


class TestSynthesisResult:
    def test_success(self):
        result = SynthesisResult(success=True, message="ok")
        assert result.success is True

    def test_failure(self):
        result = SynthesisResult(success=False, message="TTS engine unavailable")
        assert result.success is False
        assert "unavailable" in result.message

    def test_audio_bytes_field(self):
        result = SynthesisResult(success=True, audio_bytes=b"\x00\x01\x02")
        assert len(result.audio_bytes) == 3

    def test_audio_path_field(self):
        result = SynthesisResult(success=True, audio_path="/tmp/speech.wav")
        assert result.audio_path == "/tmp/speech.wav"

    def test_duration_field(self):
        result = SynthesisResult(success=True, duration_s=1.5)
        assert result.duration_s == 1.5


class TestStubVoiceAdapter:
    def test_is_voice_adapter(self):
        adapter = StubVoiceAdapter()
        assert isinstance(adapter, VoiceAdapter)

    def test_is_available(self):
        adapter = StubVoiceAdapter()
        assert adapter.is_available() is True

    def test_listen_returns_empty_when_queue_empty(self):
        adapter = StubVoiceAdapter()
        assert adapter.listen() == ""

    def test_enqueue_single(self):
        adapter = StubVoiceAdapter()
        adapter.enqueue("go to kitchen")
        assert adapter.listen() == "go to kitchen"

    def test_enqueue_multiple_consumed_in_order(self):
        adapter = StubVoiceAdapter()
        adapter.enqueue("first", "second", "third")
        assert adapter.listen() == "first"
        assert adapter.listen() == "second"
        assert adapter.listen() == "third"
        assert adapter.listen() == ""  # queue empty

    def test_speak_records_synthesis(self):
        adapter = StubVoiceAdapter()
        adapter.speak("hello robot")
        assert len(adapter.syntheses) == 1
        assert adapter.syntheses[0].success is True

    def test_speak_with_result_returns_synthesis_result(self):
        adapter = StubVoiceAdapter()
        result = adapter.speak_with_result("task complete")
        assert isinstance(result, SynthesisResult)
        assert result.success is True

    def test_speak_duration_proportional_to_text_length(self):
        adapter = StubVoiceAdapter()
        short = adapter.speak_with_result("hi")
        long_ = adapter.speak_with_result("navigate to the charging station now")
        assert long_.duration_s > short.duration_s

    def test_listen_with_result_returns_transcription_result(self):
        adapter = StubVoiceAdapter()
        adapter.enqueue("patrol area A")
        result = adapter.listen_with_result()
        assert isinstance(result, TranscriptionResult)
        assert result.text == "patrol area A"
        assert result.confidence == 1.0

    def test_listen_with_result_records_transcription(self):
        adapter = StubVoiceAdapter()
        adapter.enqueue("stop")
        adapter.listen_with_result()
        assert len(adapter.transcriptions) == 1

    def test_config_sets_language(self):
        cfg = VoiceConfig(language="fr")
        adapter = StubVoiceAdapter(config=cfg)
        adapter.enqueue("bonjour")
        result = adapter.listen_with_result()
        assert result.language == "fr"

    def test_multiple_speak_calls_accumulate(self):
        adapter = StubVoiceAdapter()
        adapter.speak("one")
        adapter.speak("two")
        adapter.speak("three")
        assert len(adapter.syntheses) == 3

    def test_listen_without_enqueue_returns_empty_transcription(self):
        adapter = StubVoiceAdapter()
        result = adapter.listen_with_result()
        assert result.text == ""

    def test_default_config_applied_when_none_passed(self):
        adapter = StubVoiceAdapter()
        assert adapter.config.backend == "stub"
        assert adapter.config.timeout_s == 5.0

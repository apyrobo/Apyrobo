"""
Voice/Speech adapter — STT input and TTS output for robot interaction.

VC-01: Provides VoiceAdapter ABC with concrete implementations:
    - WhisperAdapter: local STT via openai-whisper + piper TTS
    - PiperAdapter: lightweight local TTS via piper-tts
    - OpenAIVoiceAdapter: cloud STT/TTS via OpenAI Whisper + TTS APIs
    - MockVoiceAdapter: deterministic adapter for testing

Also provides voice_loop() helper for interactive listen-plan-execute-speak cycles.
"""

from __future__ import annotations

import abc
import io
import logging
import tempfile
import wave
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class VoiceAdapter(abc.ABC):
    """
    Abstract voice adapter for speech-to-text and text-to-speech.

    Subclass this to add new STT/TTS backends.
    """

    @abc.abstractmethod
    def listen(self, timeout_s: float = 5.0) -> str:
        """Record audio and return transcribed text."""
        ...

    @abc.abstractmethod
    def speak(self, text: str) -> None:
        """Convert text to speech and play it."""
        ...

    def is_available(self) -> bool:
        """Check if the adapter's dependencies are installed."""
        return True


# ---------------------------------------------------------------------------
# WhisperAdapter — local STT via openai-whisper
# ---------------------------------------------------------------------------

class WhisperAdapter(VoiceAdapter):
    """
    STT using OpenAI's open-source Whisper model (local inference).

    Requires: pip install openai-whisper sounddevice numpy

    For TTS, delegates to piper if available, otherwise uses a simple
    subprocess call.
    """

    def __init__(self, model_size: str = "base", device: str | None = None) -> None:
        self._model_size = model_size
        self._device = device
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self._model_size, device=self._device)
        return self._model

    def listen(self, timeout_s: float = 5.0) -> str:
        """Record from microphone and transcribe with Whisper."""
        import numpy as np
        import sounddevice as sd

        sample_rate = 16000
        audio = sd.rec(
            int(timeout_s * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        audio_np = audio.flatten()
        model = self._load_model()
        result = model.transcribe(audio_np, fp16=False)
        text = result.get("text", "").strip()
        logger.info("WhisperAdapter.listen: %r", text)
        return text

    def listen_from_file(self, audio_path: str) -> str:
        """Transcribe from an audio file (for testing)."""
        model = self._load_model()
        result = model.transcribe(audio_path, fp16=False)
        text = result.get("text", "").strip()
        logger.info("WhisperAdapter.listen_from_file: %r", text)
        return text

    def speak(self, text: str) -> None:
        """Speak using piper TTS via subprocess."""
        import subprocess

        try:
            proc = subprocess.run(
                ["piper", "--output_raw"],
                input=text.encode(),
                capture_output=True,
                timeout=30,
            )
            if proc.returncode == 0 and proc.stdout:
                subprocess.run(
                    ["aplay", "-r", "22050", "-f", "S16_LE", "-c", "1"],
                    input=proc.stdout,
                    timeout=30,
                )
            else:
                logger.warning("piper TTS returned non-zero: %d", proc.returncode)
        except FileNotFoundError:
            logger.error("piper or aplay not found — install piper-tts")
            raise

    def is_available(self) -> bool:
        try:
            import whisper  # noqa: F401
            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# PiperAdapter — lightweight local TTS (offline-capable)
# ---------------------------------------------------------------------------

class PiperAdapter(VoiceAdapter):
    """
    TTS-only adapter using piper-tts for fast, offline speech synthesis.

    STT is not supported — listen() raises NotImplementedError.

    Requires: pip install piper-tts (or piper binary on PATH)
    """

    def __init__(self, voice: str | None = None) -> None:
        self._voice = voice

    def listen(self, timeout_s: float = 5.0) -> str:
        raise NotImplementedError("PiperAdapter is TTS-only; use WhisperAdapter for STT")

    def speak(self, text: str) -> None:
        """Synthesize and play speech using piper."""
        import subprocess

        cmd = ["piper", "--output_raw"]
        if self._voice:
            cmd.extend(["--model", self._voice])

        proc = subprocess.run(
            cmd,
            input=text.encode(),
            capture_output=True,
            timeout=30,
        )
        if proc.returncode == 0 and proc.stdout:
            subprocess.run(
                ["aplay", "-r", "22050", "-f", "S16_LE", "-c", "1"],
                input=proc.stdout,
                timeout=30,
            )
        else:
            logger.warning("piper TTS failed: %d", proc.returncode)

    def is_available(self) -> bool:
        import shutil
        return shutil.which("piper") is not None


# ---------------------------------------------------------------------------
# OpenAIVoiceAdapter — cloud STT/TTS via OpenAI APIs
# ---------------------------------------------------------------------------

class OpenAIVoiceAdapter(VoiceAdapter):
    """
    Cloud-based STT (Whisper API) and TTS (OpenAI TTS API).

    Requires: pip install openai sounddevice numpy
    Set OPENAI_API_KEY environment variable.
    """

    def __init__(self, model: str = "whisper-1", tts_model: str = "tts-1",
                 tts_voice: str = "alloy", api_key: str | None = None) -> None:
        self._model = model
        self._tts_model = tts_model
        self._tts_voice = tts_voice
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import openai
            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def listen(self, timeout_s: float = 5.0) -> str:
        """Record audio and transcribe via OpenAI Whisper API."""
        import numpy as np
        import sounddevice as sd

        sample_rate = 16000
        audio = sd.rec(
            int(timeout_s * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()

        # Convert to WAV bytes for the API
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        buf.seek(0)
        buf.name = "audio.wav"

        client = self._get_client()
        transcript = client.audio.transcriptions.create(
            model=self._model,
            file=buf,
        )
        text = transcript.text.strip()
        logger.info("OpenAIVoiceAdapter.listen: %r", text)
        return text

    def listen_from_file(self, audio_path: str) -> str:
        """Transcribe from an audio file via OpenAI API."""
        client = self._get_client()
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model=self._model,
                file=f,
            )
        return transcript.text.strip()

    def speak(self, text: str) -> None:
        """Synthesize speech via OpenAI TTS API and play it."""
        client = self._get_client()
        response = client.audio.speech.create(
            model=self._tts_model,
            voice=self._tts_voice,
            input=text,
        )

        # Write to temp file and play
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(response.content)
            tmp_path = f.name

        import subprocess
        try:
            subprocess.run(["ffplay", "-nodisp", "-autoexit", tmp_path],
                           capture_output=True, timeout=30)
        except FileNotFoundError:
            # Fallback to aplay (requires wav conversion)
            logger.warning("ffplay not found; audio saved to %s", tmp_path)
        finally:
            import os
            os.unlink(tmp_path)

    def is_available(self) -> bool:
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# MockVoiceAdapter — for testing
# ---------------------------------------------------------------------------

class MockVoiceAdapter(VoiceAdapter):
    """
    Deterministic voice adapter for testing.

    Returns pre-configured responses for listen() and records speak() calls.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self._response_idx = 0
        self.spoken: list[str] = []

    def listen(self, timeout_s: float = 5.0) -> str:
        if not self._responses:
            return ""
        text = self._responses[self._response_idx % len(self._responses)]
        self._response_idx += 1
        return text

    def speak(self, text: str) -> None:
        self.spoken.append(text)
        logger.info("MockVoiceAdapter.speak: %r", text)

    def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# voice_loop — interactive listen-plan-execute-speak cycle
# ---------------------------------------------------------------------------

def voice_loop(
    agent: Any,
    robot: Any,
    adapter: VoiceAdapter,
    max_turns: int | None = None,
    on_listen: Any = None,
    on_result: Any = None,
) -> list[dict[str, Any]]:
    """
    Run a voice interaction loop: listen -> plan+execute -> speak result.

    Args:
        agent: Agent instance for planning/execution
        robot: Robot instance
        adapter: VoiceAdapter for STT/TTS
        max_turns: Maximum number of turns (None = infinite)
        on_listen: Optional callback(text) after transcription
        on_result: Optional callback(result) after execution

    Returns:
        List of turn records: [{"input": str, "result": TaskResult, "summary": str}]
    """
    turns: list[dict[str, Any]] = []
    turn_count = 0

    while max_turns is None or turn_count < max_turns:
        # Listen
        text = adapter.listen()
        if not text:
            continue

        if on_listen:
            on_listen(text)

        # Check for stop commands
        if text.strip().lower() in ("stop", "quit", "exit", "bye"):
            adapter.speak("Goodbye.")
            break

        # Execute
        result = agent.execute(task=text, robot=robot)
        summary = f"Task {result.status.value}: {result.steps_completed}/{result.steps_total} steps completed"
        if result.error:
            summary += f". Error: {result.error}"

        if on_result:
            on_result(result)

        # Speak result
        adapter.speak(summary)

        turns.append({
            "input": text,
            "result": result,
            "summary": summary,
        })
        turn_count += 1

    return turns

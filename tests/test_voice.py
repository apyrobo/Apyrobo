"""
Tests for VC-01: Voice/Speech adapter — STT input and TTS output.
"""

from __future__ import annotations

import pytest

from apyrobo.voice import (
    VoiceAdapter,
    MockVoiceAdapter,
    WhisperAdapter,
    PiperAdapter,
    OpenAIVoiceAdapter,
    voice_loop,
)


# ---------------------------------------------------------------------------
# MockVoiceAdapter tests
# ---------------------------------------------------------------------------

class TestMockVoiceAdapter:
    def test_listen_returns_configured_responses(self):
        adapter = MockVoiceAdapter(responses=["hello", "go to room 3"])
        assert adapter.listen() == "hello"
        assert adapter.listen() == "go to room 3"

    def test_listen_cycles_through_responses(self):
        adapter = MockVoiceAdapter(responses=["a", "b"])
        assert adapter.listen() == "a"
        assert adapter.listen() == "b"
        assert adapter.listen() == "a"  # wraps around

    def test_listen_empty_responses(self):
        adapter = MockVoiceAdapter()
        assert adapter.listen() == ""

    def test_speak_records_text(self):
        adapter = MockVoiceAdapter()
        adapter.speak("hello world")
        adapter.speak("task complete")
        assert adapter.spoken == ["hello world", "task complete"]

    def test_is_available(self):
        adapter = MockVoiceAdapter()
        assert adapter.is_available() is True

    def test_is_voice_adapter(self):
        adapter = MockVoiceAdapter()
        assert isinstance(adapter, VoiceAdapter)


# ---------------------------------------------------------------------------
# WhisperAdapter tests (without actual whisper model)
# ---------------------------------------------------------------------------

class TestWhisperAdapter:
    def test_instantiation(self):
        adapter = WhisperAdapter(model_size="tiny")
        assert adapter._model_size == "tiny"

    def test_is_voice_adapter(self):
        adapter = WhisperAdapter()
        assert isinstance(adapter, VoiceAdapter)


# ---------------------------------------------------------------------------
# PiperAdapter tests
# ---------------------------------------------------------------------------

class TestPiperAdapter:
    def test_listen_raises_not_implemented(self):
        adapter = PiperAdapter()
        with pytest.raises(NotImplementedError, match="TTS-only"):
            adapter.listen()

    def test_is_voice_adapter(self):
        adapter = PiperAdapter()
        assert isinstance(adapter, VoiceAdapter)


# ---------------------------------------------------------------------------
# OpenAIVoiceAdapter tests
# ---------------------------------------------------------------------------

class TestOpenAIVoiceAdapter:
    def test_instantiation(self):
        adapter = OpenAIVoiceAdapter(tts_voice="nova")
        assert adapter._tts_voice == "nova"

    def test_is_voice_adapter(self):
        adapter = OpenAIVoiceAdapter()
        assert isinstance(adapter, VoiceAdapter)


# ---------------------------------------------------------------------------
# voice_loop tests
# ---------------------------------------------------------------------------

class TestVoiceLoop:
    def test_voice_loop_one_turn(self):
        """voice_loop() runs one complete listen-plan-execute-speak cycle in mock mode."""
        from apyrobo.core.robot import Robot
        from apyrobo.skills.agent import Agent

        robot = Robot.discover("mock://turtlebot4")
        agent = Agent(provider="rule")
        adapter = MockVoiceAdapter(responses=["go to (2, 3)", "stop"])

        turns = voice_loop(
            agent=agent,
            robot=robot,
            adapter=adapter,
            max_turns=1,
        )

        assert len(turns) == 1
        assert turns[0]["input"] == "go to (2, 3)"
        assert turns[0]["result"] is not None
        assert turns[0]["summary"]
        # Adapter should have spoken the result summary
        assert len(adapter.spoken) == 1

    def test_voice_loop_stop_command(self):
        """voice_loop exits when 'stop' is heard."""
        from apyrobo.core.robot import Robot
        from apyrobo.skills.agent import Agent

        robot = Robot.discover("mock://turtlebot4")
        agent = Agent(provider="rule")
        adapter = MockVoiceAdapter(responses=["stop"])

        turns = voice_loop(
            agent=agent,
            robot=robot,
            adapter=adapter,
            max_turns=10,
        )

        assert len(turns) == 0
        assert "Goodbye." in adapter.spoken

    def test_voice_loop_with_callbacks(self):
        """voice_loop calls on_listen and on_result callbacks."""
        from apyrobo.core.robot import Robot
        from apyrobo.skills.agent import Agent

        robot = Robot.discover("mock://turtlebot4")
        agent = Agent(provider="rule")
        adapter = MockVoiceAdapter(responses=["navigate to (1, 1)"])

        listened = []
        results = []

        turns = voice_loop(
            agent=agent,
            robot=robot,
            adapter=adapter,
            max_turns=1,
            on_listen=lambda t: listened.append(t),
            on_result=lambda r: results.append(r),
        )

        assert len(listened) == 1
        assert len(results) == 1


# ---------------------------------------------------------------------------
# SPEAK skill dispatch
# ---------------------------------------------------------------------------

class TestSpeakSkillHandler:
    def test_speak_handler_registered(self):
        """SPEAK skill dispatches to handler."""
        import apyrobo.skills.builtins  # noqa: F401 — triggers registration
        from apyrobo.skills.handlers import dispatch

        class FakeRobot:
            _voice_adapter = MockVoiceAdapter()
            def capabilities(self):
                return None

        robot = FakeRobot()
        result = dispatch("speak", robot, {"text": "hello robot"})
        assert result is True
        assert robot._voice_adapter.spoken == ["hello robot"]

    def test_speak_handler_no_adapter(self):
        """SPEAK skill works even without a voice adapter."""
        import apyrobo.skills.builtins  # noqa: F401
        from apyrobo.skills.handlers import dispatch

        class FakeRobot:
            pass

        result = dispatch("speak", FakeRobot(), {"text": "hello"})
        assert result is True

    def test_speak_handler_no_text(self):
        """SPEAK skill returns False when no text provided."""
        import apyrobo.skills.builtins  # noqa: F401
        from apyrobo.skills.handlers import dispatch

        class FakeRobot:
            pass

        result = dispatch("speak", FakeRobot(), {})
        assert result is False

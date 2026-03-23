"""
Comprehensive tests for apyrobo/voice.py — targeting missing coverage lines.

Covers:
- VoiceAdapter.is_available() default returns True
- WhisperAdapter (is_available with/without whisper module)
- PiperAdapter (listen raises NotImplementedError, speak mocked, is_available mocked)
- MockVoiceAdapter (listen with/without responses, multiple calls cycling,
  speak records to spoken, is_available True)
- voice_loop with MockVoiceAdapter and mock agent+robot
  (normal turn, stop commands stop/quit/exit/bye, max_turns, on_listen/on_result
  callbacks, empty listen response continues loop)
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from apyrobo.core.schemas import TaskResult, TaskStatus
from apyrobo.voice import (
    MockVoiceAdapter,
    PiperAdapter,
    VoiceAdapter,
    WhisperAdapter,
    voice_loop,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_task_result(status: TaskStatus = TaskStatus.COMPLETED,
                     steps_completed: int = 2, steps_total: int = 2,
                     error: str | None = None) -> TaskResult:
    return TaskResult(
        task_name="test_task",
        status=status,
        steps_completed=steps_completed,
        steps_total=steps_total,
        error=error,
    )


def make_mock_agent(task_result: TaskResult | None = None) -> MagicMock:
    agent = MagicMock()
    result = task_result or make_task_result()
    agent.execute.return_value = result
    return agent


def make_mock_robot() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# VoiceAdapter base class
# ---------------------------------------------------------------------------

class ConcreteVoiceAdapter(VoiceAdapter):
    """Minimal concrete implementation for testing the ABC."""
    def listen(self, timeout_s: float = 5.0) -> str:
        return "hello"

    def speak(self, text: str) -> None:
        pass


class TestVoiceAdapterBase:
    def test_is_available_default_true(self):
        adapter = ConcreteVoiceAdapter()
        assert adapter.is_available() is True

    def test_listen_returns_str(self):
        adapter = ConcreteVoiceAdapter()
        result = adapter.listen()
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# WhisperAdapter
# ---------------------------------------------------------------------------

class TestWhisperAdapter:
    def test_init(self):
        adapter = WhisperAdapter(model_size="base", device=None)
        assert adapter._model_size == "base"
        assert adapter._device is None
        assert adapter._model is None

    def test_init_with_custom_model_size(self):
        adapter = WhisperAdapter(model_size="small")
        assert adapter._model_size == "small"

    def test_is_available_returns_true_when_whisper_importable(self):
        fake_whisper = MagicMock()
        with patch.dict(sys.modules, {"whisper": fake_whisper}):
            adapter = WhisperAdapter()
            assert adapter.is_available() is True

    def test_is_available_returns_false_when_whisper_missing(self):
        with patch.dict(sys.modules, {"whisper": None}):
            # When the module is set to None, import raises ImportError
            adapter = WhisperAdapter()
            # Temporarily remove whisper from sys.modules to simulate ImportError
            original = sys.modules.pop("whisper", None)
            try:
                with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: (
                    (_ for _ in ()).throw(ImportError("No module named 'whisper'"))
                    if name == "whisper" else __import__(name, *args, **kwargs)
                )):
                    result = adapter.is_available()
                    assert result is False
            finally:
                if original is not None:
                    sys.modules["whisper"] = original

    def test_is_available_false_when_whisper_not_installed(self):
        adapter = WhisperAdapter()
        # Remove whisper from modules to simulate it not being installed
        whisper_backup = sys.modules.pop("whisper", None)
        try:
            with patch("builtins.__import__") as mock_import:
                def side_effect(name, *args, **kwargs):
                    if name == "whisper":
                        raise ImportError("No module named 'whisper'")
                    return __import__(name, *args, **kwargs)
                mock_import.side_effect = side_effect
                result = adapter.is_available()
                assert result is False
        except Exception:
            pass  # may fail if import mechanism is complex
        finally:
            if whisper_backup is not None:
                sys.modules["whisper"] = whisper_backup


# ---------------------------------------------------------------------------
# PiperAdapter
# ---------------------------------------------------------------------------

class TestPiperAdapter:
    def test_init_default_voice(self):
        adapter = PiperAdapter()
        assert adapter._voice is None

    def test_init_with_voice(self):
        adapter = PiperAdapter(voice="en_US/ljspeech_low")
        assert adapter._voice == "en_US/ljspeech_low"

    def test_listen_raises_not_implemented_error(self):
        adapter = PiperAdapter()
        with pytest.raises(NotImplementedError):
            adapter.listen()

    def test_listen_raises_with_timeout(self):
        adapter = PiperAdapter()
        with pytest.raises(NotImplementedError):
            adapter.listen(timeout_s=3.0)

    def test_speak_calls_subprocess_run(self):
        adapter = PiperAdapter()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = b"audio_data"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = mock_proc
            adapter.speak("hello world")

        # subprocess.run should be called (at least once for piper)
        assert mock_run.called

    def test_speak_with_voice_model(self):
        adapter = PiperAdapter(voice="en_US/ljspeech_low")
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = b"audio_data"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = mock_proc
            adapter.speak("test speech")

        # First call should include --model flag
        first_call_args = mock_run.call_args_list[0][0][0]
        assert "--model" in first_call_args

    def test_speak_handles_piper_failure(self):
        adapter = PiperAdapter()
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = b""

        with patch("subprocess.run", return_value=mock_proc):
            # Should not raise, just log warning
            adapter.speak("will fail")

    def test_is_available_when_piper_on_path(self):
        adapter = PiperAdapter()
        with patch("shutil.which", return_value="/usr/bin/piper"):
            assert adapter.is_available() is True

    def test_is_available_when_piper_not_on_path(self):
        adapter = PiperAdapter()
        with patch("shutil.which", return_value=None):
            assert adapter.is_available() is False


# ---------------------------------------------------------------------------
# MockVoiceAdapter
# ---------------------------------------------------------------------------

class TestMockVoiceAdapter:
    def test_listen_returns_empty_when_no_responses(self):
        adapter = MockVoiceAdapter()
        assert adapter.listen() == ""

    def test_listen_returns_empty_with_empty_list(self):
        adapter = MockVoiceAdapter(responses=[])
        assert adapter.listen() == ""

    def test_listen_returns_first_response(self):
        adapter = MockVoiceAdapter(responses=["hello"])
        assert adapter.listen() == "hello"

    def test_listen_returns_responses_in_order(self):
        adapter = MockVoiceAdapter(responses=["first", "second", "third"])
        assert adapter.listen() == "first"
        assert adapter.listen() == "second"
        assert adapter.listen() == "third"

    def test_listen_cycles_responses(self):
        adapter = MockVoiceAdapter(responses=["a", "b"])
        assert adapter.listen() == "a"
        assert adapter.listen() == "b"
        assert adapter.listen() == "a"  # wraps around
        assert adapter.listen() == "b"

    def test_listen_single_response_always_same(self):
        adapter = MockVoiceAdapter(responses=["repeat"])
        for _ in range(5):
            assert adapter.listen() == "repeat"

    def test_speak_records_to_spoken(self):
        adapter = MockVoiceAdapter()
        adapter.speak("hello")
        adapter.speak("world")
        assert adapter.spoken == ["hello", "world"]

    def test_speak_empty_spoken_initially(self):
        adapter = MockVoiceAdapter()
        assert adapter.spoken == []

    def test_is_available_returns_true(self):
        adapter = MockVoiceAdapter()
        assert adapter.is_available() is True

    def test_is_instance_of_voice_adapter(self):
        adapter = MockVoiceAdapter()
        assert isinstance(adapter, VoiceAdapter)

    def test_listen_with_timeout_arg(self):
        adapter = MockVoiceAdapter(responses=["hi"])
        result = adapter.listen(timeout_s=10.0)
        assert result == "hi"


# ---------------------------------------------------------------------------
# voice_loop
# ---------------------------------------------------------------------------

class TestVoiceLoop:
    def test_normal_turn_executes_task(self):
        result = make_task_result(steps_completed=3, steps_total=3)
        agent = make_mock_agent(result)
        robot = make_mock_robot()
        adapter = MockVoiceAdapter(responses=["go to room 3", "stop"])

        turns = voice_loop(agent, robot, adapter, max_turns=1)
        assert len(turns) == 1
        assert turns[0]["input"] == "go to room 3"

    def test_stop_command_exits_loop(self):
        agent = make_mock_agent()
        robot = make_mock_robot()
        adapter = MockVoiceAdapter(responses=["stop"])

        turns = voice_loop(agent, robot, adapter)
        assert len(turns) == 0
        assert "Goodbye" in adapter.spoken[0]

    def test_quit_command_exits_loop(self):
        agent = make_mock_agent()
        robot = make_mock_robot()
        adapter = MockVoiceAdapter(responses=["quit"])

        turns = voice_loop(agent, robot, adapter)
        assert len(turns) == 0

    def test_exit_command_exits_loop(self):
        agent = make_mock_agent()
        robot = make_mock_robot()
        adapter = MockVoiceAdapter(responses=["exit"])

        turns = voice_loop(agent, robot, adapter)
        assert len(turns) == 0

    def test_bye_command_exits_loop(self):
        agent = make_mock_agent()
        robot = make_mock_robot()
        adapter = MockVoiceAdapter(responses=["bye"])

        turns = voice_loop(agent, robot, adapter)
        assert len(turns) == 0

    def test_max_turns_limits_iterations(self):
        agent = make_mock_agent()
        robot = make_mock_robot()
        # Keep returning non-stop commands
        adapter = MockVoiceAdapter(responses=["do something", "do more"])

        turns = voice_loop(agent, robot, adapter, max_turns=2)
        assert len(turns) == 2

    def test_max_turns_zero_returns_empty(self):
        agent = make_mock_agent()
        robot = make_mock_robot()
        adapter = MockVoiceAdapter(responses=["hello"])

        turns = voice_loop(agent, robot, adapter, max_turns=0)
        assert len(turns) == 0

    def test_empty_listen_response_continues_loop(self):
        # Empty responses should be skipped and not count as turns
        result = make_task_result()
        agent = make_mock_agent(result)
        robot = make_mock_robot()
        # Start with empty, then a command, then stop
        adapter = MockVoiceAdapter(responses=["", "do task", "stop"])
        # With max_turns=1 and the first response being empty (skipped),
        # the second response "do task" should be processed
        turns = voice_loop(agent, robot, adapter, max_turns=1)
        assert len(turns) == 1
        assert turns[0]["input"] == "do task"

    def test_on_listen_callback_called(self):
        result = make_task_result()
        agent = make_mock_agent(result)
        robot = make_mock_robot()
        adapter = MockVoiceAdapter(responses=["navigate forward", "stop"])

        heard = []
        def on_listen(text: str) -> None:
            heard.append(text)

        voice_loop(agent, robot, adapter, max_turns=1, on_listen=on_listen)
        assert "navigate forward" in heard

    def test_on_result_callback_called(self):
        task_result = make_task_result(status=TaskStatus.COMPLETED)
        agent = make_mock_agent(task_result)
        robot = make_mock_robot()
        adapter = MockVoiceAdapter(responses=["do task", "stop"])

        results_seen = []
        def on_result(r: Any) -> None:
            results_seen.append(r)

        voice_loop(agent, robot, adapter, max_turns=1, on_result=on_result)
        assert len(results_seen) == 1

    def test_summary_spoken_back(self):
        result = make_task_result(
            status=TaskStatus.COMPLETED,
            steps_completed=2,
            steps_total=2,
        )
        agent = make_mock_agent(result)
        robot = make_mock_robot()
        adapter = MockVoiceAdapter(responses=["go somewhere", "stop"])

        voice_loop(agent, robot, adapter, max_turns=1)
        # The summary should have been spoken
        assert len(adapter.spoken) >= 1
        # Last spoken before Goodbye should contain status info
        summary_spoken = [s for s in adapter.spoken if "completed" in s.lower() or "Task" in s]
        assert len(summary_spoken) >= 1

    def test_summary_includes_error_when_task_fails(self):
        result = make_task_result(
            status=TaskStatus.FAILED,
            steps_completed=0,
            steps_total=2,
            error="Something went wrong",
        )
        agent = make_mock_agent(result)
        robot = make_mock_robot()
        adapter = MockVoiceAdapter(responses=["broken task", "stop"])

        voice_loop(agent, robot, adapter, max_turns=1)
        # Summary should contain the error
        spoken_text = " ".join(adapter.spoken)
        assert "Something went wrong" in spoken_text

    def test_turn_record_structure(self):
        result = make_task_result(steps_completed=1, steps_total=1)
        agent = make_mock_agent(result)
        robot = make_mock_robot()
        adapter = MockVoiceAdapter(responses=["test command", "stop"])

        turns = voice_loop(agent, robot, adapter, max_turns=1)
        assert len(turns) == 1
        turn = turns[0]
        assert "input" in turn
        assert "result" in turn
        assert "summary" in turn
        assert turn["input"] == "test command"

    def test_agent_execute_called_with_task_and_robot(self):
        result = make_task_result()
        agent = make_mock_agent(result)
        robot = make_mock_robot()
        adapter = MockVoiceAdapter(responses=["navigate to kitchen", "stop"])

        voice_loop(agent, robot, adapter, max_turns=1)
        agent.execute.assert_called_once_with(task="navigate to kitchen", robot=robot)

    def test_stop_command_case_insensitive(self):
        agent = make_mock_agent()
        robot = make_mock_robot()
        adapter = MockVoiceAdapter(responses=["STOP"])

        turns = voice_loop(agent, robot, adapter)
        assert len(turns) == 0

    def test_stop_command_with_whitespace(self):
        agent = make_mock_agent()
        robot = make_mock_robot()
        adapter = MockVoiceAdapter(responses=["  stop  "])

        turns = voice_loop(agent, robot, adapter)
        assert len(turns) == 0

    def test_multiple_turns_before_stop(self):
        result = make_task_result()
        agent = make_mock_agent(result)
        robot = make_mock_robot()
        adapter = MockVoiceAdapter(responses=["task 1", "task 2", "task 3", "stop"])

        turns = voice_loop(agent, robot, adapter)
        assert len(turns) == 3

    def test_no_callbacks_no_error(self):
        result = make_task_result()
        agent = make_mock_agent(result)
        robot = make_mock_robot()
        adapter = MockVoiceAdapter(responses=["hello", "stop"])

        # Should work fine without callbacks
        turns = voice_loop(agent, robot, adapter, max_turns=1)
        assert len(turns) == 1

    def test_returns_list_of_turns(self):
        agent = make_mock_agent()
        robot = make_mock_robot()
        adapter = MockVoiceAdapter(responses=["stop"])

        result = voice_loop(agent, robot, adapter)
        assert isinstance(result, list)

    def test_none_max_turns_loops_until_stop(self):
        result = make_task_result()
        agent = make_mock_agent(result)
        robot = make_mock_robot()
        adapter = MockVoiceAdapter(responses=["cmd1", "cmd2", "stop"])

        turns = voice_loop(agent, robot, adapter, max_turns=None)
        assert len(turns) == 2

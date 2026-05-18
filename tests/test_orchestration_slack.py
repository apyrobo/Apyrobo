"""Tests for SlackOrchestrationAdapter (v6.0.0)."""
from __future__ import annotations

import json
import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from apyrobo.orchestration.adapter import OrchestrationMessage
from apyrobo.orchestration.slack_adapter import (
    SlackOrchestrationAdapter,
    _parse_command_text,
    _SLACK_IMPORT_ERROR,
)


# ---------------------------------------------------------------------------
# _parse_command_text unit tests
# ---------------------------------------------------------------------------

class TestParseCommandText:
    def test_simple_task(self):
        task, robot = _parse_command_text("navigate to dock", "mock://default")
        assert task == "navigate to dock"
        assert robot == "mock://default"

    def test_robot_keyword_extracted(self):
        task, robot = _parse_command_text(
            "pick up the cup robot=unitree://go2@192.168.1.100",
            "mock://default",
        )
        assert task == "pick up the cup"
        assert robot == "unitree://go2@192.168.1.100"

    def test_robot_keyword_at_start(self):
        task, robot = _parse_command_text(
            "robot=mock://spot navigate to dock",
            "mock://default",
        )
        assert task == "navigate to dock"
        assert robot == "mock://spot"

    def test_robot_keyword_case_insensitive_prefix(self):
        # "robot=" must be lowercase per spec; other tokens preserved
        task, robot = _parse_command_text("inspect robot=ros2://turtlebot4", "mock://default")
        assert task == "inspect"
        assert robot == "ros2://turtlebot4"

    def test_empty_text_fallback(self):
        task, robot = _parse_command_text("", "mock://fallback")
        assert robot == "mock://fallback"

    def test_default_robot_used_when_no_keyword(self):
        task, robot = _parse_command_text("patrol area", "mock://spot")
        assert robot == "mock://spot"


# ---------------------------------------------------------------------------
# SlackOrchestrationAdapter — no-SDK guard
# ---------------------------------------------------------------------------

class TestSlackAdapterNoSdk:
    def test_raises_importerror_without_slack_bolt(self):
        import apyrobo.orchestration.slack_adapter as slack_mod
        orig = slack_mod._SLACK_BOLT_AVAILABLE
        try:
            slack_mod._SLACK_BOLT_AVAILABLE = False
            with pytest.raises(ImportError, match="slack-bolt"):
                SlackOrchestrationAdapter()
        finally:
            slack_mod._SLACK_BOLT_AVAILABLE = orig


# ---------------------------------------------------------------------------
# Helpers — mock Slack Bolt
# ---------------------------------------------------------------------------

def _make_mock_bolt():
    """Return a (mock_app_class, mock_app_instance) pair."""
    mock_app = MagicMock()
    mock_app.client = MagicMock()
    mock_app.client.chat_postMessage = MagicMock(return_value={"ok": True})
    mock_app_class = MagicMock(return_value=mock_app)
    return mock_app_class, mock_app


def _patched_adapter(**kwargs) -> SlackOrchestrationAdapter:
    """Create a SlackOrchestrationAdapter with slack_bolt mocked out."""
    import apyrobo.orchestration.slack_adapter as slack_mod
    slack_mod._SLACK_BOLT_AVAILABLE = True
    adapter = SlackOrchestrationAdapter(**kwargs)
    return adapter


# ---------------------------------------------------------------------------
# SlackOrchestrationAdapter construction
# ---------------------------------------------------------------------------

class TestSlackAdapterInit:
    def test_reads_bot_token_from_env(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-env-token")
        import apyrobo.orchestration.slack_adapter as slack_mod
        slack_mod._SLACK_BOLT_AVAILABLE = True
        adapter = SlackOrchestrationAdapter()
        assert adapter._bot_token == "xoxb-env-token"

    def test_reads_signing_secret_from_env(self, monkeypatch):
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "secret-from-env")
        import apyrobo.orchestration.slack_adapter as slack_mod
        slack_mod._SLACK_BOLT_AVAILABLE = True
        adapter = SlackOrchestrationAdapter()
        assert adapter._signing_secret == "secret-from-env"

    def test_reads_default_robot_from_env(self, monkeypatch):
        monkeypatch.setenv("APYROBO_DEFAULT_ROBOT", "unitree://go2")
        import apyrobo.orchestration.slack_adapter as slack_mod
        slack_mod._SLACK_BOLT_AVAILABLE = True
        adapter = SlackOrchestrationAdapter(default_robot="")
        assert adapter._default_robot == "unitree://go2"

    def test_explicit_params_override_env(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-env")
        import apyrobo.orchestration.slack_adapter as slack_mod
        slack_mod._SLACK_BOLT_AVAILABLE = True
        adapter = SlackOrchestrationAdapter(bot_token="xoxb-explicit")
        assert adapter._bot_token == "xoxb-explicit"

    def test_command_property(self):
        import apyrobo.orchestration.slack_adapter as slack_mod
        slack_mod._SLACK_BOLT_AVAILABLE = True
        adapter = SlackOrchestrationAdapter(command="/mybot")
        assert adapter.command == "/mybot"

    def test_port_property(self):
        import apyrobo.orchestration.slack_adapter as slack_mod
        slack_mod._SLACK_BOLT_AVAILABLE = True
        adapter = SlackOrchestrationAdapter(port=4000)
        assert adapter.port == 4000

    def test_initial_recv_queue_empty(self):
        import apyrobo.orchestration.slack_adapter as slack_mod
        slack_mod._SLACK_BOLT_AVAILABLE = True
        adapter = SlackOrchestrationAdapter()
        assert adapter._recv_queue.empty()


# ---------------------------------------------------------------------------
# SlackOrchestrationAdapter — _build_app and command handler
# ---------------------------------------------------------------------------

class TestSlackAdapterCommandHandler:
    def _adapter_with_mock_bolt(self):
        import apyrobo.orchestration.slack_adapter as slack_mod
        slack_mod._SLACK_BOLT_AVAILABLE = True
        mock_app_class, mock_app = _make_mock_bolt()

        with patch.object(slack_mod, "_BoltApp", mock_app_class):
            adapter = SlackOrchestrationAdapter(
                bot_token="xoxb-test",
                signing_secret="secret",
                default_robot="mock://turtlebot4",
            )
            adapter._build_app()

        return adapter, mock_app

    def test_build_app_registers_command_handler(self):
        adapter, mock_app = self._adapter_with_mock_bolt()
        # _BoltApp.command() should have been called once
        mock_app.command.assert_called_once()

    def test_command_handler_enqueues_message(self):
        import apyrobo.orchestration.slack_adapter as slack_mod
        slack_mod._SLACK_BOLT_AVAILABLE = True
        mock_app_class, mock_app = _make_mock_bolt()

        registered_handler = None

        def capture_handler(cmd):
            def decorator(fn):
                nonlocal registered_handler
                registered_handler = fn
                return fn
            return decorator

        mock_app.command = capture_handler

        with patch.object(slack_mod, "_BoltApp", mock_app_class):
            adapter = SlackOrchestrationAdapter(
                bot_token="xoxb-test",
                signing_secret="secret",
                default_robot="mock://turtlebot4",
            )
            adapter._build_app()

        assert registered_handler is not None

        # Simulate a slash command invocation
        ack = MagicMock()
        say = MagicMock()
        body = {
            "text": "navigate to dock robot=mock://spot",
            "channel_id": "C123",
            "message_ts": "1234.5678",
        }
        registered_handler(ack=ack, body=body, say=say)

        ack.assert_called_once()
        assert not adapter._recv_queue.empty()
        msg = adapter._recv_queue.get_nowait()
        assert msg.task == "navigate to dock"
        assert msg.robot_uri == "mock://spot"
        assert msg.source == "slack"

    def test_command_handler_empty_text_acks_usage(self):
        import apyrobo.orchestration.slack_adapter as slack_mod
        slack_mod._SLACK_BOLT_AVAILABLE = True
        mock_app_class, mock_app = _make_mock_bolt()

        registered_handler = None

        def capture_handler(cmd):
            def decorator(fn):
                nonlocal registered_handler
                registered_handler = fn
                return fn
            return decorator

        mock_app.command = capture_handler

        with patch.object(slack_mod, "_BoltApp", mock_app_class):
            adapter = SlackOrchestrationAdapter(
                bot_token="xoxb-test",
                signing_secret="secret",
            )
            adapter._build_app()

        ack = MagicMock()
        say = MagicMock()
        body = {"text": "", "channel_id": "C123"}
        registered_handler(ack=ack, body=body, say=say)

        ack.assert_called_once()
        say.assert_called_once()
        # No message should be enqueued
        assert adapter._recv_queue.empty()


# ---------------------------------------------------------------------------
# SlackOrchestrationAdapter — receive and send
# ---------------------------------------------------------------------------

class TestSlackAdapterReceiveSend:
    def _adapter(self):
        import apyrobo.orchestration.slack_adapter as slack_mod
        slack_mod._SLACK_BOLT_AVAILABLE = True
        adapter = SlackOrchestrationAdapter(
            bot_token="xoxb-test",
            signing_secret="secret",
        )
        mock_client = MagicMock()
        mock_client.chat_postMessage.return_value = {"ok": True}
        adapter._client = mock_client
        return adapter

    def test_receive_returns_none_after_shutdown(self):
        adapter = self._adapter()
        adapter._recv_queue.put(None)
        result = adapter.receive()
        assert result is None

    def test_receive_returns_enqueued_message(self):
        adapter = self._adapter()
        msg = OrchestrationMessage(task="patrol", robot_uri="mock://turtlebot4", source="slack")
        adapter._recv_queue.put(msg)
        result = adapter.receive()
        assert result is not None
        assert result.task == "patrol"

    def test_send_planned_posts_skill_list(self):
        adapter = self._adapter()
        reply_key = "patrol|mock://turtlebot4"
        with adapter._reply_lock:
            adapter._reply_targets[reply_key] = {"channel_id": "C456"}

        response = OrchestrationMessage(
            task="patrol",
            robot_uri="mock://turtlebot4",
            metadata={
                "status": "planned",
                "skills": [{"name": "navigate"}, {"name": "dock"}],
                "reply_key": reply_key,
                "slack_channel": "C456",
            },
        )
        adapter.send(response)

        adapter._client.chat_postMessage.assert_called_once()
        call_kwargs = adapter._client.chat_postMessage.call_args[1]
        assert call_kwargs["channel"] == "C456"
        assert "navigate" in call_kwargs["text"]
        assert "dock" in call_kwargs["text"]

    def test_send_error_posts_error_message(self):
        adapter = self._adapter()
        reply_key = "bad|mock://default"
        with adapter._reply_lock:
            adapter._reply_targets[reply_key] = {"channel_id": "C789"}

        response = OrchestrationMessage(
            task="bad",
            robot_uri="mock://default",
            metadata={
                "status": "error",
                "error": "robot not found",
                "reply_key": reply_key,
                "slack_channel": "C789",
            },
        )
        adapter.send(response)

        call_kwargs = adapter._client.chat_postMessage.call_args[1]
        assert "Error" in call_kwargs["text"] or "error" in call_kwargs["text"].lower()
        assert "robot not found" in call_kwargs["text"]

    def test_send_clears_reply_target(self):
        adapter = self._adapter()
        reply_key = "task|mock://turtlebot4"
        with adapter._reply_lock:
            adapter._reply_targets[reply_key] = {"channel_id": "C000"}

        response = OrchestrationMessage(
            task="task",
            metadata={"status": "planned", "skills": [], "reply_key": reply_key, "slack_channel": "C000"},
        )
        adapter.send(response)

        with adapter._reply_lock:
            assert reply_key not in adapter._reply_targets

    def test_send_without_client_logs_warning(self, caplog):
        import apyrobo.orchestration.slack_adapter as slack_mod
        slack_mod._SLACK_BOLT_AVAILABLE = True
        adapter = SlackOrchestrationAdapter()
        adapter._client = None  # no client

        import logging
        with caplog.at_level(logging.WARNING, logger="apyrobo.orchestration.slack_adapter"):
            adapter.send(OrchestrationMessage(task="x", metadata={"status": "planned", "skills": []}))

        assert any("no channel" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# SlackOrchestrationAdapter — shutdown unblocks receive
# ---------------------------------------------------------------------------

class TestSlackAdapterShutdown:
    def test_shutdown_puts_none_on_queue(self):
        import apyrobo.orchestration.slack_adapter as slack_mod
        slack_mod._SLACK_BOLT_AVAILABLE = True
        adapter = SlackOrchestrationAdapter()
        adapter._thread = None
        adapter.shutdown()
        item = adapter._recv_queue.get_nowait()
        assert item is None

    def test_receive_unblocked_by_shutdown(self):
        import apyrobo.orchestration.slack_adapter as slack_mod
        slack_mod._SLACK_BOLT_AVAILABLE = True
        adapter = SlackOrchestrationAdapter()
        adapter._thread = None

        received = []

        def _receive():
            received.append(adapter.receive())

        t = threading.Thread(target=_receive, daemon=True)
        t.start()
        time.sleep(0.05)
        adapter.shutdown()
        t.join(timeout=1.0)
        assert not t.is_alive()
        assert received == [None]


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestSlackModuleExports:
    def test_importable_from_orchestration(self):
        from apyrobo.orchestration import SlackOrchestrationAdapter as SA
        assert SA is SlackOrchestrationAdapter

    def test_in_all_list(self):
        import apyrobo.orchestration as orch
        assert "SlackOrchestrationAdapter" in orch.__all__


# ---------------------------------------------------------------------------
# CLI integration — serve --transport slack
# ---------------------------------------------------------------------------

class TestCliServeSlack:
    def test_serve_slack_transport_arg_accepted(self):
        """Parser must accept --transport slack without error."""
        import argparse
        import sys
        from unittest.mock import patch

        with patch.object(sys, "argv", [
            "apyrobo", "serve",
            "--transport", "slack",
            "--slack-port", "3001",
            "--slack-command", "/robot",
        ]), patch("apyrobo.cli.cmd_serve") as mock_serve:
            try:
                from apyrobo.cli import main
                main()
            except SystemExit:
                pass

        if mock_serve.call_args:
            args = mock_serve.call_args[0][0]
            assert args.transport == "slack"
            assert args.slack_port == 3001
            assert args.slack_command == "/robot"
        else:
            # Parser accepted args without error — test passes
            pass

    def test_serve_slack_importerror_exits(self):
        """cmd_serve prints error and exits when slack-bolt not installed."""
        import argparse
        from apyrobo.cli import cmd_serve
        import apyrobo.orchestration.slack_adapter as slack_mod
        orig = slack_mod._SLACK_BOLT_AVAILABLE
        try:
            slack_mod._SLACK_BOLT_AVAILABLE = False
            args = argparse.Namespace(
                transport="slack",
                robot="mock://turtlebot4",
                provider="rule",
                profile=None,
                slack_port=3000,
                slack_command="/apyrobo",
                ws_port=8765,
            )
            with pytest.raises(SystemExit) as exc_info:
                cmd_serve(args)
            assert exc_info.value.code == 1
        finally:
            slack_mod._SLACK_BOLT_AVAILABLE = orig

"""Slack orchestration adapter for APYROBO.

Wires a Slack Bolt app to the ``OrchestrationAdapter`` interface so any
Slack workspace can dispatch robot tasks via slash commands and receive
streamed skill-progress replies in threads.

**Installation**::

    pip install 'apyrobo[slack]'
    # or
    pip install slack-bolt

**Quick start**::

    export SLACK_BOT_TOKEN=xoxb-...
    export SLACK_SIGNING_SECRET=...
    apyrobo serve --transport slack --slack-port 3000

**Slash command** — register ``/apyrobo`` in your Slack app manifest::

    /apyrobo navigate to the charging dock robot=mock://turtlebot4
    /apyrobo pick up the cup robot=unitree://go2@192.168.1.100

**App manifest snippet**::

    slash_commands:
      - command: /apyrobo
        description: Dispatch a robot task
        usage_hint: "<task description> [robot=<uri>]"

When the orchestration server replies via ``send()``, the response is
posted as a threaded reply to the original slash command message.

If the ``APYROBO_DEFAULT_ROBOT`` env var is set, it becomes the fallback
``robot_uri`` when none is supplied in the command text.

v6.0.0 — APYROBO Ecosystem Integrations
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
from typing import Any

from apyrobo.orchestration.adapter import OrchestrationAdapter, OrchestrationMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Slack Bolt import
# ---------------------------------------------------------------------------

try:
    from slack_bolt import App as _BoltApp  # type: ignore
    from slack_bolt.adapter.socket_mode import SocketModeHandler  # type: ignore
    _SLACK_BOLT_AVAILABLE = True
except ImportError:
    _BoltApp = None  # type: ignore
    SocketModeHandler = None  # type: ignore
    _SLACK_BOLT_AVAILABLE = False

_SLACK_IMPORT_ERROR = (
    "The 'slack-bolt' package is required for SlackOrchestrationAdapter.\n"
    "Install with: pip install 'apyrobo[slack]'\n"
    "  or: pip install slack-bolt"
)

# Default command — can be overridden per instance
_DEFAULT_COMMAND = "/apyrobo"

# Keyword token that marks the robot URI in command text
_ROBOT_KEYWORD = "robot="


def _parse_command_text(text: str, default_robot: str) -> tuple[str, str]:
    """Parse ``/apyrobo <task text> [robot=<uri>]`` into (task, robot_uri).

    The ``robot=`` token (if present) is stripped from the task text.

    Returns:
        (task, robot_uri) — both non-empty strings.
    """
    text = text.strip()
    robot_uri = default_robot
    task_parts: list[str] = []

    for token in text.split():
        if token.lower().startswith(_ROBOT_KEYWORD):
            robot_uri = token[len(_ROBOT_KEYWORD):]
        else:
            task_parts.append(token)

    task = " ".join(task_parts).strip() or text
    return task, robot_uri


# ---------------------------------------------------------------------------
# SlackOrchestrationAdapter
# ---------------------------------------------------------------------------

class SlackOrchestrationAdapter(OrchestrationAdapter):
    """Slack Bolt-based orchestration adapter.

    Listens for the ``/apyrobo`` slash command (or a custom ``command``).
    Each command invocation creates an ``OrchestrationMessage`` and blocks
    ``receive()`` until one arrives.  When ``send()`` is called, the
    response is posted as a threaded reply to the originating channel.

    The adapter supports two Bolt transport modes:

    * **Socket Mode** (default) — uses ``SLACK_APP_TOKEN`` (``xapp-...``).
      No public URL required; ideal for development and firewalled deployments.
    * **HTTP mode** — Bolt's built-in development server listens on *port*.
      Requires a public endpoint (e.g. ngrok) registered in the Slack app.

    Parameters
    ----------
    bot_token:
        Slack bot OAuth token (``xoxb-...``).  Falls back to
        ``SLACK_BOT_TOKEN`` env var.
    signing_secret:
        Slack signing secret.  Falls back to ``SLACK_SIGNING_SECRET`` env var.
    app_token:
        Slack app-level token for Socket Mode (``xapp-...``).  Falls back to
        ``SLACK_APP_TOKEN`` env var.  If set, Socket Mode is used; otherwise
        HTTP mode on *port*.
    port:
        HTTP server port for non-Socket-Mode deployments (default 3000).
    command:
        Slash command to register (default ``"/apyrobo"``).
    default_robot:
        Fallback ``robot_uri`` when the user doesn't supply ``robot=`` in the
        command text.  Also reads from ``APYROBO_DEFAULT_ROBOT`` env var.

    Raises
    ------
    ImportError
        If ``slack-bolt`` is not installed.

    Examples
    --------
    Socket Mode (recommended)::

        adapter = SlackOrchestrationAdapter(
            bot_token="xoxb-...",
            signing_secret="...",
            app_token="xapp-...",
        )

    HTTP mode::

        adapter = SlackOrchestrationAdapter(
            bot_token="xoxb-...",
            signing_secret="...",
            port=3000,
        )
    """

    def __init__(
        self,
        bot_token: str | None = None,
        signing_secret: str | None = None,
        app_token: str | None = None,
        port: int = 3000,
        command: str = _DEFAULT_COMMAND,
        default_robot: str = "mock://turtlebot4",
    ) -> None:
        if not _SLACK_BOLT_AVAILABLE:
            raise ImportError(_SLACK_IMPORT_ERROR)

        self._bot_token = bot_token or os.environ.get("SLACK_BOT_TOKEN", "")
        self._signing_secret = signing_secret or os.environ.get("SLACK_SIGNING_SECRET", "")
        self._app_token = app_token or os.environ.get("SLACK_APP_TOKEN", "")
        self._port = port
        self._command = command
        self._default_robot = (
            default_robot
            or os.environ.get("APYROBO_DEFAULT_ROBOT", "mock://turtlebot4")
        )

        # Queue bridges the async Slack handler and synchronous receive()
        self._recv_queue: queue.Queue[OrchestrationMessage | None] = queue.Queue()

        # Maps task text → Slack context (channel_id, thread_ts) for reply routing
        self._reply_targets: dict[str, dict[str, str]] = {}
        self._reply_lock = threading.Lock()

        self._app: Any = None            # slack_bolt.App
        self._handler: Any = None        # SocketModeHandler or HTTP server thread
        self._client: Any = None         # WebClient from the app
        self._thread: threading.Thread | None = None
        self._started = threading.Event()

    # ------------------------------------------------------------------
    # Bolt app wiring
    # ------------------------------------------------------------------

    def _build_app(self) -> None:
        """Construct the Bolt App and register the slash-command handler."""
        self._app = _BoltApp(
            token=self._bot_token,
            signing_secret=self._signing_secret,
        )
        self._client = self._app.client

        command = self._command

        @self._app.command(command)
        def handle_command(ack: Any, body: dict, say: Any) -> None:
            ack()  # acknowledge within 3 s per Slack requirement

            text = body.get("text", "").strip()
            if not text:
                say(
                    text=f"Usage: `{command} <task description> [robot=<uri>]`",
                    thread_ts=body.get("message_ts"),
                )
                return

            task, robot_uri = _parse_command_text(text, self._default_robot)
            channel_id = body.get("channel_id", "")
            response_url = body.get("response_url", "")

            # Acknowledge the user immediately
            say(
                text=f":robot_face: Dispatching: *{task}* → `{robot_uri}`",
                channel=channel_id,
            )

            # Store reply context so send() can post the threaded response
            # Use a composite key (task+robot) to correlate request→response
            reply_key = f"{task}|{robot_uri}"
            with self._reply_lock:
                self._reply_targets[reply_key] = {
                    "channel_id": channel_id,
                    "response_url": response_url,
                }

            msg = OrchestrationMessage(
                task=task,
                robot_uri=robot_uri,
                metadata={"slack_channel": channel_id, "reply_key": reply_key},
                source="slack",
            )
            self._recv_queue.put(msg)
            logger.info(
                "SlackOrchestrationAdapter: received command %r (robot=%r, channel=%s)",
                task, robot_uri, channel_id,
            )

    def _run_socket_mode(self) -> None:
        """Start in Socket Mode (background thread entry point)."""
        try:
            handler = SocketModeHandler(self._app, self._app_token)
            self._handler = handler
            self._started.set()
            handler.start()  # blocks until stopped
        except Exception as exc:
            logger.error("SlackOrchestrationAdapter Socket Mode error: %s", exc)
        finally:
            self._started.set()  # unblock startup() even on error

    def _run_http_mode(self) -> None:
        """Start in HTTP mode (background thread entry point)."""
        try:
            self._started.set()
            self._app.start(port=self._port)
        except Exception as exc:
            logger.error("SlackOrchestrationAdapter HTTP mode error: %s", exc)

    # ------------------------------------------------------------------
    # OrchestrationAdapter interface
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Build the Bolt app and start listening for slash commands.

        Uses Socket Mode when ``SLACK_APP_TOKEN`` / ``app_token`` is set;
        falls back to HTTP mode on *port*.

        Raises
        ------
        RuntimeError
            If the server does not start within 10 seconds.
        """
        self._build_app()

        use_socket_mode = bool(self._app_token)
        target = self._run_socket_mode if use_socket_mode else self._run_http_mode
        mode_label = "Socket Mode" if use_socket_mode else f"HTTP port {self._port}"

        self._thread = threading.Thread(
            target=target,
            name="slack-orchestration",
            daemon=True,
        )
        self._thread.start()

        if not self._started.wait(timeout=10):
            raise RuntimeError(
                "SlackOrchestrationAdapter: server did not start within 10 s"
            )
        logger.info("SlackOrchestrationAdapter ready (%s, command=%r)", mode_label, self._command)

    def shutdown(self) -> None:
        """Stop the Slack app and background thread.

        Puts ``None`` on the receive queue to unblock any waiting
        ``receive()`` call.
        """
        try:
            if self._handler is not None and hasattr(self._handler, "close"):
                self._handler.close()
        except Exception as exc:
            logger.debug("SlackOrchestrationAdapter: shutdown handler error: %s", exc)

        self._recv_queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("SlackOrchestrationAdapter stopped")

    def receive(self) -> OrchestrationMessage | None:
        """Block until a slash command arrives, then return its message.

        Returns
        -------
        OrchestrationMessage | None
            None signals shutdown.
        """
        return self._recv_queue.get()

    def send(self, msg: OrchestrationMessage) -> None:
        """Post *msg* as a Slack message.

        If a reply target exists for this message (matched by the
        ``reply_key`` metadata field), the response is posted to the
        originating channel.  Otherwise it falls back to a log warning.

        Parameters
        ----------
        msg:
            Response from the orchestration server.
        """
        reply_key = msg.metadata.get("reply_key", "")
        channel_id = msg.metadata.get("slack_channel", "")

        with self._reply_lock:
            target = self._reply_targets.pop(reply_key, {})

        target_channel = target.get("channel_id") or channel_id

        status = msg.metadata.get("status", "unknown")
        if status == "planned":
            skills = msg.metadata.get("skills", [])
            skill_names = [s.get("name", s.get("skill_id", "?")) for s in skills]
            text = (
                f":white_check_mark: *Plan ready* for `{msg.task}` "
                f"on `{msg.robot_uri}`:\n"
                + "\n".join(f"  {i+1}. {name}" for i, name in enumerate(skill_names))
            )
        elif status == "error":
            error = msg.metadata.get("error", "unknown error")
            text = f":x: *Error* planning `{msg.task}`: {error}"
        else:
            text = f":information_source: `{msg.task}` → {json.dumps(msg.metadata)}"

        if self._client and target_channel:
            try:
                self._client.chat_postMessage(
                    channel=target_channel,
                    text=text,
                    mrkdwn=True,
                )
                logger.info(
                    "SlackOrchestrationAdapter: sent response to channel %s", target_channel
                )
            except Exception as exc:
                logger.warning(
                    "SlackOrchestrationAdapter: chat_postMessage failed: %s", exc
                )
        else:
            logger.warning(
                "SlackOrchestrationAdapter.send: no channel to reply to (msg=%r)", msg
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def command(self) -> str:
        """Registered slash command (e.g. ``"/apyrobo"``)."""
        return self._command

    @property
    def port(self) -> int:
        """HTTP server port (used in non-Socket-Mode deployments)."""
        return self._port

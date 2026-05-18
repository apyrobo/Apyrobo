"""Tests for WebSocketOrchestrationAdapter.

These tests cover:
- Instantiation (with and without the websockets library)
- startup / shutdown lifecycle
- Receiving a JSON message sent from a WebSocket client
- Broadcast to multiple connected clients via send()
- MockOrchestrationAdapter startup_called / shutdown_called semantics
  (as a sanity reference, also tested in test_orchestration.py)

The tests use plain threading (no pytest-asyncio required) so they work
whether or not pytest-asyncio is installed, and they avoid flaky timing by
using threading.Event / queue.Queue synchronisation primitives.
"""
from __future__ import annotations

import json
import queue
import socket
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from apyrobo.orchestration.adapter import (
    MockOrchestrationAdapter,
    OrchestrationMessage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WEBSOCKETS_AVAILABLE: bool
try:
    import websockets  # noqa: F401
    _WEBSOCKETS_AVAILABLE = True
except ImportError:
    _WEBSOCKETS_AVAILABLE = False

ws_only = pytest.mark.skipif(
    not _WEBSOCKETS_AVAILABLE,
    reason="websockets library not installed",
)


def _free_port() -> int:
    """Return a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ws_send_and_collect(
    port: int,
    payload: str,
    *,
    collect_responses: int = 0,
    timeout: float = 5.0,
) -> list[str]:
    """Open a WebSocket connection, send *payload*, collect *collect_responses*
    responses, then close.  Returns the collected response strings."""
    import asyncio

    responses: list[str] = []

    async def _run() -> None:
        uri = f"ws://127.0.0.1:{port}"
        import websockets.client
        async with websockets.client.connect(uri) as ws:
            await ws.send(payload)
            for _ in range(collect_responses):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    responses.append(msg)
                except asyncio.TimeoutError:
                    break

    asyncio.run(_run())
    return responses


# ---------------------------------------------------------------------------
# Tests — import / instantiation
# ---------------------------------------------------------------------------

class TestWebSocketAdapterImport:
    """Verify ImportError behaviour when websockets is absent."""

    def test_import_error_without_websockets(self) -> None:
        """When websockets is not importable, constructing the adapter raises ImportError."""
        import sys
        import importlib

        # Temporarily hide the websockets package.
        original = sys.modules.get("websockets")
        sys.modules["websockets"] = None  # type: ignore[assignment]

        # Also patch the module-level flag in adapter.py.
        import apyrobo.orchestration.adapter as _mod
        original_flag = _mod._WEBSOCKETS_AVAILABLE
        _mod._WEBSOCKETS_AVAILABLE = False

        try:
            with pytest.raises(ImportError, match="pip install"):
                from apyrobo.orchestration.adapter import WebSocketOrchestrationAdapter
                WebSocketOrchestrationAdapter()
        finally:
            if original is None:
                sys.modules.pop("websockets", None)
            else:
                sys.modules["websockets"] = original
            _mod._WEBSOCKETS_AVAILABLE = original_flag

    @ws_only
    def test_instantiation_defaults(self) -> None:
        """WebSocketOrchestrationAdapter can be instantiated with default args."""
        from apyrobo.orchestration.adapter import WebSocketOrchestrationAdapter

        adapter = WebSocketOrchestrationAdapter()
        assert adapter._host == "0.0.0.0"
        assert adapter._port == 8765

    @ws_only
    def test_instantiation_custom(self) -> None:
        """Custom host/port are stored correctly."""
        from apyrobo.orchestration.adapter import WebSocketOrchestrationAdapter

        adapter = WebSocketOrchestrationAdapter(host="127.0.0.1", port=9999)
        assert adapter._host == "127.0.0.1"
        assert adapter._port == 9999


# ---------------------------------------------------------------------------
# Tests — lifecycle (startup / shutdown)
# ---------------------------------------------------------------------------

@ws_only
class TestWebSocketAdapterLifecycle:

    def test_startup_and_shutdown(self) -> None:
        """Adapter starts and stops without error."""
        from apyrobo.orchestration.adapter import WebSocketOrchestrationAdapter

        port = _free_port()
        adapter = WebSocketOrchestrationAdapter(host="127.0.0.1", port=port)
        adapter.startup()

        # Server should be listening — connect a plain socket.
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass  # connection succeeded

        adapter.shutdown()
        # After shutdown the thread should have ended.
        if adapter._thread is not None:
            adapter._thread.join(timeout=3)
        assert adapter._thread is None or not adapter._thread.is_alive()

    def test_double_shutdown_is_safe(self) -> None:
        """Calling shutdown twice does not raise."""
        from apyrobo.orchestration.adapter import WebSocketOrchestrationAdapter

        port = _free_port()
        adapter = WebSocketOrchestrationAdapter(host="127.0.0.1", port=port)
        adapter.startup()
        adapter.shutdown()
        adapter.shutdown()  # second call — must not raise


# ---------------------------------------------------------------------------
# Tests — receive()
# ---------------------------------------------------------------------------

@ws_only
class TestWebSocketAdapterReceive:

    def test_receive_single_message(self) -> None:
        """A JSON message sent from a client is returned by receive()."""
        from apyrobo.orchestration.adapter import WebSocketOrchestrationAdapter

        port = _free_port()
        adapter = WebSocketOrchestrationAdapter(host="127.0.0.1", port=port)
        adapter.startup()

        payload = json.dumps({"task": "go home", "robot_uri": "mock://turtle"})

        # Send the message from a background thread so receive() can block.
        def _send() -> None:
            time.sleep(0.1)  # let receive() block first
            _ws_send_and_collect(port, payload)

        t = threading.Thread(target=_send, daemon=True)
        t.start()

        msg = adapter.receive()
        t.join(timeout=5)
        adapter.shutdown()

        assert msg is not None
        assert msg.task == "go home"
        assert msg.robot_uri == "mock://turtle"

    def test_receive_multiple_messages(self) -> None:
        """Multiple sequential messages are all delivered via receive()."""
        from apyrobo.orchestration.adapter import WebSocketOrchestrationAdapter

        port = _free_port()
        adapter = WebSocketOrchestrationAdapter(host="127.0.0.1", port=port)
        adapter.startup()

        tasks = ["navigate to A", "navigate to B", "dock"]
        payloads = [json.dumps({"task": t}) for t in tasks]

        received: list[OrchestrationMessage] = []
        done = threading.Event()

        def _collector() -> None:
            for _ in tasks:
                m = adapter.receive()
                if m is None:
                    break
                received.append(m)
            done.set()

        threading.Thread(target=_collector, daemon=True).start()

        # Send all messages from a WS client.
        import asyncio

        async def _send_all() -> None:
            import websockets.client
            uri = f"ws://127.0.0.1:{port}"
            async with websockets.client.connect(uri) as ws:
                for p in payloads:
                    await ws.send(p)
                    await asyncio.sleep(0.05)

        asyncio.run(_send_all())
        done.wait(timeout=5)
        adapter.shutdown()

        assert [m.task for m in received] == tasks

    def test_receive_raw_text_fallback(self) -> None:
        """Non-JSON text is wrapped as a task with the raw string."""
        from apyrobo.orchestration.adapter import WebSocketOrchestrationAdapter

        port = _free_port()
        adapter = WebSocketOrchestrationAdapter(host="127.0.0.1", port=port)
        adapter.startup()

        def _send() -> None:
            time.sleep(0.05)
            _ws_send_and_collect(port, "this is not json")

        threading.Thread(target=_send, daemon=True).start()

        msg = adapter.receive()
        adapter.shutdown()

        assert msg is not None
        assert msg.task == "this is not json"

    def test_shutdown_unblocks_receive(self) -> None:
        """shutdown() causes a blocking receive() to return None."""
        from apyrobo.orchestration.adapter import WebSocketOrchestrationAdapter

        port = _free_port()
        adapter = WebSocketOrchestrationAdapter(host="127.0.0.1", port=port)
        adapter.startup()

        result: list[OrchestrationMessage | None] = []

        def _recv() -> None:
            result.append(adapter.receive())

        t = threading.Thread(target=_recv, daemon=True)
        t.start()
        time.sleep(0.1)
        adapter.shutdown()
        t.join(timeout=3)

        assert result == [None]


# ---------------------------------------------------------------------------
# Tests — send() / broadcast
# ---------------------------------------------------------------------------

@ws_only
class TestWebSocketAdapterSend:

    def test_send_broadcasts_to_connected_client(self) -> None:
        """send() delivers the response JSON to a connected client."""
        from apyrobo.orchestration.adapter import WebSocketOrchestrationAdapter

        port = _free_port()
        adapter = WebSocketOrchestrationAdapter(host="127.0.0.1", port=port)
        adapter.startup()

        collected: list[str] = []
        client_ready = threading.Event()
        client_done = threading.Event()

        import asyncio

        async def _client() -> None:
            import websockets.client
            uri = f"ws://127.0.0.1:{port}"
            async with websockets.client.connect(uri) as ws:
                # Signal that we are connected and listening.
                client_ready.set()
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    collected.append(raw)
                except asyncio.TimeoutError:
                    pass
            client_done.set()

        def _run_client() -> None:
            asyncio.run(_client())

        threading.Thread(target=_run_client, daemon=True).start()
        client_ready.wait(timeout=5)
        time.sleep(0.05)  # let the client settle into recv()

        response = OrchestrationMessage(
            task="go home",
            robot_uri="mock://turtle",
            metadata={"status": "planned"},
            source="test",
        )
        adapter.send(response)
        client_done.wait(timeout=5)
        adapter.shutdown()

        assert len(collected) == 1
        data = json.loads(collected[0])
        assert data["task"] == "go home"
        assert data["metadata"]["status"] == "planned"

    def test_send_broadcasts_to_multiple_clients(self) -> None:
        """send() delivers the message to all connected clients."""
        from apyrobo.orchestration.adapter import WebSocketOrchestrationAdapter

        port = _free_port()
        adapter = WebSocketOrchestrationAdapter(host="127.0.0.1", port=port)
        adapter.startup()

        n_clients = 3
        all_ready = threading.Barrier(n_clients + 1)  # clients + main thread
        results: list[list[str]] = [[] for _ in range(n_clients)]
        done_events = [threading.Event() for _ in range(n_clients)]

        import asyncio

        def _make_client(idx: int) -> None:
            async def _run() -> None:
                import websockets.client
                uri = f"ws://127.0.0.1:{port}"
                async with websockets.client.connect(uri) as ws:
                    all_ready.wait()
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        results[idx].append(raw)
                    except asyncio.TimeoutError:
                        pass
                done_events[idx].set()

            asyncio.run(_run())

        threads = [
            threading.Thread(target=_make_client, args=(i,), daemon=True)
            for i in range(n_clients)
        ]
        for t in threads:
            t.start()

        all_ready.wait()
        time.sleep(0.1)  # give clients a moment to enter recv()

        response = OrchestrationMessage(task="broadcast test")
        adapter.send(response)

        for ev in done_events:
            ev.wait(timeout=5)
        adapter.shutdown()

        for i, r in enumerate(results):
            assert len(r) == 1, f"client {i} got {len(r)} messages, expected 1"
            data = json.loads(r[0])
            assert data["task"] == "broadcast test"

    def test_send_with_no_clients_does_not_raise(self) -> None:
        """send() is a no-op when no clients are connected."""
        from apyrobo.orchestration.adapter import WebSocketOrchestrationAdapter

        port = _free_port()
        adapter = WebSocketOrchestrationAdapter(host="127.0.0.1", port=port)
        adapter.startup()

        # No client connected — must not raise.
        adapter.send(OrchestrationMessage(task="nobody home"))
        adapter.shutdown()


# ---------------------------------------------------------------------------
# Tests — MockOrchestrationAdapter startup_called / shutdown_called
# ---------------------------------------------------------------------------

class TestMockAdapterSemantics:
    """Verify the startup_called / shutdown_called contract on MockAdapter."""

    def test_startup_shutdown_flags(self) -> None:
        adapter = MockOrchestrationAdapter(tasks=["a"])
        assert not adapter.startup_called
        assert not adapter.shutdown_called

        adapter.startup()
        assert adapter.startup_called
        assert not adapter.shutdown_called

        adapter.shutdown()
        assert adapter.startup_called
        assert adapter.shutdown_called

    def test_receive_exhausts_queue(self) -> None:
        adapter = MockOrchestrationAdapter(tasks=["x", "y"])
        assert adapter.receive().task == "x"
        assert adapter.receive().task == "y"
        assert adapter.receive() is None

    def test_send_accumulates(self) -> None:
        adapter = MockOrchestrationAdapter()
        m1 = OrchestrationMessage(task="t1")
        m2 = OrchestrationMessage(task="t2")
        adapter.send(m1)
        adapter.send(m2)
        assert [m.task for m in adapter.sent] == ["t1", "t2"]

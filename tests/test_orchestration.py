"""Tests for orchestration adapter base."""
from __future__ import annotations

import io
import json
import sys
import subprocess

import pytest

from apyrobo.orchestration import (
    OrchestrationAdapter,
    OrchestrationServer,
    StdioOrchestrationAdapter,
    MockOrchestrationAdapter,
    OrchestrationMessage,
)


# ---------------------------------------------------------------------------
# OrchestrationMessage
# ---------------------------------------------------------------------------

class TestOrchestrationMessage:
    def test_defaults(self):
        msg = OrchestrationMessage(task="go to dock")
        assert msg.task == "go to dock"
        # Empty robot_uri means "server's default robot" and is omitted from
        # the wire form so the message stays schema-valid.
        assert msg.robot_uri == ""
        assert "robot_uri" not in msg.to_dict()
        assert msg.metadata == {}
        assert msg.source == ""

    def test_to_dict(self):
        msg = OrchestrationMessage(task="patrol", robot_uri="mock://spot")
        d = msg.to_dict()
        assert d["task"] == "patrol"
        assert d["robot_uri"] == "mock://spot"
        assert "metadata" in d

    def test_from_dict(self):
        d = {"task": "pick up box", "robot_uri": "mock://ur5", "source": "slack"}
        msg = OrchestrationMessage.from_dict(d)
        assert msg.task == "pick up box"
        assert msg.robot_uri == "mock://ur5"
        assert msg.source == "slack"

    def test_from_dict_defaults(self):
        msg = OrchestrationMessage.from_dict({"task": "hello"})
        assert msg.robot_uri == ""

    def test_round_trip(self):
        original = OrchestrationMessage(
            task="move to shelf A",
            robot_uri="mock://mir100",
            metadata={"priority": 1},
            source="ros_topic",
        )
        d = original.to_dict()
        restored = OrchestrationMessage.from_dict(d)
        assert restored.task == original.task
        assert restored.robot_uri == original.robot_uri
        assert restored.metadata == original.metadata


# ---------------------------------------------------------------------------
# MockOrchestrationAdapter
# ---------------------------------------------------------------------------

class TestMockOrchestrationAdapter:
    def test_receive_returns_tasks_in_order(self):
        adapter = MockOrchestrationAdapter(tasks=["task1", "task2"])
        msg1 = adapter.receive()
        msg2 = adapter.receive()
        assert msg1.task == "task1"
        assert msg2.task == "task2"

    def test_receive_returns_none_when_empty(self):
        adapter = MockOrchestrationAdapter(tasks=[])
        assert adapter.receive() is None

    def test_receive_exhausted_returns_none(self):
        adapter = MockOrchestrationAdapter(tasks=["only"])
        adapter.receive()
        assert adapter.receive() is None

    def test_send_accumulates(self):
        adapter = MockOrchestrationAdapter()
        adapter.send(OrchestrationMessage(task="response1"))
        adapter.send(OrchestrationMessage(task="response2"))
        assert len(adapter.sent) == 2

    def test_startup_sets_flag(self):
        adapter = MockOrchestrationAdapter()
        assert not adapter.startup_called
        adapter.startup()
        assert adapter.startup_called

    def test_shutdown_sets_flag(self):
        adapter = MockOrchestrationAdapter()
        adapter.shutdown()
        assert adapter.shutdown_called

    def test_accepts_message_objects(self):
        msg = OrchestrationMessage(task="from object", robot_uri="mock://spot")
        adapter = MockOrchestrationAdapter(tasks=[msg])
        received = adapter.receive()
        assert received.robot_uri == "mock://spot"


# ---------------------------------------------------------------------------
# StdioOrchestrationAdapter
# ---------------------------------------------------------------------------

class TestStdioOrchestrationAdapter:
    def _adapter_from_text(self, text: str) -> StdioOrchestrationAdapter:
        return StdioOrchestrationAdapter(infile=io.StringIO(text), outfile=io.StringIO())

    def test_receive_valid_json(self):
        line = json.dumps({"task": "navigate to dock"}) + "\n"
        adapter = self._adapter_from_text(line)
        msg = adapter.receive()
        assert msg is not None
        assert msg.task == "navigate to dock"

    def test_receive_eof_returns_none(self):
        adapter = self._adapter_from_text("")
        assert adapter.receive() is None

    def test_receive_plain_text_as_task(self):
        adapter = self._adapter_from_text("go to kitchen\n")
        msg = adapter.receive()
        assert msg is not None
        assert msg.task == "go to kitchen"

    def test_send_writes_json_line(self):
        out = io.StringIO()
        adapter = StdioOrchestrationAdapter(infile=io.StringIO(), outfile=out)
        adapter.send(OrchestrationMessage(task="result"))
        out.seek(0)
        data = json.loads(out.read().strip())
        assert data["task"] == "result"

    def test_receive_multiple_lines(self):
        text = (
            json.dumps({"task": "task A"}) + "\n"
            + json.dumps({"task": "task B"}) + "\n"
        )
        adapter = self._adapter_from_text(text)
        msgs = [adapter.receive(), adapter.receive()]
        assert msgs[0].task == "task A"
        assert msgs[1].task == "task B"


# ---------------------------------------------------------------------------
# OrchestrationServer
# ---------------------------------------------------------------------------

class TestOrchestrationServer:
    def _make_agent(self):
        from apyrobo.skills.agent import Agent
        return Agent(provider="rule")

    def test_run_with_empty_adapter(self):
        adapter = MockOrchestrationAdapter(tasks=[])
        agent = self._make_agent()
        server = OrchestrationServer(adapter, agent)
        server.run()  # should return immediately
        assert adapter.startup_called
        assert adapter.shutdown_called

    def test_run_processes_tasks(self):
        adapter = MockOrchestrationAdapter(tasks=["move forward", "stop"])
        agent = self._make_agent()
        server = OrchestrationServer(adapter, agent)
        server.run()
        assert len(adapter.sent) == 2

    def test_response_has_status(self):
        adapter = MockOrchestrationAdapter(tasks=["navigate to dock"])
        agent = self._make_agent()
        server = OrchestrationServer(adapter, agent)
        server.run()
        response = adapter.sent[0]
        assert "status" in response.metadata

    def test_max_iterations_respected(self):
        tasks = ["task1", "task2", "task3", "task4"]
        adapter = MockOrchestrationAdapter(tasks=tasks)
        agent = self._make_agent()
        server = OrchestrationServer(adapter, agent, max_iterations=2)
        server.run()
        assert len(adapter.sent) == 2

    def test_iterations_counter(self):
        adapter = MockOrchestrationAdapter(tasks=["a", "b", "c"])
        agent = self._make_agent()
        server = OrchestrationServer(adapter, agent)
        server.run()
        assert server._iterations == 3

    def test_error_in_handle_returns_error_response(self):
        adapter = MockOrchestrationAdapter(tasks=["deliberately invalid 🤖"])
        agent = self._make_agent()
        server = OrchestrationServer(adapter, agent, default_robot=None)
        server.run()
        # Should not raise; response has status planned or error
        assert len(adapter.sent) == 1

    def test_source_is_orchestration_server(self):
        adapter = MockOrchestrationAdapter(tasks=["move forward"])
        agent = self._make_agent()
        server = OrchestrationServer(adapter, agent)
        server.run()
        assert adapter.sent[0].source == "orchestration_server"


# ---------------------------------------------------------------------------
# Execute mode (serve --execute)
# ---------------------------------------------------------------------------

class TestExecuteMode:
    def _make_agent(self):
        from apyrobo.skills.agent import Agent
        return Agent(provider="rule")

    def test_plan_only_by_default(self):
        adapter = MockOrchestrationAdapter(tasks=["go to (2, 3)"])
        server = OrchestrationServer(adapter, self._make_agent())
        server.run()
        assert "execution" not in adapter.sent[0].metadata

    def test_execute_reports_completion_and_moves_robot(self):
        from apyrobo.core.robot import Robot
        robot = Robot.discover("mock://exec_bot")
        adapter = MockOrchestrationAdapter(tasks=["go to (2, 3)"])
        server = OrchestrationServer(
            adapter, self._make_agent(),
            default_robot=robot, default_robot_uri="mock://exec_bot",
            execute_tasks=True,
        )
        server.run()
        response = adapter.sent[0]
        assert response.metadata["status"] == "planned"
        execution = response.metadata["execution"]
        assert execution["status"] == "completed"
        assert execution["steps_completed"] >= 1
        assert robot._adapter.position == (2.0, 3.0)

    def test_execute_response_stays_schema_valid(self):
        jsonschema = pytest.importorskip("jsonschema")
        import pathlib
        schema = json.loads(
            (pathlib.Path(__file__).parent.parent
             / "spec" / "schemas" / "orchestration-message.schema.json").read_text()
        )
        adapter = MockOrchestrationAdapter(tasks=["go to (1, 1)"])
        server = OrchestrationServer(
            adapter, self._make_agent(), execute_tasks=True,
        )
        server.run()
        jsonschema.validate(adapter.sent[0].to_dict(), schema)

    def test_planning_error_skips_execution(self):
        adapter = MockOrchestrationAdapter(
            tasks=[OrchestrationMessage(task="go to (1, 1)", robot_uri="bogus://x")]
        )
        server = OrchestrationServer(
            adapter, self._make_agent(), execute_tasks=True,
        )
        server.run()
        response = adapter.sent[0]
        assert response.metadata["status"] == "error"
        assert "execution" not in response.metadata


# ---------------------------------------------------------------------------
# Robot lifecycle on server shutdown
# ---------------------------------------------------------------------------

class TestServerRobotLifecycle:
    def _make_agent(self):
        from apyrobo.skills.agent import Agent
        return Agent(provider="rule")

    def test_discovered_robot_disconnected_after_run(self):
        adapter = MockOrchestrationAdapter(
            tasks=[OrchestrationMessage(task="move forward", robot_uri="mock://spot")]
        )
        server = OrchestrationServer(adapter, self._make_agent())
        server.run()
        # The server discovered mock://spot itself, so it must release it.
        assert "mock://spot" not in server._robot_cache

    def test_discovered_robot_connection_is_closed(self):
        adapter = MockOrchestrationAdapter(tasks=[])
        server = OrchestrationServer(adapter, self._make_agent())
        robot = server._resolve_robot("mock://spot")
        robot.connect()
        assert robot.is_connected
        server.run()
        assert not robot.is_connected

    def test_default_robot_left_connected(self):
        from apyrobo.core.robot import Robot
        default = Robot.discover("mock://turtlebot4")
        default.connect()
        adapter = MockOrchestrationAdapter(tasks=["move forward"])
        server = OrchestrationServer(adapter, self._make_agent(), default_robot=default)
        server.run()
        # Caller owns the default robot: still connected, still cached.
        assert default.is_connected
        assert server._robot_cache.get("mock://turtlebot4") is default

    def test_failing_disconnect_does_not_block_others(self):
        adapter = MockOrchestrationAdapter(tasks=[])
        server = OrchestrationServer(adapter, self._make_agent())
        bad = server._resolve_robot("mock://bad")
        good = server._resolve_robot("mock://good")
        good.connect()

        def boom():
            raise RuntimeError("disconnect failed")
        bad.disconnect = boom

        server.run()  # must not raise
        assert not good.is_connected
        assert not server._robot_cache

    def test_adapter_shutdown_still_called_when_release_fails(self):
        adapter = MockOrchestrationAdapter(tasks=[])
        server = OrchestrationServer(adapter, self._make_agent())
        robot = server._resolve_robot("mock://spot")

        def boom():
            raise RuntimeError("disconnect failed")
        robot.disconnect = boom

        server.run()
        assert adapter.shutdown_called


# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------

class TestOrchestrationAdapterABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            OrchestrationAdapter()  # type: ignore

    def test_subclass_must_implement_receive_send(self):
        class Incomplete(OrchestrationAdapter):
            pass
        with pytest.raises(TypeError):
            Incomplete()  # type: ignore

    def test_minimal_concrete_subclass(self):
        class Minimal(OrchestrationAdapter):
            def receive(self):
                return None
            def send(self, msg):
                pass
        adapter = Minimal()
        assert adapter.receive() is None


# ---------------------------------------------------------------------------
# CLI serve smoke test
# ---------------------------------------------------------------------------

class TestServeCLI:
    def test_serve_command_exists(self):
        result = subprocess.run(
            [sys.executable, "-m", "apyrobo", "serve", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "serve" in result.stdout or "stdio" in result.stdout.lower() or "robot" in result.stdout

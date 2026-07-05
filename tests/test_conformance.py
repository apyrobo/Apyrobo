"""Tests for the APYROBO Protocol conformance suite (apyrobo/conformance/)."""
from __future__ import annotations

import json

import pytest

from apyrobo.conformance import ConformanceReport, run_conformance
from apyrobo.conformance.adapter_checks import run_adapter_checks
from apyrobo.conformance.report import REPORT_FORMAT_VERSION, SPEC_VERSION
from apyrobo.conformance.wire_checks import WireClient, run_wire_checks
from apyrobo.core.adapters import MockAdapter
from apyrobo.orchestration.adapter import (
    OrchestrationMessage,
    OrchestrationServer,
)
from apyrobo.skills.agent import Agent


def by_id(report: ConformanceReport) -> dict[str, str]:
    return {c.check_id: c.status for c in report.checks}


# ---------------------------------------------------------------------------
# Report model
# ---------------------------------------------------------------------------

class TestConformanceReport:
    def test_json_shape(self):
        report = ConformanceReport(target="mock://x", kind="adapter")
        report.add("T-01", "a check", "MUST", "spec.md §1", "pass")
        report.add("T-02", "another", "SHOULD", "spec.md §2", "warn", "details")
        data = json.loads(report.to_json())
        assert data["apyrobo_conformance_report"] == REPORT_FORMAT_VERSION
        assert data["spec_version"] == SPEC_VERSION
        assert data["target"] == "mock://x"
        assert data["kind"] == "adapter"
        assert data["summary"] == {"pass": 1, "warn": 1, "fail": 0, "skip": 0}
        assert data["conformant"] is True
        assert data["checks"][0] == {
            "id": "T-01", "title": "a check", "level": "MUST",
            "spec_ref": "spec.md §1", "status": "pass", "details": "",
        }

    def test_must_failure_makes_non_conformant(self):
        report = ConformanceReport(target="x://y", kind="adapter")
        report.add("T-01", "a check", "MUST", "spec.md", "fail", "broke")
        assert report.conformant is False
        assert "NOT CONFORMANT" in report.render_text()

    def test_should_warning_stays_conformant(self):
        report = ConformanceReport(target="x://y", kind="adapter")
        report.add("T-01", "a check", "SHOULD", "spec.md", "warn")
        assert report.conformant is True
        assert "Result: CONFORMANT" in report.render_text()


# ---------------------------------------------------------------------------
# Adapter-contract checks
# ---------------------------------------------------------------------------

class FailFastAdapter(MockAdapter):
    """Mock variant that rejects motion commands while disconnected."""

    def move(self, x, y, speed=None):
        if not self.is_connected:
            raise ConnectionError("not connected")
        super().move(x, y, speed)


class BrokenStopAdapter(MockAdapter):
    """Non-conformant: stop() depends on the connection."""

    def stop(self):
        if not self.is_connected:
            raise RuntimeError("stop needs a connection")
        super().stop()


class RaisingRotateAdapter(MockAdapter):
    """Non-conformant: optional op raises instead of defaulting."""

    def rotate(self, angle_rad, speed=None):
        raise NotImplementedError("no rotation")


class BadCapabilitiesAdapter(MockAdapter):
    def get_capabilities(self):
        raise RuntimeError("capability query exploded")


class NonBoolGripperAdapter(MockAdapter):
    def gripper_open(self):
        return "yes"


class TestAdapterChecks:
    def run(self, adapter) -> ConformanceReport:
        report = ConformanceReport(target="test://adapter", kind="adapter")
        run_adapter_checks(adapter, report)
        return report

    def test_mock_adapter_is_conformant(self):
        report = self.run(MockAdapter("conf-test"))
        assert report.conformant, report.render_text()
        # MockAdapter accepts moves while disconnected — SHOULD warning.
        assert by_id(report)["FAIL-01"] == "warn"

    def test_fail_fast_adapter_has_no_warnings(self):
        report = self.run(FailFastAdapter("conf-test"))
        assert report.conformant, report.render_text()
        assert report.summary["warn"] == 0
        assert by_id(report)["FAIL-01"] == "pass"

    def test_gazebo_adapter_is_conformant(self):
        report = run_conformance("gazebo://conf-test")
        assert report.kind == "adapter"
        assert report.conformant, report.render_text()

    def test_broken_stop_fails_safety_check(self):
        report = self.run(BrokenStopAdapter("conf-test"))
        assert not report.conformant
        assert by_id(report)["SAF-01"] == "fail"

    def test_raising_rotate_fails_optional_op_check(self):
        report = self.run(RaisingRotateAdapter("conf-test"))
        assert not report.conformant
        assert by_id(report)["OPT-01"] == "fail"

    def test_broken_capabilities_fails_declaration_check(self):
        report = self.run(BadCapabilitiesAdapter("conf-test"))
        assert not report.conformant
        assert by_id(report)["CAP-01"] == "fail"

    def test_non_bool_gripper_fails(self):
        report = self.run(NonBoolGripperAdapter("conf-test"))
        assert not report.conformant
        assert by_id(report)["OPT-02"] == "fail"

    def test_schema_check_runs_when_jsonschema_present(self):
        pytest.importorskip("jsonschema")
        report = self.run(MockAdapter("conf-test"))
        assert by_id(report)["CAP-04"] == "pass"


# ---------------------------------------------------------------------------
# Wire-protocol checks (in-process transport around the reference server)
# ---------------------------------------------------------------------------

class InProcessWireClient(WireClient):
    """Feeds text through the reference server's stdio framing, in-process."""

    def __init__(self, server: OrchestrationServer) -> None:
        self.server = server
        self._responses: list[dict] = []

    def send_text(self, text: str) -> None:
        text = text.strip()
        try:
            msg = OrchestrationMessage.from_dict(json.loads(text))
        except json.JSONDecodeError:
            msg = OrchestrationMessage(task=text)  # reference behavior (§2.3)
        self._responses.append(self.server._handle(msg).to_dict())

    def receive(self, timeout: float) -> dict | None:
        return self._responses.pop(0) if self._responses else None


class DroppingWireClient(InProcessWireClient):
    """Non-conformant server stand-in: dies on malformed input."""

    def __init__(self, server: OrchestrationServer) -> None:
        super().__init__(server)
        self._dead = False

    def send_text(self, text: str) -> None:
        if self._dead:
            raise ConnectionError("connection closed")
        try:
            json.loads(text)
        except json.JSONDecodeError:
            self._dead = True
            return
        super().send_text(text)


@pytest.fixture
def reference_server() -> OrchestrationServer:
    from apyrobo.orchestration.adapter import MockOrchestrationAdapter

    return OrchestrationServer(
        MockOrchestrationAdapter(),
        Agent(provider="rule"),
        default_robot_uri="mock://wire-default",
    )


class TestWireChecks:
    def test_reference_server_is_conformant(self, reference_server):
        report = ConformanceReport(target="in-process", kind="wire-protocol")
        client = InProcessWireClient(reference_server)
        run_wire_checks(client, report, robot_uri="mock://wire-probe", timeout=5.0)
        assert report.conformant, report.render_text()
        assert report.summary["fail"] == 0
        statuses = by_id(report)
        assert statuses["WP-01"] == "pass"
        assert statuses["WP-05"] == "pass"
        assert statuses["WP-08"] == "pass"

    def test_connection_dropped_on_malformed_input_fails(self, reference_server):
        report = ConformanceReport(target="in-process", kind="wire-protocol")
        client = DroppingWireClient(reference_server)
        run_wire_checks(client, report, robot_uri="mock://wire-probe", timeout=1.0)
        assert not report.conformant
        assert by_id(report)["WP-06"] == "fail"

    def test_unresponsive_server_skips_remaining_checks(self):
        class SilentClient(WireClient):
            def send_text(self, text: str) -> None:
                pass

            def receive(self, timeout: float):
                return None

        report = ConformanceReport(target="in-process", kind="wire-protocol")
        run_wire_checks(SilentClient(), report, timeout=0.1)
        statuses = by_id(report)
        assert statuses["WP-01"] == "fail"
        assert all(
            status == "skip"
            for check_id, status in statuses.items()
            if check_id != "WP-01"
        )


# ---------------------------------------------------------------------------
# Server default-robot resolution (the bug WP-05 originally caught)
# ---------------------------------------------------------------------------

class TestServerRobotResolution:
    def test_explicit_robot_uri_wins_over_default_robot(self):
        from apyrobo.core.robot import Robot
        from apyrobo.orchestration.adapter import MockOrchestrationAdapter

        adapter = MockOrchestrationAdapter(
            tasks=[OrchestrationMessage(task="go", robot_uri="bogus://nowhere")]
        )
        server = OrchestrationServer(
            adapter,
            Agent(provider="rule"),
            default_robot=Robot.discover("mock://default-bot"),
            default_robot_uri="mock://default-bot",
        )
        server.run()
        response = adapter.sent[0]
        assert response.metadata["status"] == "error"
        assert response.robot_uri == "bogus://nowhere"

    def test_absent_robot_uri_uses_default_and_echoes_resolved(self):
        from apyrobo.orchestration.adapter import MockOrchestrationAdapter

        adapter = MockOrchestrationAdapter(tasks=["go to the dock"])
        server = OrchestrationServer(
            adapter,
            Agent(provider="rule"),
            default_robot_uri="mock://default-bot",
        )
        server.run()
        response = adapter.sent[0]
        assert response.metadata["status"] == "planned"
        assert response.robot_uri == "mock://default-bot"


# ---------------------------------------------------------------------------
# Runner dispatch
# ---------------------------------------------------------------------------

class TestRunner:
    def test_adapter_target(self):
        report = run_conformance("mock://runner-test")
        assert report.kind == "adapter"
        assert report.conformant

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="No adapter registered"):
            run_conformance("definitely-not-registered://x")

    def test_target_without_form_raises(self):
        with pytest.raises(ValueError, match="unrecognized conformance target"):
            run_conformance("not-a-target")

    def test_adapter_target_without_name_raises(self):
        with pytest.raises(ValueError, match="robot name"):
            run_conformance("mock://")

    def test_stdio_target_without_command_raises(self):
        with pytest.raises(ValueError, match="stdio target needs a command"):
            run_conformance("stdio:")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestConformanceCLI:
    def test_json_output_and_exit_code(self, capsys):
        from unittest.mock import patch

        from apyrobo.cli import main

        with patch(
            "sys.argv", ["apyrobo", "conformance", "mock://cli-test", "--json"]
        ), pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["conformant"] is True
        assert data["kind"] == "adapter"

    def test_non_conformant_target_exits_1(self, capsys):
        from unittest.mock import patch

        from apyrobo.cli import main
        from apyrobo.core.adapters import _ADAPTER_REGISTRY, register_adapter_class

        register_adapter_class("conf-broken-test", BrokenStopAdapter)
        try:
            with patch(
                "sys.argv", ["apyrobo", "conformance", "conf-broken-test://x"]
            ), pytest.raises(SystemExit) as excinfo:
                main()
        finally:
            del _ADAPTER_REGISTRY["conf-broken-test"]
        assert excinfo.value.code == 1
        assert "NOT CONFORMANT" in capsys.readouterr().out

"""Conformance runner — dispatches a target URI to the right check suite.

Target forms:

* ``scheme://name`` — instantiate the registered capability adapter and run
  the adapter-contract checks in-process.
* ``ws://host:port`` / ``wss://…`` — run the wire-protocol checks against a
  live orchestration server.
* ``stdio:<command>`` — spawn *command* and run the wire-protocol checks
  over its stdin/stdout.
"""
from __future__ import annotations

from apyrobo.conformance.report import ConformanceReport


def run_conformance(
    target: str,
    robot_uri: str = "mock://conformance-probe",
    timeout: float = 15.0,
) -> ConformanceReport:
    """Run the appropriate conformance checks for *target* and return the report."""
    if target.startswith(("ws://", "wss://")):
        from apyrobo.conformance.wire_checks import (
            WebSocketWireClient,
            run_wire_checks,
        )

        report = ConformanceReport(target=target, kind="wire-protocol")
        client = WebSocketWireClient(target, connect_timeout=timeout)
        try:
            run_wire_checks(client, report, robot_uri=robot_uri, timeout=timeout)
        finally:
            client.close()
        return report

    if target.startswith("stdio:"):
        from apyrobo.conformance.wire_checks import StdioWireClient, run_wire_checks

        command = target[len("stdio:"):].strip()
        if not command:
            raise ValueError("stdio target needs a command: stdio:<command>")
        report = ConformanceReport(target=target, kind="wire-protocol")
        client = StdioWireClient(command)
        try:
            run_wire_checks(client, report, robot_uri=robot_uri, timeout=timeout)
        finally:
            client.close()
        return report

    if "://" in target:
        from apyrobo.conformance.adapter_checks import run_adapter_checks
        from apyrobo.core.adapters import get_adapter

        scheme, _, robot_name = target.partition("://")
        if not robot_name:
            raise ValueError(
                f"adapter target needs a robot name: {scheme}://<name>"
            )
        adapter = get_adapter(scheme, robot_name)
        report = ConformanceReport(target=target, kind="adapter")
        run_adapter_checks(adapter, report)
        return report

    raise ValueError(
        f"unrecognized conformance target {target!r} — expected an adapter URI "
        "(scheme://name), a WebSocket server (ws://host:port), or a spawned "
        "stdio server (stdio:<command>)"
    )

"""APYROBO Protocol conformance suite.

Runs any capability adapter or wire-protocol server against the normative
spec in ``spec/`` and produces a machine-readable report:

    apyrobo conformance mock://robot1              # adapter contract checks
    apyrobo conformance ws://localhost:8765        # wire protocol, live server
    apyrobo conformance "stdio:apyrobo serve"      # wire protocol, spawned server

See ``docs/conformance.md`` for the check catalog and the conformant badge
program.
"""
from apyrobo.conformance.report import SPEC_VERSION, CheckResult, ConformanceReport
from apyrobo.conformance.runner import run_conformance

__all__ = [
    "CheckResult",
    "ConformanceReport",
    "SPEC_VERSION",
    "run_conformance",
]

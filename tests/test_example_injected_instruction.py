"""
examples/07_injected_instruction.py sends the calls a hijacked agent would
make to the real ``apyrobo mcp`` server over the official MCP Python SDK: the
move into the no-go zone and the same goal as a task are refused with code
collision_zone and the robot does not move, the fast move is clamped to
0.5 m/s, and the audit log has one decision line per call.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("mcp")
anyio = pytest.importorskip("anyio")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "07_injected_instruction.py"


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("injected_instruction_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def log(example, tmp_path_factory):
    audit = tmp_path_factory.mktemp("audit") / "audit.jsonl"

    async def run_with_timeout():
        with anyio.fail_after(60):
            return await example.run(str(audit))

    return anyio.run(run_with_timeout)


def test_says_no_model_is_called():
    header = EXAMPLE.read_text(encoding="utf-8").split('"""')[1]
    assert "No model is called" in header
    assert "it does not test a model" in header


def test_move_into_the_zone_is_refused_and_robot_stays_put(log):
    i = log.index('move_to {"x": 3.0, "y": 3.0, "speed": 5.0}')
    assert log[i + 1].startswith("  refused [collision_zone]: REFUSED by safety policy")
    assert "did not move" in log[i + 1]
    assert log[i + 2] == "  position (0.0, 0.0) -> (0.0, 0.0)"


def test_same_goal_as_a_task_is_refused_and_robot_stays_put(log):
    i = log.index('execute_task {"task": "go to 3, 3"}')
    assert log[i + 1].startswith("  refused [collision_zone]: REFUSED by safety policy")
    assert log[i + 2] == "  position (0.0, 0.0) -> (0.0, 0.0)"


def test_fast_move_is_clamped_to_the_limit(log):
    i = log.index('move_to {"x": 1.0, "y": 1.0, "speed": 5.0}')
    assert log[i + 1] == "  ran, speed clamped: asked 5.0 m/s, ran at 0.5 m/s"
    assert log[i + 2] == "  position (0.0, 0.0) -> (1.0, 1.0)"


def test_audit_log_has_one_decision_per_call(log):
    audit = [line for line in log if line.startswith("audit: ")]
    assert audit == [
        "audit: move_to refused (enforced_speed None)",
        "audit: execute_task refused (enforced_speed None)",
        "audit: move_to clamped (enforced_speed 0.5)",
    ]

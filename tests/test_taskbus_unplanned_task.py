"""A task the agent refuses must not come back from the TaskBus as a success."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from apyrobo import Agent, MockAdapter, Robot
from apyrobo.coordination.bus import MultiAgentCoordinator, TaskBus

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def bus():
    bus = TaskBus(timeout=10.0)
    coords = []
    for name, caps in [("picker", ["PICK", "NAVIGATE"]), ("hauler", ["HAUL", "NAVIGATE"])]:
        robot = Robot(f"mock://{name}", MockAdapter(name))
        coord = MultiAgentCoordinator(
            Agent(provider="rule"), robot, bus, agent_id=name, capabilities=caps
        )
        coord.start()
        coords.append(coord)
    yield bus
    for coord in coords:
        coord.stop()


def test_refused_drive_is_not_a_success(bus):
    result = bus.dispatch("navigate_to", required_capability="PICK")
    assert result.success is False
    assert result.skills_planned == []
    assert result.error.startswith("No target")


def test_planned_drive_is_a_success(bus):
    result = bus.dispatch("go to (3, 3)", required_capability="PICK")
    assert result.success is True
    assert result.error == ""
    assert len(result.skills_planned) == 1


def test_broadcast_reports_each_refusal(bus):
    results = bus.broadcast("navigate_to")
    assert len(results) == 2
    assert all(r.success is False and r.error.startswith("No target") for r in results)


def test_empty_plan_without_a_reason_still_fails():
    class Silent:
        last_refusal = None

        def plan(self, task, robot):
            from apyrobo.skills.executor import SkillGraph

            return SkillGraph()

    bus = TaskBus(timeout=5.0)
    robot = Robot("mock://s", MockAdapter("s"))
    coord = MultiAgentCoordinator(Silent(), robot, bus, agent_id="s", capabilities=["NAVIGATE"])
    coord.start()
    try:
        result = bus.dispatch("anything")
    finally:
        coord.stop()
    assert result.success is False
    assert result.error == "no skills were planned for this task"


def _run(path: Path) -> str:
    proc = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, timeout=120, cwd=REPO
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout + proc.stderr


def test_warehouse_demo_plans_every_step_on_the_right_robot():
    out = _run(REPO / "demos" / "warehouse_robots" / "demo.py")
    assert "not planned" not in out
    assert "task not planned" not in out
    assert "5/5 orders planned" in out
    deliveries = re.findall(r"\[(\w+)\] deliver to dock_\d", out)
    assert len(deliveries) == 5 and set(deliveries) == {"hauler_bot"}
    shelves = re.findall(r"\[(\w+)\] to '\w+' shelf", out)
    assert len(shelves) == 10 and set(shelves) == {"picker_bot"}


def test_concept_doc_example_shows_the_refusal(tmp_path):
    text = (REPO / "docs" / "concepts" / "multi_agent_coordination.md").read_text()
    snippet = re.findall(r"```python\n(.*?)```", text, re.S)[0]
    script = tmp_path / "example.py"
    script.write_text(snippet)
    out = _run(script)
    assert "Handled by: picker_bot" in out
    assert "False No target" in out

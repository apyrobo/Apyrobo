"""docs/skill_authoring.md runs as written, and a failed graph says why.

Steps 1, 3 and 4 of the guide define a skill, a custom executor and a
three-skill graph. Run together on mock:// the graph must complete. When a
skill does fail, TaskResult.error names the skill and the reason, keeping
the "Skill '<id>' failed" prefix the agent's replanner parses.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from apyrobo import Robot, SkillExecutor, SkillGraph
from apyrobo.core.schemas import CapabilityType, TaskStatus
from apyrobo.skills.skill import Condition, Skill

GUIDE = Path(__file__).resolve().parent.parent / "docs" / "skill_authoring.md"


def _python_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", GUIDE.read_text(), re.S)


def test_guide_graph_completes_as_written(tmp_path: Path) -> None:
    blocks = _python_blocks()
    # 1. Define a Skill, 2. Register, 3. Custom handler, 4. Chain in a graph.
    script = "\n".join(blocks[:4]) + (
        "\nassert result.status.value == 'completed', result"
        "\nassert result.steps_completed == 3, result"
        "\nprint('GUIDE_OK')\n"
    )
    path = tmp_path / "guide_steps.py"
    path.write_text(script)
    proc = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "GUIDE_OK" in proc.stdout
    assert "completed" in proc.stdout


def test_guide_quotes_the_error_the_executor_gives() -> None:
    robot = Robot.discover("mock://guide_err")
    graph = SkillGraph()
    graph.add_skill(
        Skill(
            skill_id="go_home",
            name="Go Home",
            required_capability=CapabilityType.NAVIGATE,
        )
    )
    result = SkillExecutor(robot).execute_graph(graph)
    assert result.error == "Skill 'go_home' failed: No handler registered for 'go_home'"
    assert result.error in GUIDE.read_text()


def _no_handler_skill(skill_id: str) -> Skill:
    return Skill(
        skill_id=skill_id,
        name=skill_id,
        required_capability=CapabilityType.NAVIGATE,
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_failed_graph_error_names_missing_handler(parallel: bool) -> None:
    robot = Robot.discover("mock://graph_err")
    graph = SkillGraph()
    graph.add_skill(_no_handler_skill("no_such_skill_xyz"))
    result = SkillExecutor(robot).execute_graph(graph, parallel=parallel)
    assert result.status == TaskStatus.FAILED
    assert result.error.startswith("Skill 'no_such_skill_xyz' failed: ")
    assert "No handler registered for 'no_such_skill_xyz'" in result.error


def test_failed_graph_error_in_a_parallel_layer() -> None:
    robot = Robot.discover("mock://graph_layer")
    graph = SkillGraph()
    graph.add_skill(_no_handler_skill("layer_missing_a"))
    graph.add_skill(_no_handler_skill("layer_missing_b"))
    result = SkillExecutor(robot).execute_graph(graph, parallel=True)
    assert result.status == TaskStatus.FAILED
    assert re.match(r"Skill 'layer_missing_[ab]' failed: No handler registered", result.error)


def test_failed_graph_error_names_failed_precondition() -> None:
    robot = Robot.discover("mock://graph_pre")
    skill = Skill(
        skill_id="navigate_to",
        name="Navigate",
        required_capability=CapabilityType.NAVIGATE,
        preconditions=[
            Condition(
                name="door_open",
                check_type="state",
                parameters={"key": "door_open", "value": True},
            )
        ],
        parameters={"x": 1.0, "y": 1.0},
    )
    graph = SkillGraph()
    graph.add_skill(skill)
    result = SkillExecutor(robot).execute_graph(graph)
    assert result.status == TaskStatus.FAILED
    assert result.error.startswith("Skill 'navigate_to' failed: precondition failed: ")


def test_reason_keeps_the_prefix_the_replanner_parses() -> None:
    robot = Robot.discover("mock://graph_prefix")
    graph = SkillGraph()
    graph.add_skill(_no_handler_skill("prefix_missing_0"))
    result = SkillExecutor(robot).execute_graph(graph)
    # Same pattern as apyrobo/skills/agent.py uses to find the failed step.
    m = re.search(r"Skill '([^']+)' failed", result.error or "")
    assert m and m.group(1) == "prefix_missing_0"


def test_success_after_failure_leaves_no_stale_reason() -> None:
    robot = Robot.discover("mock://graph_ok")
    executor = SkillExecutor(robot)
    bad = SkillGraph()
    bad.add_skill(_no_handler_skill("stale_missing"))
    assert executor.execute_graph(bad).status == TaskStatus.FAILED
    good = SkillGraph()
    good.add_skill(
        Skill(
            skill_id="navigate_to",
            name="Navigate",
            required_capability=CapabilityType.NAVIGATE,
            parameters={"x": 1.0, "y": 0.0},
        )
    )
    result = executor.execute_graph(good)
    assert result.status == TaskStatus.COMPLETED
    assert not result.error

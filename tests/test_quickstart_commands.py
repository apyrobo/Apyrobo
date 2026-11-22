"""docs/quickstart_5min.md runs as written on mock://.

Every Python block is run on its own, in a fresh namespace, the way a new
user pastes it, and the results its comments promise are checked. The
install commands are checked to work outside a clone of the repo.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "quickstart_5min.md"


def _blocks(lang: str) -> list[str]:
    text = DOC.read_text(encoding="utf-8")
    return re.findall(rf"```{lang}\n(.*?)```", text, re.S)


def _run(block: str, capsys) -> tuple[dict, str]:
    namespace: dict = {}
    exec(compile(block, str(DOC), "exec"), namespace)
    return namespace, capsys.readouterr().out


def _block_with(snippet: str) -> str:
    found = [b for b in _blocks("python") if snippet in b]
    assert len(found) == 1, f"expected one python block with {snippet!r}"
    return found[0]


def test_every_python_block_runs_on_its_own(capsys):
    blocks = _blocks("python")
    assert len(blocks) >= 6
    for block in blocks:
        _run(block, capsys)


def test_step_1_prints_the_capabilities_and_position_it_shows(capsys):
    block = _block_with('Robot.discover("mock://turtlebot4")\n\n# What can it do?')
    assert "# → ['navigate_to', 'rotate', 'pick_object', 'place_object']" in block
    assert "# → Position: (2.0, 3.0)" in block
    _, out = _run(block, capsys)
    assert "Capabilities: ['navigate_to', 'rotate', 'pick_object', 'place_object']" in out
    assert "Position: (2.0, 3.0)" in out


def test_step_2_completes_both_steps(capsys):
    block = _block_with("executor.execute_graph(graph)")
    assert "# → Result: completed (2/2 steps)" in block
    _, out = _run(block, capsys)
    assert "Result: completed (2/2 steps)" in out


def test_step_3_rule_agent_completes_the_task(capsys):
    block = _block_with('agent.execute("pick up the object at (5, 3)", robot)')
    namespace, _ = _run(block, capsys)
    result = namespace["result"]
    assert result.status.value == "completed"
    assert result.steps_completed == result.steps_total > 0


def test_step_3_llm_snippet_imports_what_it_uses(capsys):
    block = _block_with('provider="llm"')
    namespace, _ = _run(block, capsys)
    assert namespace["agent"] is not None
    text = DOC.read_text(encoding="utf-8")
    before = text[: text.index(block)]
    assert "apyrobo[llm]" in before.rsplit("## Step 3", 1)[1]


def test_step_4_clamps_to_1_5_and_blocks_the_zone(capsys):
    block = _block_with("SafetyEnforcer(robot")
    assert "# → clamped to 1.5 m/s" in block
    namespace, out = _run(block, capsys)
    clamp = namespace["enforcer"].interventions[0]
    assert clamp["type"] == "speed_clamped"
    assert clamp["enforced"] == pytest.approx(1.5)
    assert "Blocked:" in out


def test_step_5_swarm_has_two_robots_and_completes(capsys):
    block = _block_with("SwarmCoordinator(bus)")
    namespace, out = _run(block, capsys)
    assert "Swarm: 2 robots" in out
    assert namespace["result"].status.value == "completed"


def test_install_commands_work_outside_a_clone():
    # A new user reads this page on the web, not inside the repo: an
    # editable install only works after the clone and cd.
    blocks = _blocks("bash")
    install = [b for b in blocks if "pip install" in b]
    assert install
    for block in install:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            if line.startswith("pip install -e"):
                assert any(prev.startswith("cd ") for prev in lines[:i]), block
    assert any("pip install apyrobo" in b for b in install)

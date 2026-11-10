"""Keep the code shown in the docs in step with the package a reader installs."""

from __future__ import annotations

import argparse
import importlib
import re
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from apyrobo import cli
from apyrobo.core.schemas import CapabilityType
from apyrobo.profiles import ProfileRegistry

ROOT = Path(__file__).resolve().parent.parent
DOCS = sorted(
    [ROOT / "README.md", *(ROOT / "docs").rglob("*.md"), *(ROOT / "demos").rglob("*.md")]
)
PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.S)
ANY_BLOCK = re.compile(r"```[\w-]*\n(.*?)```", re.S)
FROM_IMPORT = re.compile(r"^\s*from (apyrobo(?:\.[\w.]+)?) import ([^#\n]+)", re.M)


def _blocks(pattern: re.Pattern[str]) -> list[tuple[str, str]]:
    return [
        (str(path.relative_to(ROOT)), block)
        for path in DOCS
        for block in pattern.findall(path.read_text(encoding="utf-8"))
    ]


def _documented_imports() -> list[tuple[str, str, str]]:
    found = set()
    for doc, block in _blocks(PYTHON_BLOCK):
        # Skips multi-line parenthesised imports' continuation lines, which
        # the opening line's module check still covers.
        for module, names in FROM_IMPORT.findall(block):
            for name in names.strip(" ()").split(","):
                name = name.split(" as ")[0].strip()
                if name and name != "*":
                    found.add((doc, module, name))
    return sorted(found)


@pytest.mark.parametrize("doc,module,name", _documented_imports())
def test_a_documented_import_exists(doc, module, name):
    loaded = importlib.import_module(module)
    if not hasattr(loaded, name):
        importlib.import_module(f"{module}.{name}")


def _subcommands() -> set[str]:
    captured: dict[str, argparse.ArgumentParser] = {}

    class StopParsingError(Exception):
        pass

    def stop(self, *args, **kwargs):
        captured["parser"] = self
        raise StopParsingError

    with (
        mock.patch.object(argparse.ArgumentParser, "parse_args", stop),
        pytest.raises(StopParsingError),
    ):
        cli.main()
    actions = captured["parser"]._subparsers._group_actions
    return set(actions[0].choices)


def test_every_documented_command_exists():
    known = _subcommands()
    assert {"execute", "mcp", "profiles"} <= known
    used = set()
    for doc, block in _blocks(ANY_BLOCK):
        for sub in re.findall(r"^\s*(?:\$ )?apyrobo (\w[\w-]*)", block, re.M):
            used.add((doc, sub))
    missing = sorted(item for item in used if item[1] not in known)
    assert not missing, f"docs use commands apyrobo does not have: {missing}"


def test_policy_check_exits_1_on_a_violation_as_documented(tmp_path):
    db = str(tmp_path / "policies.db")
    base = [sys.executable, "-m", "apyrobo", "policy"]
    add = subprocess.run(
        [*base, "add", "never exceed 0.5 m/s near humans", "--db", db],
        capture_output=True,
        text=True,
    )
    assert add.returncode == 0, add.stderr
    check = subprocess.run(
        [*base, "check", "navigate at 2 m/s near humans", "--db", db],
        capture_output=True,
        text=True,
    )
    assert check.returncode == 1, check.stdout + check.stderr


def test_the_readme_profile_table_matches_the_registry():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    rows = dict(re.findall(r"^\| `([\w-]+)` \| [^|]+ \| `([^`]+)` \|$", text, re.M))
    expected = {p.name: p.litellm_model for p in ProfileRegistry().all()}
    assert {k: rows.get(k) for k in expected} == expected


def test_a_profile_passes_a_provider_qualified_model_to_the_planner():
    seen = {}

    class FakeAgent:
        def __init__(self, provider, model=None):
            seen.update(provider=provider, model=model)

        def plan(self, task, robot):
            raise SystemExit(0)

    args = argparse.Namespace(
        robot="mock://turtlebot4", provider="llm", profile="jetson-orin", task="go to (1, 2)"
    )
    with mock.patch.object(cli, "Agent", FakeAgent), pytest.raises(SystemExit):
        cli.cmd_plan(args)
    assert seen == {"provider": "llm", "model": "ollama/llama3.2:3b"}


def test_the_custom_adapter_in_the_docs_runs():
    text = (ROOT / "docs" / "concepts" / "adapter_pattern.md").read_text(encoding="utf-8")
    block = next(b for b in PYTHON_BLOCK.findall(text) if "class MyRobotAdapter" in b)
    namespace: dict = {}
    exec(block, namespace)
    adapter = namespace["MyRobotAdapter"]("demo")
    caps = adapter.get_capabilities()
    assert caps.capabilities[0].capability_type is CapabilityType.NAVIGATE
    adapter.move(1.0, 2.0, 0.5)
    assert adapter.target == (1.0, 2.0, 0.5)
    adapter.stop()
    assert adapter.target is None


def _video_tutorial_blocks() -> list[str]:
    text = (ROOT / "docs" / "video_tutorial.md").read_text(encoding="utf-8")
    return [b for b in PYTHON_BLOCK.findall(text) if "execute" in b and "result" in b]


def test_the_video_tutorial_code_runs_and_prints_what_it_says(capsys):
    blocks = _video_tutorial_blocks()
    assert len(blocks) == 3  # skill graph, agent, swarm
    for block in blocks:
        # The LLM half of the agent block needs a model and an API key.
        exec(compile(block.split("# LLM-powered agent")[0], "video_tutorial.md", "exec"), {})
    out = capsys.readouterr().out.splitlines()
    assert out.count("completed") == 3
    assert "4/4 steps" in out
    layers = [line for line in out if line.startswith("['")]
    expected = ["navigate_to_0", "pick_object_1", "navigate_to_2", "place_object_3"]
    assert layers == [f"['{skill_id}']" for skill_id in expected]


def test_the_video_tutorial_graph_visits_both_waypoints():
    from apyrobo import Robot

    block = _video_tutorial_blocks()[0]
    visited: list[tuple[float, float]] = []
    real_discover = Robot.discover

    def discover(uri, **kwargs):
        robot = real_discover(uri, **kwargs)
        real_move = robot.move

        def move(x, y, speed=None):
            visited.append((x, y))
            return real_move(x, y, speed)

        robot.move = move
        return robot

    with mock.patch.object(Robot, "discover", side_effect=discover):
        exec(compile(block, "video_tutorial.md", "exec"), {})
    assert visited == [(5.0, 3.0), (0.0, 0.0)]

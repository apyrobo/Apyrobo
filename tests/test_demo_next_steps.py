"""The "Next steps" the demos print must name things that exist.

Each demo ends with hints a new user copies straight into a terminal or a
Python shell. These tests run the demos and check that every hint works on
the current code: flags parse, classes import, URIs resolve, docs exist.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DEMOS = REPO / "demos"


def _run(script: Path, *args: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True, text=True, timeout=120, cwd=REPO,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _next_steps(output: str) -> str:
    assert "Next steps:" in output
    return output.split("Next steps:", 1)[1]


def _load_drone_demo():
    spec = importlib.util.spec_from_file_location(
        "drone_survey_demo", DEMOS / "drone_survey" / "demo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# drone_survey
# ---------------------------------------------------------------------------

def test_drone_fleet_size_flag_flies_that_many_drones():
    out = _run(DEMOS / "drone_survey" / "demo.py", "--fleet-size", "25")
    assert "APYROBO — 25-Drone Coordinated Survey" in out
    assert "Sectors covered: 25/25" in out
    assert len(re.findall(r"^\s+\[drone_\d+\] sector", out, re.M)) == 25


def test_drone_default_fleet_is_ten():
    out = _run(DEMOS / "drone_survey" / "demo.py")
    assert "Sectors covered: 10/10" in out


@pytest.mark.parametrize("bad", ["0", "-3", "ten", "2.5"])
def test_drone_fleet_size_refuses_bad_values(bad, capsys):
    demo = _load_drone_demo()
    with pytest.raises(SystemExit) as exc:
        demo.main(["--fleet-size", bad])
    assert exc.value.code == 2
    assert "--fleet-size" in capsys.readouterr().err


def test_drone_hint_names_the_flag_that_exists():
    hints = _next_steps(_run(DEMOS / "drone_survey" / "demo.py"))
    assert "--fleet-size 100" in hints


def test_drone_demo_names_no_adapter_that_does_not_exist():
    for path in (DEMOS / "drone_survey" / "demo.py", DEMOS / "drone_survey" / "README.md"):
        text = path.read_text()
        for missing in ("DDSAdapter", "dds://", "MAVLink"):
            assert missing not in text, f"{path.name} still names {missing}"
    assert (REPO / "docs" / "adapter_authoring.md").exists()


# ---------------------------------------------------------------------------
# warehouse_robots
# ---------------------------------------------------------------------------

def test_warehouse_real_robot_hint_runs():
    hints = _next_steps(_run(DEMOS / "warehouse_robots" / "demo.py"))
    imports = re.findall(r"`(import [\w.]+)`", hints)
    calls = re.findall(r"`(Robot\.discover\([^`]+\))`", hints)
    assert imports == ["import apyrobo.core.unitree_adapter"]
    assert len(calls) == 1
    namespace: dict = {}
    exec(imports[0], namespace)
    exec("from apyrobo import Robot", namespace)
    robot = eval(calls[0], namespace)
    assert "adapter=UnitreeAdapter" in repr(robot)


# ---------------------------------------------------------------------------
# humanoid_nlp
# ---------------------------------------------------------------------------

def test_humanoid_llm_hint_points_at_a_doc_that_exists():
    hints = _next_steps(_run(DEMOS / "humanoid_nlp" / "demo.py"))
    assert "LLMAgent" not in hints
    docs = re.findall(r"(docs/[\w/]+\.md)", hints)
    assert docs, hints
    for doc in docs:
        assert (REPO / doc).exists(), doc

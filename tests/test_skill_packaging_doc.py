"""docs/skill_packaging.md: the test-skill commands it shows must work.

The guide used to say ``apyrobo test-skill go_to_waypoint --repeat 5`` runs a
packaged skill. test-skill does not load installed packages, so that command
fails with "not found". These tests build the guide's own files and run the
commands the guide now shows, plus the ones it says do not work.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "skill_packaging.md"


def _guide() -> str:
    return GUIDE.read_text(encoding="utf-8")


def _skill_source() -> str:
    """The guide's navigation.py block (section 3)."""
    for lang, body in re.findall(r"```(\w+)\n(.*?)```", _guide(), re.S):
        if lang == "python" and body.startswith("# apyrobo_skills_myrobot/navigation.py"):
            return body
    raise AssertionError("navigation.py block not found in docs/skill_packaging.md")


def _test_skill(cwd: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "apyrobo", "test-skill", *argv],
        cwd=cwd, capture_output=True, text=True, timeout=120,
    )


def _shown_test_skill_commands() -> list[list[str]]:
    cmds = []
    for line in _guide().splitlines():
        line = line.strip()
        if line.startswith("apyrobo test-skill "):
            cmds.append(re.findall(r"'[^']*'|\S+", line)[2:])
    return [[a[1:-1] if a.startswith("'") else a for a in c] for c in cmds]


def test_guide_no_longer_claims_test_skill_finds_a_packaged_skill_by_id():
    for argv in _shown_test_skill_commands():
        assert argv[0] != "go_to_waypoint", (
            "apyrobo test-skill does not load installed packages, so a bare skill ID "
            "from a package is not found"
        )


@pytest.mark.parametrize("argv", _shown_test_skill_commands())
def test_every_test_skill_command_shown_runs(tmp_path, argv):
    (tmp_path / "go_to_waypoint.py").write_text(_skill_source(), encoding="utf-8")
    proc = _test_skill(tmp_path, *argv)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "go_to_waypoint" in proc.stdout
    assert "not found" not in proc.stderr


def test_params_from_the_guide_reach_the_skill(tmp_path):
    (tmp_path / "go_to_waypoint.py").write_text(_skill_source(), encoding="utf-8")
    proc = _test_skill(tmp_path, "go_to_waypoint.py", "--params", '{"waypoint_id": "wp_42"}')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "'wp_42'" in proc.stdout


def test_what_the_guide_says_does_not_work_really_does_not(tmp_path):
    pkg = tmp_path / "apyrobo_skills_myrobot"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "navigation.py").write_text(_skill_source(), encoding="utf-8")

    by_id = _test_skill(tmp_path, "go_to_waypoint")
    assert by_id.returncode != 0
    assert "'go_to_waypoint' not found" in by_id.stderr

    by_module = _test_skill(tmp_path, "apyrobo_skills_myrobot/navigation.py")
    assert by_module.returncode != 0
    assert "'navigation' not found" in by_module.stderr

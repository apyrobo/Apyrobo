"""`apyrobo skills` with no flags lists the built-in skills.

Before, plain `apyrobo skills` printed nothing and exited 0, so a new user
saw no skills; only `apyrobo skills --list` listed them.
"""

import json
import subprocess
import sys

from apyrobo.skills.skill import BUILTIN_SKILLS


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "skills", *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_plain_skills_lists_every_builtin():
    r = _run()
    assert r.returncode == 0, r.stderr
    assert "Built-in Skills:" in r.stdout
    for skill_id in BUILTIN_SKILLS:
        assert skill_id in r.stdout


def test_plain_skills_matches_list_flag():
    assert _run().stdout == _run("--list").stdout


def test_export_alone_prints_only_json():
    skill_id = next(iter(BUILTIN_SKILLS))
    r = _run("--export", skill_id)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["skill_id"] == skill_id


def test_unknown_export_names_the_known_skills_on_stderr():
    r = _run("--export", "no.such.skill")
    assert r.returncode == 1
    assert r.stdout == ""
    assert "Unknown skill: no.such.skill" in r.stderr
    for skill_id in BUILTIN_SKILLS:
        assert skill_id in r.stderr

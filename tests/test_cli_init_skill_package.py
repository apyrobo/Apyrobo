"""`apyrobo init <name>` must generate a skill package that works as printed.

The scaffold used to crash on import (an unknown ``@skill`` argument) and its
``register`` called a method that does not exist, so both generated tests failed.
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from apyrobo.cli import cmd_init
from apyrobo.core.robot import Robot
from apyrobo.skills.decorators import _DECORATED_SKILLS
from apyrobo.skills.handlers import _DEFAULT_REGISTRY


def _generate(tmp_path: Path, name: str = "hello-arm") -> Path:
    out = tmp_path / name
    cmd_init(argparse.Namespace(
        name=name, adapter=False, description="", author="",
        directory=str(out), force=False,
    ))
    return out


@pytest.fixture
def generated(tmp_path):
    """Generate a skill package and put its src/ on sys.path; clean up after."""
    before_decorated = set(_DECORATED_SKILLS)
    before_handlers = set(_DEFAULT_REGISTRY.skill_ids())
    out = _generate(tmp_path)
    sys.path.insert(0, str(out / "src"))
    yield out
    sys.path[:] = [p for p in sys.path if str(tmp_path) not in p]
    for mod in [m for m in sys.modules if m.startswith("apyrobo_skills_hello_arm")]:
        sys.modules.pop(mod, None)
    for sid in set(_DECORATED_SKILLS) - before_decorated:
        _DECORATED_SKILLS.pop(sid, None)
    for sid in set(_DEFAULT_REGISTRY.skill_ids()) - before_handlers:
        _DEFAULT_REGISTRY._handlers.pop(sid, None)


def test_generated_package_imports(generated):
    module = importlib.import_module("apyrobo_skills_hello_arm")
    assert callable(module.register)


def test_register_takes_no_arguments_and_wires_the_skill(generated):
    module = importlib.import_module("apyrobo_skills_hello_arm")
    module.register()
    assert "hello_arm_hello" in _DECORATED_SKILLS
    robot = Robot.discover("mock://test")
    assert _DEFAULT_REGISTRY.dispatch("hello_arm_hello", robot, {"message": "hi"}) is True


def test_skill_called_directly_returns_true(generated):
    skills = importlib.import_module("apyrobo_skills_hello_arm.skills")
    assert skills.hello_arm_hello(Robot.discover("mock://test")) is True


def test_generated_tests_pass(tmp_path):
    out = _generate(tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(out / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=out, env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "2 passed" in proc.stdout


def test_next_steps_only_name_commands_that_work(tmp_path, capsys):
    _generate(tmp_path)
    printed = capsys.readouterr().out
    assert "test-skill" not in printed
    assert "pytest" in printed
    readme = (tmp_path / "hello-arm" / "README.md").read_text()
    assert "apyrobo execute" not in readme
    assert ".[dev]" not in readme

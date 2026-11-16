"""`apyrobo init <name>` refuses names that cannot become a Python package.

A space, a dot or a leading digit used to print a package whose pyproject.toml
did not parse or whose skills.py did not import, with "Created" and exit 0.
A quote in --description or --author broke pyproject.toml the same way.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pytest

from apyrobo.cli import cmd_init

tomllib = pytest.importorskip("tomllib" if sys.version_info >= (3, 11) else "tomli")


def _init(tmp_path: Path, name: str, description: str = "", author: str = "") -> Path:
    out = tmp_path / "out"
    cmd_init(argparse.Namespace(
        name=name, adapter=False, description=description, author=author,
        directory=str(out), force=False,
    ))
    return out


@pytest.mark.parametrize("name", ["My Robot", "123bot", "a.b", "-x", "ünï", "", "arm/v2"])
def test_bad_name_is_refused_before_writing(tmp_path, capsys, name):
    with pytest.raises(SystemExit) as exc:
        _init(tmp_path, name)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "not a valid skill package name" in err
    assert not (tmp_path / "out").exists()


def test_bad_name_suggests_a_working_one(tmp_path, capsys):
    with pytest.raises(SystemExit):
        _init(tmp_path, "My Robot")
    assert "Try: apyrobo init my-robot" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["hello-arm", "arm_2", "Rover", "class"])
def test_good_names_still_generate_valid_files(tmp_path, name):
    out = _init(tmp_path, name)
    tomllib.loads((out / "pyproject.toml").read_text())
    for py in (out / "src").rglob("*.py"):
        ast.parse(py.read_text())


def test_quotes_in_description_and_author_keep_pyproject_valid(tmp_path):
    description = 'Says "hi" \\ to C:\\robots'
    author = "Ann O'Neil \"AJ\""
    out = _init(tmp_path, "quote-arm", description=description, author=author)
    data = tomllib.loads((out / "pyproject.toml").read_text())
    assert data["project"]["description"] == description
    assert data["project"]["authors"] == [{"name": author}]

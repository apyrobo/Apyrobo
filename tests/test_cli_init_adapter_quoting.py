"""`apyrobo init --adapter` writes a pyproject.toml and __init__.py that parse.

The description went into pyproject.toml and the package docstring raw, so a
quote or backslash broke both; the scheme was a bare TOML key, so a '.' (a
dotted key: the entry point landed under a nested table and never resolved)
or a '+' (a parse error) broke the entry point, though both are valid in a
URI scheme and the scaffold accepts them.
"""
from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from apyrobo.cli import cmd_init

tomllib = pytest.importorskip("tomllib" if sys.version_info >= (3, 11) else "tomli")


def _init(tmp_path: Path, scheme: str, description: str = "", author: str = "") -> Path:
    out = tmp_path / "out"
    cmd_init(argparse.Namespace(
        name=scheme, adapter=True, description=description, author=author,
        directory=str(out), force=False,
    ))
    return out


def _module_dir(out: Path) -> Path:
    (pkg,) = [p for p in (out / "src").iterdir() if p.is_dir()]
    return pkg


@pytest.mark.parametrize("scheme,module,cls", [
    ("zenith", "apyrobo_adapter_zenith", "ZenithAdapter"),
    ("my.bot", "apyrobo_adapter_my_bot", "MyBotAdapter"),
    ("my+bot", "apyrobo_adapter_my_bot", "MyBotAdapter"),
    ("a-b.c", "apyrobo_adapter_a_b_c", "ABCAdapter"),
])
def test_entry_point_is_named_after_the_whole_scheme(tmp_path, scheme, module, cls):
    out = _init(tmp_path, scheme)
    data = tomllib.loads((out / "pyproject.toml").read_text())
    eps = data["project"]["entry-points"]["apyrobo.adapters"]
    assert eps == {scheme: f"{module}:{cls}"}


def test_quotes_and_backslashes_in_description_and_author(tmp_path):
    description = 'Says "hi" \\ to C:\\robots """ and more'
    author = "Ann O'Neil \"AJ\" \\x"
    out = _init(tmp_path, "quotebot", description=description, author=author)
    data = tomllib.loads((out / "pyproject.toml").read_text())
    assert data["project"]["description"] == description
    assert data["project"]["authors"] == [{"name": author}]
    init_py = (_module_dir(out) / "__init__.py").read_text()
    tree = ast.parse(init_py)
    assert description in ast.get_docstring(tree)


def test_newline_in_description_keeps_files_valid(tmp_path):
    out = _init(tmp_path, "linebot", description="first\nsecond")
    data = tomllib.loads((out / "pyproject.toml").read_text())
    assert data["project"]["description"] == "first\nsecond"
    for py in (out / "src").rglob("*.py"):
        ast.parse(py.read_text())


def test_dotted_scheme_package_tests_pass(tmp_path):
    out = _init(tmp_path, "my.bot")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=out, capture_output=True, text=True, timeout=300,
        env={**os.environ, "PYTHONPATH": str(out / "src")},
    )
    assert result.returncode == 0, result.stdout + result.stderr

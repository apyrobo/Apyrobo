"""server.json (official MCP Registry metadata) stays in step with the package.

The registry reads server.json to list ``apyrobo mcp`` and verifies PyPI
ownership by finding ``mcp-name: <server name>`` in the package description
(README.md). These tests catch the ways that metadata silently goes stale:
a version bump that skips server.json, a renamed server, or a CLI flag that
was renamed or removed.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SERVER_JSON = ROOT / "server.json"


@pytest.fixture(scope="module")
def server() -> dict:
    return json.loads(SERVER_JSON.read_text(encoding="utf-8"))


def _pyproject_field(field: str) -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = text.split("[project]", 1)[1].split("\n[", 1)[0]
    match = re.search(rf'^{field}\s*=\s*"([^"]+)"', project, re.MULTILINE)
    assert match, f"no {field} in [project]"
    return match.group(1)


@pytest.fixture(scope="module")
def mcp_help() -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "apyrobo", "mcp", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_required_fields_and_limits(server: dict) -> None:
    assert server["$schema"].startswith("https://static.modelcontextprotocol.io/schemas/")
    assert re.fullmatch(r"[a-zA-Z0-9.-]+/[a-zA-Z0-9._-]+", server["name"])
    assert 1 <= len(server["description"]) <= 100
    assert server["repository"]["url"] == "https://github.com/apyrobo/Apyrobo"


def test_versions_match_pyproject(server: dict) -> None:
    version = _pyproject_field("version")
    assert server["version"] == version
    assert [p["version"] for p in server["packages"]] == [version]


def test_package_is_the_pypi_stdio_server(server: dict) -> None:
    (package,) = server["packages"]
    assert package["registryType"] == "pypi"
    assert package["registryBaseUrl"] == "https://pypi.org"
    assert package["identifier"] == _pyproject_field("name")
    assert package["transport"] == {"type": "stdio"}
    assert package["packageArguments"][0] == {"type": "positional", "value": "mcp"}


def test_readme_carries_the_mcp_name_marker(server: dict) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    # The registry needs a boundary (whitespace, tag or comment close) after the name.
    pattern = rf"mcp-name: {re.escape(server['name'])}(?=\s|<|-->)"
    assert re.search(pattern, readme)


def test_named_arguments_are_real_flags(server: dict, mcp_help: str) -> None:
    named = [a for a in server["packages"][0]["packageArguments"] if a["type"] == "named"]
    assert named
    for arg in named:
        assert re.search(rf"(?<![\w-]){re.escape(arg['name'])}(?![\w-])", mcp_help), arg["name"]
        if "choices" in arg:
            assert "{" + ",".join(arg["choices"]) + "}" in mcp_help, arg["name"]

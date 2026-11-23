"""CONTRIBUTING.md's development setup names things that exist.

The Docker option used to run ``docker compose -f docker/docker-compose.yml
exec apyrobo bash``, but that file has no ``apyrobo`` service, so the command
failed with 'service "apyrobo" is not running'. It also told contributors to
run ``ruff format .``, which rewrites hundreds of files they never touched.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRIBUTING = ROOT / "CONTRIBUTING.md"


def _bash_lines() -> list[str]:
    text = CONTRIBUTING.read_text(encoding="utf-8")
    lines: list[str] = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        for line in block.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def _compose_commands() -> list[list[str]]:
    return [shlex.split(line) for line in _bash_lines() if line.startswith("docker compose")]


def test_contributing_has_docker_compose_commands() -> None:
    assert _compose_commands(), "expected a Docker option in CONTRIBUTING.md"


def test_compose_files_exist() -> None:
    for argv in _compose_commands():
        compose_file = argv[argv.index("-f") + 1]
        assert (ROOT / compose_file).is_file(), compose_file


def test_run_and_exec_name_a_service_in_that_file() -> None:
    checked = 0
    for argv in _compose_commands():
        compose_file = argv[argv.index("-f") + 1]
        services = yaml.safe_load((ROOT / compose_file).read_text())["services"]
        for verb in ("run", "exec"):
            if verb not in argv:
                continue
            rest = [a for a in argv[argv.index(verb) + 1 :] if not a.startswith("-")]
            assert rest, f"no service after {verb}: {argv}"
            assert rest[0] in services, (
                f"{' '.join(argv)}: no service {rest[0]!r} in {compose_file} "
                f"(has {sorted(services)})"
            )
            checked += 1
    assert checked, "expected a docker compose run/exec command"


def test_no_whole_tree_format() -> None:
    for line in _bash_lines():
        argv = shlex.split(line)
        if argv[:2] == ["ruff", "format"]:
            assert "." not in argv[2:], line


def test_no_empty_links() -> None:
    text = CONTRIBUTING.read_text(encoding="utf-8")
    assert "](#)" not in text

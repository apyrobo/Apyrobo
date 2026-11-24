"""The README's docker compose lines name a compose file and services that exist."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.S)


def _compose_lines() -> list[str]:
    lines = []
    for block in BASH_BLOCK.findall(README):
        joined = block.replace("\\\n", " ")
        for line in joined.splitlines():
            line = line.split("#", 1)[0].strip()
            if line.startswith(("docker compose", "docker-compose")):
                lines.append(line)
    return lines


def test_readme_has_compose_lines():
    assert _compose_lines(), "expected the README to show at least one docker compose line"


@pytest.mark.parametrize("line", _compose_lines())
def test_compose_line_names_real_file_and_services(line):
    # Run from the repo root as the README says: there is no compose file there,
    # so each line must name one with -f.
    assert line.startswith("docker compose "), f"use 'docker compose', not docker-compose: {line}"
    words = shlex.split(line)
    assert "-f" in words, f"no -f: no compose file at the repo root: {line}"
    compose_file = ROOT / words[words.index("-f") + 1]
    assert compose_file.is_file(), f"{compose_file} does not exist"
    services = yaml.safe_load(compose_file.read_text(encoding="utf-8"))["services"]

    profiles = [words[i + 1] for i, w in enumerate(words) if w == "--profile"]
    for profile in profiles:
        assert any(
            profile in (svc.get("profiles") or []) for svc in services.values()
        ), f"no service has profile {profile!r}"

    for verb in ("up", "run"):
        if verb in words:
            named = [w for w in words[words.index(verb) + 1 :] if not w.startswith("-")]
            for name in named:
                assert name in services, f"{name!r} is not a service in {compose_file.name}"


def test_first_real_robot_points_to_the_guide_not_the_gateway():
    section = README.split("## Connecting to a Robot", 1)[1].split("\n## ", 1)[0]
    assert "docs/first-real-robot.md" in section
    # The gateway image is python:3.10-slim with no ROS 2.
    runtime = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.10-slim AS runtime" in runtime
    assert "up apyrobo-api" not in section

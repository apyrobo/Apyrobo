"""docs/skill_packaging.md: the test-skill commands it shows must work.

These tests build the guide's own files and run every test-skill command the
guide shows: the bare skill ID against the guide's package installed (an
apyrobo.skills entry point on the path), the .py file forms against the file.
"""

from __future__ import annotations

import os
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


def _test_skill(
    cwd: Path, *argv: str, pythonpath: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if pythonpath is not None:
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(pythonpath), env.get("PYTHONPATH", "")) if p
        )
    return subprocess.run(
        [sys.executable, "-m", "apyrobo", "test-skill", *argv],
        cwd=cwd, capture_output=True, text=True, timeout=120, env=env,
    )


def _shown_test_skill_commands() -> list[list[str]]:
    cmds = []
    for line in _guide().splitlines():
        line = line.strip()
        if line.startswith("apyrobo test-skill "):
            cmds.append(re.findall(r"'[^']*'|\S+", line)[2:])
    return [[a[1:-1] if a.startswith("'") else a for a in c] for c in cmds]


def _init_source() -> str:
    """The guide's __init__.py block (section 4)."""
    for lang, body in re.findall(r"```(\w+)\n(.*?)```", _guide(), re.S):
        if lang == "python" and body.startswith("# apyrobo_skills_myrobot/__init__.py"):
            return body
    raise AssertionError("__init__.py block not found in docs/skill_packaging.md")


def _install_guide_package(site: Path) -> None:
    """Lay the guide's package and an apyrobo.skills entry point into ``site``."""
    pkg = site / "apyrobo_skills_myrobot"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(_init_source(), encoding="utf-8")
    (pkg / "navigation.py").write_text(_skill_source(), encoding="utf-8")
    dist = site / "apyrobo_skills_myrobot-1.0.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: apyrobo-skills-myrobot\nVersion: 1.0.0\n",
        encoding="utf-8",
    )
    (dist / "entry_points.txt").write_text(
        "[apyrobo.skills]\nmyrobot = apyrobo_skills_myrobot:register\n", encoding="utf-8",
    )


def test_guide_shows_the_packaged_skill_by_id():
    assert any(argv[0] == "go_to_waypoint" for argv in _shown_test_skill_commands())


@pytest.mark.parametrize("argv", _shown_test_skill_commands())
def test_every_test_skill_command_shown_runs(tmp_path, argv):
    (tmp_path / "go_to_waypoint.py").write_text(_skill_source(), encoding="utf-8")
    site = tmp_path / "site"
    _install_guide_package(site)
    proc = _test_skill(tmp_path, *argv, pythonpath=site)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "go_to_waypoint" in proc.stdout
    assert "not found" not in proc.stderr


def test_params_from_the_guide_reach_the_skill(tmp_path):
    (tmp_path / "go_to_waypoint.py").write_text(_skill_source(), encoding="utf-8")
    proc = _test_skill(tmp_path, "go_to_waypoint.py", "--params", '{"waypoint_id": "wp_42"}')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "'wp_42'" in proc.stdout


def test_packaged_skill_id_is_not_found_without_the_package(tmp_path):
    by_id = _test_skill(tmp_path, "go_to_waypoint")
    assert by_id.returncode != 0
    assert "'go_to_waypoint' not found" in by_id.stderr


def test_module_file_runs_its_only_skill(tmp_path):
    pkg = tmp_path / "apyrobo_skills_myrobot"
    pkg.mkdir()
    (pkg / "navigation.py").write_text(_skill_source(), encoding="utf-8")
    proc = _test_skill(tmp_path, "apyrobo_skills_myrobot/navigation.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Skill:    go_to_waypoint" in proc.stdout


# ---------------------------------------------------------------------------
# Section 6/7: the registry commands the guide shows must work as written.
# ---------------------------------------------------------------------------

def _shown_skill_commands(sub: str) -> list[str]:
    """Every ``apyrobo skill <sub>`` command in the guide, continuation lines joined."""
    cmds = []
    for _lang, body in re.findall(r"```(\w*)\n(.*?)```", _guide(), re.S):
        joined = re.sub(r"\\\n\s*", " ", body)
        for line in joined.splitlines():
            line = line.split("  #")[0].strip()
            if line.startswith(f"apyrobo skill {sub} "):
                cmds.append(line)
    return cmds


class _StubRegistry:
    """A tiny HTTP registry that records requests and answers like the real one."""

    def __init__(self) -> None:
        import http.server
        import json
        import threading

        self.requests: list[tuple[str, str, dict, object]] = []
        stub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _reply(self, code: int, data: object) -> None:
                raw = json.dumps(data).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n).decode())
                stub.requests.append(("POST", self.path, dict(self.headers), body))
                pkg = body["package"]
                self._reply(201, {"status": "published", "name": pkg["name"],
                                  "version": pkg["version"]})

            def do_GET(self) -> None:  # noqa: N802
                stub.requests.append(("GET", self.path, dict(self.headers), None))
                self._reply(200, [{"name": "apyrobo-skills-myrobot", "version": "1.0.0",
                                   "description": "MyRobot skill pack for APYROBO",
                                   "tags": []}])

            def log_message(self, *args: object) -> None:
                pass

        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub_registry():
    reg = _StubRegistry()
    yield reg
    reg.close()


def _run_shown(cmd: str, registry: str, token: str) -> subprocess.CompletedProcess[str]:
    import shlex

    argv = [a.replace("$APYROBO_REGISTRY_TOKEN", token) for a in shlex.split(cmd)]
    assert argv[0] == "apyrobo"
    return subprocess.run(
        [sys.executable, "-m", "apyrobo", *argv[1:], "--registry", registry],
        capture_output=True, text=True, timeout=120,
    )


def test_guide_shows_one_publish_and_one_search():
    assert len(_shown_skill_commands("publish")) == 1
    assert _shown_skill_commands("search")


def test_guide_tells_how_to_start_the_registry_it_publishes_to():
    text = _guide()
    assert "apyrobo registry start" in text
    assert "pip install 'apyrobo[registry]'" in text
    assert "export APYROBO_REGISTRY_TOKEN=" in text
    assert "hosted registry" not in text


def test_publish_command_shown_publishes_the_guide_package(stub_registry):
    from apyrobo.registry.models import PublishRequest

    (cmd,) = _shown_skill_commands("publish")
    proc = _run_shown(cmd, stub_registry.url, "change-me")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Published: apyrobo-skills-myrobot v1.0.0" in proc.stdout

    (method, path, headers, body) = stub_registry.requests[-1]
    assert (method, path) == ("POST", "/skills")
    auth = {k.lower(): v for k, v in headers.items()}["authorization"]
    assert auth == "Bearer change-me"
    # What the command sends is what `apyrobo registry start` accepts.
    req = PublishRequest.model_validate(body)
    assert req.package.name == "apyrobo-skills-myrobot"
    assert req.package.version == "1.0.0"
    assert req.package.description == "MyRobot skill pack for APYROBO"


@pytest.mark.parametrize("cmd", _shown_skill_commands("search"))
def test_search_command_shown_finds_the_published_package(stub_registry, cmd):
    proc = _run_shown(cmd, stub_registry.url, "")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "apyrobo-skills-myrobot" in proc.stdout
    assert stub_registry.requests[-1][1] == "/search?q=myrobot"

"""apyrobo doctor's Docker and skill-registry warnings say what each is for.

A new user on mock:// (or a robot with ROS 2) needs neither Docker nor a local
skill registry, so both warnings say they are optional and what needs them,
and each Docker case names its own cause instead of always "Docker not found".
"""

from __future__ import annotations

import subprocess
import sys
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from apyrobo import cli
from apyrobo.cli import _check_docker, _check_skill_registry


def _run_returning(returncode: int, stderr: bytes = b""):
    return lambda *a, **kw: SimpleNamespace(returncode=returncode, stderr=stderr, stdout=b"")


def _run_raising(exc: BaseException):
    def run(*a, **kw):
        raise exc
    return run


class TestDockerCheck:
    def test_available_passes(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _run_returning(0))
        assert _check_docker().status == "pass"

    @pytest.mark.parametrize(
        "run",
        [
            _run_raising(FileNotFoundError()),
            _run_raising(subprocess.TimeoutExpired("docker", 3)),
            _run_returning(1, b"Cannot connect to the Docker daemon\n"),
        ],
        ids=["missing", "timeout", "daemon-down"],
    )
    def test_every_warning_says_docker_is_optional_and_for_what(self, monkeypatch, run):
        monkeypatch.setattr(subprocess, "run", run)
        result = _check_docker()
        assert result.status == "warn"
        assert "Optional" in result.hint
        assert "docs/QUICKSTART.md" in result.hint
        assert "mock://" in result.hint and "ros2://" in result.hint
        # The old hint blamed "ros2:// integration tests", which users never run.
        assert "integration tests" not in result.hint

    def test_missing_says_not_found(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _run_raising(FileNotFoundError()))
        assert _check_docker().message == "Docker not found"

    def test_daemon_down_does_not_say_not_found_and_quotes_docker(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            _run_returning(1, b"\nCannot connect to the Docker daemon at unix:///var/run/docker.sock\n"),
        )
        result = _check_docker()
        assert "not found" not in result.message.lower()
        assert "not found" not in result.hint.lower()
        assert "installed" in result.message
        assert "Cannot connect to the Docker daemon" in result.hint
        assert "Start the Docker daemon" in result.hint

    def test_daemon_down_without_stderr_names_the_exit_code(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _run_returning(125))
        assert "exit code 125" in _check_docker().hint

    def test_timeout_says_it_timed_out_not_missing(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run", _run_raising(subprocess.TimeoutExpired("docker", 3))
        )
        result = _check_docker()
        assert "3 s" in result.message
        assert "not found" not in result.hint.lower()
        assert "timed out" in result.hint


class TestSkillRegistryCheck:
    @pytest.fixture
    def unreachable(self, monkeypatch):
        def _raise(*a, **kw):
            raise urllib.error.URLError("connection refused")
        monkeypatch.setattr(urllib.request, "urlopen", _raise)

    def test_warning_says_registry_is_optional_and_what_uses_it(self, unreachable):
        result = _check_skill_registry()
        assert result.status == "warn"
        assert result.hint.startswith("Optional")
        assert "apyrobo registry search/install/publish" in result.hint
        assert "apyrobo skill search/publish" in result.hint
        assert "apyrobo registry start" in result.hint

    def test_without_uvicorn_the_hint_installs_the_extra_first(self, unreachable, monkeypatch):
        monkeypatch.setattr(cli, "_uvicorn_installed", lambda: False)
        hint = _check_skill_registry().hint
        assert "pip install 'apyrobo[registry]', then apyrobo registry start" in hint

    def test_with_uvicorn_the_hint_is_just_the_start_command(self, unreachable, monkeypatch):
        monkeypatch.setattr(cli, "_uvicorn_installed", lambda: True)
        hint = _check_skill_registry().hint
        assert "pip install" not in hint
        assert hint.endswith("To run one here: apyrobo registry start")

    @pytest.mark.parametrize(
        "argv",
        [
            ["registry", "search", "--help"],
            ["registry", "install", "--help"],
            ["registry", "publish", "--help"],
            ["skill", "search", "--help"],
            ["skill", "publish", "--help"],
        ],
    )
    def test_hint_names_only_commands_the_cli_has(self, argv, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["apyrobo", *argv])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
        assert "usage: apyrobo " + " ".join(argv[:2]) in capsys.readouterr().out

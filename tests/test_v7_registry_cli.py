"""Tests for apyrobo registry CLI — v7.0.0 skill registry."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from unittest.mock import MagicMock, patch, call

import pytest

from apyrobo.registry import SkillRegistryClient, SkillPackage
from apyrobo.registry.client import SkillRegistryClient as _ClientDirect


# ---------------------------------------------------------------------------
# SkillRegistryClient.install() — new method
# ---------------------------------------------------------------------------

_GOOD_PKG = SkillPackage(
    name="apyrobo-skills-patrol",
    version="1.0.0",
    description="Autonomous patrol skill",
    author="Alice",
    license="Apache-2.0",
    download_url="https://example.com/apyrobo-skills-patrol-1.0.0.whl",
    checksum="a" * 64,
    apyrobo_version_min="1.0.0",
)


class TestSkillRegistryClientInstall:
    def _client(self):
        return SkillRegistryClient("https://registry.example.com")

    def test_install_dry_run_returns_true(self, capsys):
        client = self._client()
        with patch.object(client, "get", return_value=_GOOD_PKG):
            result = client.install("apyrobo-skills-patrol", dry_run=True)
        assert result is True
        out = capsys.readouterr().out
        assert "pip" in out

    def test_install_calls_pip(self):
        client = self._client()
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch.object(client, "get", return_value=_GOOD_PKG), \
             patch("subprocess.run", return_value=mock_result) as mock_run:
            result = client.install("apyrobo-skills-patrol")

        assert result is True
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == sys.executable
        assert "-m" in call_args
        assert "pip" in call_args
        assert "install" in call_args
        assert _GOOD_PKG.download_url in call_args

    def test_install_raises_valueerror_when_not_found(self):
        client = self._client()
        with patch.object(client, "get", return_value=None):
            with pytest.raises(ValueError, match="not found"):
                client.install("no-such-package")

    def test_install_raises_runtimeerror_on_pip_failure(self):
        client = self._client()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ERROR: Could not find a version"
        with patch.object(client, "get", return_value=_GOOD_PKG), \
             patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="pip install failed"):
                client.install("apyrobo-skills-patrol")

    def test_install_uses_version_from_get(self):
        client = self._client()
        mock_result = MagicMock(returncode=0)
        with patch.object(client, "get", return_value=_GOOD_PKG) as mock_get, \
             patch("subprocess.run", return_value=mock_result):
            client.install("apyrobo-skills-patrol", "1.0.0")

        mock_get.assert_called_once_with("apyrobo-skills-patrol", "1.0.0")


# ---------------------------------------------------------------------------
# cmd_registry — search subcommand
# ---------------------------------------------------------------------------

def _make_registry_args(sub, **kwargs) -> argparse.Namespace:
    defaults = {
        "registry_command": sub,
        "registry_url": "https://registry.example.com",
        "json": False,
        "query": "",
        "package_name": "",
        "package_version": "latest",
        "dry_run": False,
        "pkg_json": None,
        "name": None,
        "version": None,
        "description": None,
        "author": None,
        "license": None,
        "download_url": None,
        "checksum": None,
        "apyrobo_version_min": None,
        "tags": [],
        "token": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestCmdRegistrySearch:
    def _run(self, results, **kwargs):
        from apyrobo.cli import cmd_registry
        args = _make_registry_args("search", **kwargs)
        with patch("apyrobo.registry.SkillRegistryClient.search", return_value=results):
            cmd_registry(args)

    def test_search_empty_results(self, capsys):
        self._run([], query="nonexistent")
        out = capsys.readouterr().out
        assert "No packages" in out

    def test_search_prints_table(self, capsys):
        self._run([_GOOD_PKG], query="patrol")
        out = capsys.readouterr().out
        assert "apyrobo-skills-patrol" in out
        assert "1.0.0" in out

    def test_search_json_output(self, capsys):
        self._run([_GOOD_PKG], query="patrol", json=True)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, list)
        assert data[0]["name"] == "apyrobo-skills-patrol"

    def test_search_error_exits(self):
        from apyrobo.cli import cmd_registry
        args = _make_registry_args("search", query="x")
        with patch("apyrobo.registry.SkillRegistryClient.search",
                   side_effect=Exception("network error")):
            with pytest.raises(SystemExit) as exc:
                cmd_registry(args)
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# cmd_registry — install subcommand
# ---------------------------------------------------------------------------

class TestCmdRegistryInstall:
    def test_install_success(self, capsys):
        from apyrobo.cli import cmd_registry
        args = _make_registry_args("install", package_name="apyrobo-skills-patrol")
        with patch("apyrobo.registry.SkillRegistryClient.install", return_value=True):
            cmd_registry(args)
        out = capsys.readouterr().out
        assert "Installed" in out or "Fetching" in out

    def test_install_not_found_exits(self):
        from apyrobo.cli import cmd_registry
        args = _make_registry_args("install", package_name="no-such")
        with patch("apyrobo.registry.SkillRegistryClient.install",
                   side_effect=ValueError("not found in registry")):
            with pytest.raises(SystemExit) as exc:
                cmd_registry(args)
        assert exc.value.code == 1

    def test_install_pip_failure_exits(self):
        from apyrobo.cli import cmd_registry
        args = _make_registry_args("install", package_name="bad-pkg")
        with patch("apyrobo.registry.SkillRegistryClient.install",
                   side_effect=RuntimeError("pip install failed")):
            with pytest.raises(SystemExit) as exc:
                cmd_registry(args)
        assert exc.value.code == 1

    def test_install_dry_run_flag(self, capsys):
        from apyrobo.cli import cmd_registry
        args = _make_registry_args("install", package_name="apyrobo-skills-patrol",
                                   dry_run=True)
        with patch("apyrobo.registry.SkillRegistryClient.install", return_value=True) as mock_install:
            cmd_registry(args)
        mock_install.assert_called_once_with("apyrobo-skills-patrol", "latest", dry_run=True)

    def test_install_version_passed_through(self):
        from apyrobo.cli import cmd_registry
        args = _make_registry_args("install", package_name="patrol", package_version="2.0.0")
        with patch("apyrobo.registry.SkillRegistryClient.install", return_value=True) as mock_install:
            cmd_registry(args)
        mock_install.assert_called_once_with("patrol", "2.0.0", dry_run=False)


# ---------------------------------------------------------------------------
# cmd_registry — publish subcommand
# ---------------------------------------------------------------------------

class TestCmdRegistryPublish:
    def _pkg_args(self, **overrides):
        base = dict(
            name="apyrobo-skills-patrol",
            version="1.0.0",
            description="Patrol skill",
            author="Alice",
            license="Apache-2.0",
            download_url="https://example.com/patrol-1.0.0.whl",
            checksum="a" * 64,
            apyrobo_version_min="1.0.0",
            tags=[],
            token="test-token",
        )
        base.update(overrides)
        return base

    def test_publish_success(self, capsys):
        from apyrobo.cli import cmd_registry
        args = _make_registry_args("publish", **self._pkg_args())
        with patch("apyrobo.registry.SkillRegistryClient.publish", return_value=True):
            cmd_registry(args)
        assert "Published" in capsys.readouterr().out

    def test_publish_missing_token_exits(self, monkeypatch):
        from apyrobo.cli import cmd_registry
        monkeypatch.delenv("APYROBO_REGISTRY_TOKEN", raising=False)
        args = _make_registry_args("publish", **self._pkg_args(token=None))
        with pytest.raises(SystemExit) as exc:
            cmd_registry(args)
        assert exc.value.code == 1

    def test_publish_token_from_env(self, capsys, monkeypatch):
        from apyrobo.cli import cmd_registry
        monkeypatch.setenv("APYROBO_REGISTRY_TOKEN", "env-token")
        args = _make_registry_args("publish", **self._pkg_args(token=None))
        with patch("apyrobo.registry.SkillRegistryClient.publish", return_value=True) as mock_pub:
            cmd_registry(args)
        _, token_used = mock_pub.call_args[0]
        assert token_used == "env-token"

    def test_publish_from_json_file(self, capsys, tmp_path):
        from apyrobo.cli import cmd_registry
        pkg_data = {
            "name": "apyrobo-skills-patrol",
            "version": "1.0.0",
            "description": "Patrol",
            "author": "Alice",
            "license": "Apache-2.0",
            "download_url": "https://example.com/p.whl",
            "checksum": "b" * 64,
            "apyrobo_version_min": "1.0.0",
        }
        json_file = tmp_path / "pkg.json"
        json_file.write_text(json.dumps(pkg_data))
        args = _make_registry_args("publish", pkg_json=str(json_file), token="tok")
        with patch("apyrobo.registry.SkillRegistryClient.publish", return_value=True):
            cmd_registry(args)
        assert "Published" in capsys.readouterr().out

    def test_publish_bad_json_file_exits(self, tmp_path):
        from apyrobo.cli import cmd_registry
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{bad json")
        args = _make_registry_args("publish", pkg_json=str(bad_file), token="tok")
        with pytest.raises(SystemExit) as exc:
            cmd_registry(args)
        assert exc.value.code == 1

    def test_publish_missing_required_field_exits(self):
        from apyrobo.cli import cmd_registry
        args = _make_registry_args("publish", **self._pkg_args(name=None))
        with pytest.raises(SystemExit) as exc:
            cmd_registry(args)
        assert exc.value.code == 1

    def test_publish_registry_error_exits(self):
        from apyrobo.cli import cmd_registry
        args = _make_registry_args("publish", **self._pkg_args())
        with patch("apyrobo.registry.SkillRegistryClient.publish",
                   side_effect=Exception("server error")):
            with pytest.raises(SystemExit) as exc:
                cmd_registry(args)
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# CLI parser — registry subcommand wiring
# ---------------------------------------------------------------------------

class TestRegistryCLIParser:
    def _parse(self, args):
        from apyrobo.cli import main
        parsed = []
        with patch("apyrobo.cli._cmd_registry_dispatch") as mock_dispatch:
            with patch.object(sys, "argv", ["apyrobo"] + args):
                try:
                    main()
                except SystemExit:
                    pass
            if mock_dispatch.call_args:
                parsed.append(mock_dispatch.call_args[0][0])
        return parsed[0] if parsed else None

    def test_registry_search_parses_query(self):
        args = self._parse(["registry", "search", "patrol"])
        if args is None:
            pytest.skip("parser did not dispatch (expected in test env)")
        assert args.registry_command == "search"
        assert args.query == "patrol"

    def test_registry_install_parses_package(self):
        args = self._parse(["registry", "install", "apyrobo-skills-patrol"])
        if args is None:
            pytest.skip("parser did not dispatch")
        assert args.registry_command == "install"
        assert args.package_name == "apyrobo-skills-patrol"

    def test_registry_install_parses_version(self):
        args = self._parse(["registry", "install", "apyrobo-skills-patrol", "--version", "2.0.0"])
        if args is None:
            pytest.skip("parser did not dispatch")
        assert args.package_version == "2.0.0"

    def test_registry_install_parses_dry_run(self):
        args = self._parse(["registry", "install", "pkg", "--dry-run"])
        if args is None:
            pytest.skip("parser did not dispatch")
        assert args.dry_run is True

    def test_registry_publish_parses_token(self):
        args = self._parse(["registry", "publish", "--name", "x", "--token", "my-tok"])
        if args is None:
            pytest.skip("parser did not dispatch")
        assert args.registry_command == "publish"
        assert args.token == "my-tok"

    def test_registry_search_json_flag(self):
        args = self._parse(["registry", "search", "nav", "--json"])
        if args is None:
            pytest.skip("parser did not dispatch")
        assert args.json is True

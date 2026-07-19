"""Tests for the bundled registry seed index and offline fallback."""
from __future__ import annotations

import urllib.error
from unittest.mock import patch

import pytest

from apyrobo.registry import SkillPackage, SkillRegistryClient
from apyrobo.registry.client import load_seed_index


# ---------------------------------------------------------------------------
# Seed index contents
# ---------------------------------------------------------------------------

class TestSeedIndex:
    def test_loads_and_validates(self):
        packages = load_seed_index()
        assert len(packages) >= 2
        names = {pkg.name for pkg in packages}
        assert "apyrobo-skills-ros-nav" in names
        assert "apyrobo-client-ts" in names

    def test_only_real_packages(self):
        # The seed index is the list of packages that actually work today —
        # scaffolds (packages/README.md) must not appear in it.
        scaffolds = {
            "apyrobo-skills-ur", "apyrobo-skills-spot", "apyrobo-skills-franka",
            "apyrobo-skills-drone-px4", "apyrobo-skills-agv",
            "apyrobo-skills-turtlebot4",
        }
        assert scaffolds.isdisjoint({pkg.name for pkg in load_seed_index()})

    def test_kinds_are_correct(self):
        by_name = {pkg.name: pkg for pkg in load_seed_index()}
        assert by_name["apyrobo-skills-ros-nav"].kind == "python"
        assert by_name["apyrobo-client-ts"].kind == "npm"


# ---------------------------------------------------------------------------
# Model additions
# ---------------------------------------------------------------------------

class TestModel:
    def _pkg(self, **overrides):
        data = dict(
            name="x", version="1.0.0", description="d", author="a",
            license="Apache-2.0", download_url="https://example.com/x.whl",
            checksum="a" * 64, apyrobo_version_min="1.0.0",
        )
        data.update(overrides)
        return SkillPackage(**data)

    def test_empty_checksum_allowed_for_vcs(self):
        assert self._pkg(checksum="").checksum == ""

    def test_bad_checksum_still_rejected(self):
        with pytest.raises(ValueError):
            self._pkg(checksum="nothex")

    def test_kind_defaults_to_python(self):
        assert self._pkg().kind == "python"

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError):
            self._pkg(kind="cargo")


# ---------------------------------------------------------------------------
# Offline fallback
# ---------------------------------------------------------------------------

def _unreachable(self, path):
    raise urllib.error.URLError("no route to host")


class TestOfflineFallback:
    def _client(self):
        return SkillRegistryClient("https://registry.example.invalid")

    def test_search_falls_back_to_seed(self, capsys):
        with patch.object(SkillRegistryClient, "_get", _unreachable):
            results = self._client().search("nav")
        assert any(pkg.name == "apyrobo-skills-ros-nav" for pkg in results)
        assert "seed index" in capsys.readouterr().err

    def test_search_empty_query_returns_all(self):
        with patch.object(SkillRegistryClient, "_get", _unreachable):
            results = self._client().search("")
        assert len(results) == len(load_seed_index())

    def test_search_matches_tags(self):
        with patch.object(SkillRegistryClient, "_get", _unreachable):
            results = self._client().search("wire-protocol")
        assert [pkg.name for pkg in results] == ["apyrobo-client-ts"]

    def test_list_all_falls_back_to_seed(self):
        with patch.object(SkillRegistryClient, "_get", _unreachable):
            results = self._client().list_all()
        assert len(results) == len(load_seed_index())

    def test_get_falls_back_to_seed(self):
        with patch.object(SkillRegistryClient, "_get", _unreachable):
            pkg = self._client().get("apyrobo-skills-ros-nav")
        assert pkg is not None
        assert pkg.download_url.startswith("git+https://")

    def test_http_error_does_not_fall_back(self):
        def _500(self, path):
            raise urllib.error.HTTPError(
                "https://registry.example.invalid/search", 500, "boom", None, None
            )
        with patch.object(SkillRegistryClient, "_get", _500):
            assert self._client().search("nav") == []

    def test_install_dry_run_resolves_from_seed(self, capsys):
        with patch.object(SkillRegistryClient, "_get", _unreachable):
            ok = self._client().install("apyrobo-skills-ros-nav", dry_run=True)
        assert ok is True
        assert "git+https://" in capsys.readouterr().out

    def test_install_npm_package_refused_with_hint(self):
        with (
            patch.object(SkillRegistryClient, "_get", _unreachable),
            pytest.raises(ValueError, match="npm"),
        ):
            self._client().install("apyrobo-client-ts")


# ---------------------------------------------------------------------------
# Server seeding
# ---------------------------------------------------------------------------

class TestServerSeed:
    def test_create_app_seed_preloads_store(self):
        # Not importorskip: pytest 8 only swallows ModuleNotFoundError, and
        # registry.server raises a custom ImportError without FastAPI.
        try:
            from apyrobo.registry import server
        except ImportError:
            pytest.skip("fastapi not installed")
        original = dict(server._store)
        server._store.clear()
        try:
            server.create_app(seed=True)
            assert "apyrobo-skills-ros-nav" in server._store
            assert "apyrobo-client-ts" in server._store
        finally:
            server._store.clear()
            server._store.update(original)

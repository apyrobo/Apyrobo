"""Tests for the apyrobo skill registry client and models."""

import pytest
from pydantic import ValidationError

from apyrobo.registry.models import SkillPackage, SkillVersion, PublishRequest


VALID_PKG = dict(
    name="apyrobo-patrol",
    version="1.0.0",
    description="Patrol skill for mobile robots.",
    author="Acme Robotics",
    license="Apache-2.0",
    tags=["navigation", "patrol"],
    download_url="https://example.com/apyrobo-patrol-1.0.0.tar.gz",
    checksum="a" * 64,
    apyrobo_version_min="1.0.0",
)


class TestSkillPackage:
    def test_valid_package(self):
        pkg = SkillPackage(**VALID_PKG)
        assert pkg.name == "apyrobo-patrol"
        assert pkg.version == "1.0.0"
        assert len(pkg.tags) == 2

    def test_invalid_checksum_short(self):
        bad = {**VALID_PKG, "checksum": "abc123"}
        with pytest.raises(ValidationError):
            SkillPackage(**bad)

    def test_invalid_checksum_non_hex(self):
        bad = {**VALID_PKG, "checksum": "z" * 64}
        with pytest.raises(ValidationError):
            SkillPackage(**bad)

    def test_checksum_normalized_lowercase(self):
        pkg = SkillPackage(**{**VALID_PKG, "checksum": "A" * 64})
        assert pkg.checksum == "a" * 64

    def test_invalid_version(self):
        bad = {**VALID_PKG, "version": "notaversion"}
        with pytest.raises(ValidationError):
            SkillPackage(**bad)

    def test_empty_tags_allowed(self):
        pkg = SkillPackage(**{**VALID_PKG, "tags": []})
        assert pkg.tags == []


class TestSkillVersion:
    def test_defaults(self):
        v = SkillVersion(version="1.0.0", published_at="2026-01-01T00:00:00Z")
        assert v.yanked is False
        assert v.yanked_reason == ""

    def test_yanked(self):
        v = SkillVersion(
            version="1.0.0",
            published_at="2026-01-01T00:00:00Z",
            yanked=True,
            yanked_reason="Critical bug",
        )
        assert v.yanked is True
        assert "Critical" in v.yanked_reason


class TestPublishRequest:
    def test_valid(self):
        pkg = SkillPackage(**VALID_PKG)
        req = PublishRequest(package=pkg, token="secret")
        assert req.token == "secret"

    def test_missing_token(self):
        pkg = SkillPackage(**VALID_PKG)
        with pytest.raises(ValidationError):
            PublishRequest(package=pkg)


class TestSkillRegistryClient:
    """Test client without real network calls."""

    def test_import(self):
        from apyrobo.registry.client import SkillRegistryClient
        client = SkillRegistryClient("http://localhost:9999")
        assert client.base_url == "http://localhost:9999"

    def test_trailing_slash_stripped(self):
        from apyrobo.registry.client import SkillRegistryClient
        client = SkillRegistryClient("http://localhost:9999/")
        assert not client.base_url.endswith("/")

    def test_get_returns_none_on_error(self):
        from apyrobo.registry.client import SkillRegistryClient
        client = SkillRegistryClient("http://localhost:0")  # unreachable
        result = client.get("some-package")
        assert result is None

    def test_search_returns_empty_on_error(self):
        from apyrobo.registry.client import SkillRegistryClient
        client = SkillRegistryClient("http://localhost:0")
        result = client.search("navigation")
        assert result == []

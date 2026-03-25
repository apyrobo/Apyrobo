"""Pydantic models for the APYROBO skill registry."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SkillPackage(BaseModel):
    """Metadata for a skill package published to the registry."""

    name: str = Field(..., description="Unique package name (e.g. 'apyrobo-patrol')")
    version: str = Field(..., description="SemVer version string")
    description: str = Field(..., description="One-paragraph description of what the skill does")
    author: str = Field(..., description="Author name or organisation")
    license: str = Field(..., description="SPDX license identifier (e.g. 'Apache-2.0')")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")
    download_url: str = Field(..., description="URL to download the wheel or tarball")
    checksum: str = Field(..., description="SHA-256 hex digest of the download artifact")
    apyrobo_version_min: str = Field(
        ..., description="Minimum APYROBO version required (e.g. '1.0.0')"
    )

    @field_validator("checksum")
    @classmethod
    def checksum_is_hex(cls, v: str) -> str:
        if len(v) != 64 or not all(c in "0123456789abcdefABCDEF" for c in v):
            raise ValueError("checksum must be a 64-character SHA-256 hex digest")
        return v.lower()

    @field_validator("version", "apyrobo_version_min")
    @classmethod
    def version_is_semver(cls, v: str) -> str:
        parts = v.split(".")
        if len(parts) < 2:
            raise ValueError(f"Version {v!r} must have at least major.minor parts")
        return v


class SkillVersion(BaseModel):
    """A single version entry in a skill's version history."""

    version: str
    published_at: str = Field(..., description="ISO 8601 timestamp")
    yanked: bool = False
    yanked_reason: str = ""


class PublishRequest(BaseModel):
    """Request body for POST /skills."""

    package: SkillPackage
    token: str = Field(..., description="Publisher authentication token")

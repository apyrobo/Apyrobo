"""FastAPI skill registry server.

Start with::

    uvicorn apyrobo.registry.server:app --host 0.0.0.0 --port 8765

Or programmatically::

    from apyrobo.registry.server import create_app
    app = create_app(auth_token="secret")
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

try:
    from fastapi import Depends, FastAPI, HTTPException, Query, status
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "apyrobo.registry.server requires FastAPI — install it with: pip install fastapi"
    ) from exc

from apyrobo.registry.models import PublishRequest, SkillPackage, SkillVersion

# ---------------------------------------------------------------------------
# In-memory store (swap for a real DB in production)
# ---------------------------------------------------------------------------

_store: dict[str, dict[str, SkillPackage]] = {}  # name -> {version -> package}

security = HTTPBearer(auto_error=False)

_DEFAULT_AUTH_TOKEN: str | None = os.environ.get("APYROBO_REGISTRY_TOKEN")


def _check_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> None:
    expected = _DEFAULT_AUTH_TOKEN
    if expected and (credentials is None or credentials.credentials != expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
        )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(auth_token: str | None = None) -> FastAPI:
    """Create and configure the registry FastAPI application.

    Args:
        auth_token: If provided, ``POST /skills`` requires this bearer token.
    """
    global _DEFAULT_AUTH_TOKEN
    if auth_token is not None:
        _DEFAULT_AUTH_TOKEN = auth_token

    app = FastAPI(
        title="APYROBO Skill Registry",
        description="Public registry for discovering and publishing APYROBO skill packages.",
        version="1.0.0",
    )

    # ------------------------------------------------------------------
    # GET /skills
    # ------------------------------------------------------------------

    @app.get("/skills", response_model=list[SkillPackage], summary="List all published skills")
    def list_skills() -> list[SkillPackage]:
        """Return the latest version of every skill in the registry."""
        result = []
        for versions in _store.values():
            latest = _latest(versions)
            if latest:
                result.append(latest)
        return result

    # ------------------------------------------------------------------
    # GET /skills/{name}
    # ------------------------------------------------------------------

    @app.get(
        "/skills/{name}",
        response_model=SkillPackage,
        summary="Get latest info for a specific skill",
    )
    def get_skill(name: str) -> SkillPackage:
        pkg = _latest(_store.get(name, {}))
        if pkg is None:
            raise HTTPException(status_code=404, detail=f"Skill {name!r} not found")
        return pkg

    # ------------------------------------------------------------------
    # GET /skills/{name}/versions
    # ------------------------------------------------------------------

    @app.get(
        "/skills/{name}/versions",
        response_model=list[SkillVersion],
        summary="List all versions of a skill",
    )
    def list_versions(name: str) -> list[SkillVersion]:
        versions = _store.get(name)
        if versions is None:
            raise HTTPException(status_code=404, detail=f"Skill {name!r} not found")
        return [SkillVersion(version=v, published_at="unknown") for v in sorted(versions)]

    # ------------------------------------------------------------------
    # POST /skills
    # ------------------------------------------------------------------

    @app.post(
        "/skills",
        status_code=status.HTTP_201_CREATED,
        summary="Publish a new skill (requires auth token)",
    )
    def publish_skill(
        body: PublishRequest,
        _: Any = Depends(_check_token),
    ) -> dict[str, str]:
        pkg = body.package
        if pkg.name not in _store:
            _store[pkg.name] = {}
        if pkg.version in _store[pkg.name]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Version {pkg.version} of {pkg.name!r} already exists",
            )
        _store[pkg.name][pkg.version] = pkg
        return {"status": "published", "name": pkg.name, "version": pkg.version}

    # ------------------------------------------------------------------
    # GET /search
    # ------------------------------------------------------------------

    @app.get("/search", response_model=list[SkillPackage], summary="Search skills")
    def search_skills(q: str = Query(..., description="Search term")) -> list[SkillPackage]:
        """Search skills by name, tag, or description (case-insensitive substring)."""
        term = q.lower()
        results = []
        for versions in _store.values():
            pkg = _latest(versions)
            if pkg is None:
                continue
            if (
                term in pkg.name.lower()
                or term in pkg.description.lower()
                or any(term in t.lower() for t in pkg.tags)
            ):
                results.append(pkg)
        return results

    return app


# ---------------------------------------------------------------------------
# Default app instance (used by uvicorn)
# ---------------------------------------------------------------------------

app = create_app()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latest(versions: dict[str, SkillPackage]) -> SkillPackage | None:
    if not versions:
        return None
    # Sort by version string — good enough for SemVer if all parts are numeric
    try:
        from packaging.version import Version

        key = sorted(versions, key=Version)[-1]
    except Exception:
        key = sorted(versions)[-1]
    return versions[key]

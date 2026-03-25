"""Client for the APYROBO skill registry."""

from __future__ import annotations

from typing import Optional
import urllib.request
import urllib.parse
import json

from .models import SkillPackage


class SkillRegistryClient:
    """Client for querying and publishing to an APYROBO skill registry.

    Example::

        client = SkillRegistryClient("https://registry.apyrobo.dev")
        results = client.search("navigation")
        for pkg in results:
            print(pkg.name, pkg.version)
    """

    def __init__(self, base_url: str = "https://registry.apyrobo.dev") -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = 10.0

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def list_all(self) -> list[SkillPackage]:
        """Return all skill packages in the registry."""
        data = self._get("/skills")
        return [SkillPackage(**item) for item in data.get("skills", [])]

    def get(self, name: str, version: str = "latest") -> Optional[SkillPackage]:
        """Fetch a specific skill package.

        Args:
            name: Package name.
            version: Version string or ``"latest"``.

        Returns:
            :class:`SkillPackage` if found, ``None`` otherwise.
        """
        try:
            data = self._get(f"/skills/{urllib.parse.quote(name)}")
            if version == "latest":
                return SkillPackage(**data)
            # Filter to the requested version
            versions = data.get("versions", [])
            for v in versions:
                if v.get("version") == version:
                    pkg_data = {**data, "version": version}
                    return SkillPackage(**pkg_data)
            return None
        except Exception:
            return None

    def search(self, query: str) -> list[SkillPackage]:
        """Search skill packages by name, tag, or description.

        Args:
            query: Free-text search query.

        Returns:
            List of matching :class:`SkillPackage` objects.
        """
        try:
            params = urllib.parse.urlencode({"q": query})
            data = self._get(f"/search?{params}")
            return [SkillPackage(**item) for item in data.get("results", [])]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def publish(self, package: SkillPackage, token: str) -> bool:
        """Publish a skill package to the registry.

        Args:
            package: The package to publish.
            token: Publisher authentication token.

        Returns:
            ``True`` on success.

        Raises:
            RuntimeError: If the publish fails.
        """
        payload = {"package": package.model_dump(), "token": token}
        response = self._post("/skills", payload)
        return response.get("status") == "ok"

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode())

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode())

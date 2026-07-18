"""Client for the APYROBO skill registry."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from importlib import resources
from typing import Optional

from .models import SkillPackage


def load_seed_index() -> list[SkillPackage]:
    """Return the bundled seed index: the packages that are real today.

    Shipped in-wheel (``seed_index.json``) so discovery works before any
    hosted registry exists and offline afterwards.
    """
    text = (
        resources.files("apyrobo.registry")
        .joinpath("seed_index.json")
        .read_text(encoding="utf-8")
    )
    data = json.loads(text)
    return [SkillPackage(**item) for item in data.get("packages", [])]


class SkillRegistryClient:
    """Client for querying and publishing to an APYROBO skill registry.

    When the registry is unreachable, read operations fall back to the
    bundled seed index and say so on stderr — search works out of the box,
    with no hosted service required.

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

    def _seed_fallback(self) -> list[SkillPackage]:
        print(
            f"note: registry {self.base_url} unreachable — using the bundled seed index",
            file=sys.stderr,
        )
        return load_seed_index()

    def list_all(self) -> list[SkillPackage]:
        """Return all skill packages in the registry.

        Falls back to the bundled seed index when the registry is
        unreachable.
        """
        try:
            data = self._get("/skills")
        except urllib.error.HTTPError:
            raise  # registry reachable but erroring — don't mask it with the seed
        except (urllib.error.URLError, OSError):
            return self._seed_fallback()
        return [SkillPackage(**item) for item in data.get("skills", [])]

    def get(self, name: str, version: str = "latest") -> Optional[SkillPackage]:
        """Fetch a specific skill package.

        Args:
            name: Package name.
            version: Version string or ``"latest"``.

        Returns:
            :class:`SkillPackage` if found, ``None`` otherwise. Falls back
            to the bundled seed index when the registry is unreachable.
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
        except urllib.error.HTTPError:
            return None
        except (urllib.error.URLError, OSError):
            for pkg in self._seed_fallback():
                if pkg.name == name and version in ("latest", pkg.version):
                    return pkg
            return None
        except Exception:
            return None

    def search(self, query: str) -> list[SkillPackage]:
        """Search skill packages by name, tag, or description.

        Args:
            query: Free-text search query.

        Returns:
            List of matching :class:`SkillPackage` objects. Falls back to
            a substring match over the bundled seed index when the registry
            is unreachable.
        """
        try:
            params = urllib.parse.urlencode({"q": query})
            data = self._get(f"/search?{params}")
            return [SkillPackage(**item) for item in data.get("results", [])]
        except urllib.error.HTTPError:
            return []
        except (urllib.error.URLError, OSError):
            q = query.lower()
            return [
                pkg
                for pkg in self._seed_fallback()
                if not q
                or q in pkg.name.lower()
                or q in pkg.description.lower()
                or any(q in tag.lower() for tag in pkg.tags)
            ]
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

    def install(
        self,
        name: str,
        version: str = "latest",
        *,
        dry_run: bool = False,
    ) -> bool:
        """Download and install a skill package via pip.

        Resolves the package on the registry, then calls
        ``pip install <download_url>`` using the same Python interpreter
        that is currently running.

        Parameters
        ----------
        name:
            Registry package name (e.g. ``"apyrobo-skills-patrol"``).
        version:
            Exact version to install, or ``"latest"`` (default).
        dry_run:
            If True, resolve and print the install command but do not run it.

        Returns
        -------
        bool
            True if installation succeeded (or dry_run was requested).

        Raises
        ------
        ValueError:
            If the package / version is not found in the registry.
        RuntimeError:
            If pip exits with a non-zero return code.
        """
        pkg = self.get(name, version)
        if pkg is None:
            raise ValueError(
                f"Package {name!r} (version={version!r}) not found in registry at {self.base_url}"
            )
        if pkg.kind != "python":
            raise ValueError(
                f"Package {name!r} is a {pkg.kind} package, not pip-installable.\n"
                f"  Get it from: {pkg.download_url}"
            )

        cmd = [sys.executable, "-m", "pip", "install", pkg.download_url]

        if dry_run:
            print("Would run:", " ".join(cmd))
            return True

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"pip install failed for {name!r}:\n{result.stderr.strip()}"
            )
        return True

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

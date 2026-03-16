"""
SK-03: Tests for CLI ``apyrobo pkg`` subcommands.

Covers:
    - pkg install <path> installs and prints skills
    - pkg list shows installed packages
    - pkg search returns scored results
    - pkg validate prints errors or OK
    - pkg remove removes and confirms
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from apyrobo.skills.package import SkillPackage
from apyrobo.skills.registry import SkillRegistry
from apyrobo.skills.skill import Skill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills"


def _run_cli(*args: str, registry_dir: str | None = None) -> subprocess.CompletedProcess:
    """Run ``python -m apyrobo.cli pkg <args>``."""
    cmd = [sys.executable, "-m", "apyrobo.cli", "pkg"]
    if registry_dir:
        cmd.extend(["--registry-dir", registry_dir])
    cmd.extend(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


@pytest.fixture
def tmp_registry(tmp_path: Path) -> Path:
    """Return a temp directory path for a fresh registry."""
    reg_dir = tmp_path / "registry"
    reg_dir.mkdir()
    return reg_dir


# ===========================================================================
# SK-03: CLI pkg command tests
# ===========================================================================


class TestPkgInstall:
    """apyrobo pkg install <path>"""

    def test_install_from_directory(self, tmp_registry: Path) -> None:
        """Installing a package directory prints skill IDs."""
        result = _run_cli(
            "install", str(SKILLS_DIR / "builtin-skills"),
            registry_dir=str(tmp_registry),
        )
        assert result.returncode == 0
        assert "Installed: builtin-skills@1.0.0" in result.stdout
        assert "navigate_to" in result.stdout

    def test_install_force_overwrites(self, tmp_registry: Path) -> None:
        """--force allows reinstalling over existing."""
        src = str(SKILLS_DIR / "builtin-skills")
        _run_cli("install", src, registry_dir=str(tmp_registry))
        result = _run_cli("install", "--force", src, registry_dir=str(tmp_registry))
        assert result.returncode == 0
        assert "Installed:" in result.stdout

    def test_install_conflict_without_force(self, tmp_registry: Path) -> None:
        """Installing the same package twice without --force fails."""
        src = str(SKILLS_DIR / "builtin-skills")
        _run_cli("install", src, registry_dir=str(tmp_registry))
        result = _run_cli("install", src, registry_dir=str(tmp_registry))
        assert result.returncode != 0
        assert "Conflict" in result.stdout


class TestPkgList:
    """apyrobo pkg list"""

    def test_list_empty(self, tmp_registry: Path) -> None:
        """Empty registry prints helpful message."""
        result = _run_cli("list", registry_dir=str(tmp_registry))
        assert result.returncode == 0
        assert "No packages installed" in result.stdout

    def test_list_after_install(self, tmp_registry: Path) -> None:
        """Lists installed packages with version and skill count."""
        _run_cli(
            "install", str(SKILLS_DIR / "builtin-skills"),
            registry_dir=str(tmp_registry),
        )
        result = _run_cli("list", registry_dir=str(tmp_registry))
        assert result.returncode == 0
        assert "builtin-skills@1.0.0" in result.stdout
        assert "7 skills" in result.stdout


class TestPkgSearch:
    """apyrobo pkg search"""

    def test_search_returns_results(self, tmp_registry: Path) -> None:
        """Searching installed packages returns scored results."""
        _run_cli(
            "install", str(SKILLS_DIR / "builtin-skills"),
            registry_dir=str(tmp_registry),
        )
        result = _run_cli("search", "navigation", registry_dir=str(tmp_registry))
        assert result.returncode == 0
        assert "builtin-skills" in result.stdout

    def test_search_no_results(self, tmp_registry: Path) -> None:
        """Searching with no matches prints message."""
        result = _run_cli("search", "nonexistent_xyz", registry_dir=str(tmp_registry))
        assert result.returncode == 0
        assert "No packages match" in result.stdout


class TestPkgValidate:
    """apyrobo pkg validate"""

    def test_validate_valid_package(self) -> None:
        """Validating a correct package prints 'valid'."""
        result = _run_cli("validate", str(SKILLS_DIR / "builtin-skills"))
        assert result.returncode == 0
        assert "valid" in result.stdout.lower()

    def test_validate_bad_directory(self) -> None:
        """Validating a nonexistent directory fails."""
        result = _run_cli("validate", "/nonexistent/path")
        assert result.returncode != 0


class TestPkgRemove:
    """apyrobo pkg remove"""

    def test_remove_installed_package(self, tmp_registry: Path) -> None:
        """Removing an installed package prints confirmation."""
        _run_cli(
            "install", str(SKILLS_DIR / "builtin-skills"),
            registry_dir=str(tmp_registry),
        )
        result = _run_cli("remove", "builtin-skills", registry_dir=str(tmp_registry))
        assert result.returncode == 0
        assert "Removed: builtin-skills" in result.stdout

    def test_remove_nonexistent(self, tmp_registry: Path) -> None:
        """Removing a package that isn't installed fails."""
        result = _run_cli("remove", "nonexistent-pkg", registry_dir=str(tmp_registry))
        assert result.returncode != 0
        assert "not installed" in result.stdout

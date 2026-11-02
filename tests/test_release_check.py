"""Unit tests for scripts/release_check.py (no network, no build)."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "release_check", ROOT / "scripts" / "release_check.py"
)
rc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["release_check"] = rc
_spec.loader.exec_module(rc)

NAME = "io.github.example/demo"


def _project(tmp_path: Path, *, version="1.2.0", server_version="1.2.0",
             pkg_version="1.2.0",
             changelog="## [1.2.0] - 2026-11-01\n\n### Added\n\n- A thing.\n",
             readme=f"# Demo\n<!-- mcp-name: {NAME} -->\n") -> Path:
    (tmp_path / "pyproject.toml").write_text(
        f'[build-system]\nrequires = []\n\n[project]\nname = "demo"\n'
        f'version = "{version}"\n\n[tool.x]\nversion = "9.9.9"\n',
        encoding="utf-8",
    )
    (tmp_path / "server.json").write_text(json.dumps({
        "name": NAME, "version": server_version,
        "packages": [{"registryType": "pypi", "identifier": "demo",
                      "version": pkg_version}],
    }), encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n" + changelog, encoding="utf-8")
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    return tmp_path


def test_pyproject_version_reads_project_table_only(tmp_path):
    assert rc.pyproject_version(_project(tmp_path)) == "1.2.0"


def test_pyproject_version_missing(tmp_path):
    assert rc.pyproject_version(tmp_path) is None


def test_versions_match(tmp_path):
    check = rc.check_versions(_project(tmp_path))
    assert check.ok, check.detail


@pytest.mark.parametrize("kw", [{"server_version": "1.1.0"}, {"pkg_version": "1.1.0"}])
def test_versions_mismatch_is_reported(tmp_path, kw):
    check = rc.check_versions(_project(tmp_path, **kw))
    assert not check.ok
    assert "1.1.0" in check.detail


def test_versions_bad_server_json(tmp_path):
    root = _project(tmp_path)
    (root / "server.json").write_text("{not json", encoding="utf-8")
    assert not rc.check_versions(root).ok


def test_changelog_section_present(tmp_path):
    assert rc.check_changelog(_project(tmp_path), "1.2.0").ok


def test_changelog_only_unreleased_fails(tmp_path):
    root = _project(tmp_path, changelog="")
    check = rc.check_changelog(root, "1.2.0")
    assert not check.ok
    assert "[1.2.0]" in check.detail


def test_changelog_version_is_not_a_prefix_match(tmp_path):
    root = _project(tmp_path, changelog="## [1.2.0rc1] - 2026-10-01\n")
    # "1.2.0" must not match "1.2.0rc1" via the regex escape + bracket
    assert not rc.check_changelog(root, "1.2.0").ok


def test_marker_from_server_json(tmp_path):
    root = _project(tmp_path)
    assert rc.mcp_marker(root) == f"mcp-name: {NAME}"
    assert rc.check_marker_in((root / "README.md").read_text(), rc.mcp_marker(root),
                              "README.md").ok


def test_marker_missing_from_readme(tmp_path):
    root = _project(tmp_path, readme="# Demo\n")
    assert not rc.check_marker_in("# Demo\n", rc.mcp_marker(root), "README.md").ok


def test_wheel_metadata(tmp_path):
    wheel = tmp_path / "demo-1.2.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr("demo/__init__.py", "")
        zf.writestr("demo-1.2.0.dist-info/METADATA",
                    f"Name: demo\n\n<!-- mcp-name: {NAME} -->\n")
    assert f"mcp-name: {NAME}" in rc.wheel_metadata(wheel)


def test_wheel_without_metadata(tmp_path):
    wheel = tmp_path / "bad.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr("demo/__init__.py", "")
    with pytest.raises(FileNotFoundError):
        rc.wheel_metadata(wheel)


def test_static_run_on_tmp_project(tmp_path):
    checks = rc.run(_project(tmp_path), static=True, keep=False)
    assert [c.ok for c in checks] == [True, True, True]


def test_main_exit_code(tmp_path, capsys):
    assert rc.main(["--root", str(_project(tmp_path)), "--static"]) == 0
    assert rc.main(["--root", str(_project(tmp_path, changelog="")), "--static"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_drive_mcp_against_this_checkout():
    """The stdio smoke step passes against the in-repo server."""
    cmd = [sys.executable, "-m", "apyrobo", "mcp", "--robot", "mock://turtlebot4",
           "--max-speed", str(rc.CLAMP_LIMIT)]
    checks = rc.drive_mcp(cmd)
    assert [c.ok for c in checks] == [True, True, True], [c.line() for c in checks]


def test_drive_mcp_reports_missing_clamp():
    """Without --max-speed the move is not clamped to the limit, and that fails."""
    cmd = [sys.executable, "-m", "apyrobo", "mcp", "--robot", "mock://turtlebot4",
           "--max-speed", "10"]
    checks = rc.drive_mcp(cmd)
    assert checks[0].ok and checks[1].ok
    assert not checks[2].ok


def test_drive_mcp_checks_reported_version():
    cmd = [sys.executable, "-m", "apyrobo", "mcp", "--robot", "mock://turtlebot4",
           "--max-speed", str(rc.CLAMP_LIMIT)]
    good = rc.drive_mcp(cmd, expected_version=rc.pyproject_version(ROOT))
    assert good[0].ok, good[0].line()
    bad = rc.drive_mcp(cmd, expected_version="0.0.0")
    assert not bad[0].ok
    assert "expected 0.0.0" in bad[0].detail


def test_drive_mcp_bad_command():
    checks = rc.drive_mcp(["/nonexistent/apyrobo", "mcp"])
    assert not checks[0].ok


# -- changelog section contents ----------------------------------------


@pytest.mark.parametrize("section", [
    "## [1.2.0] - 2026-11-01\n",
    "## [1.2.0] - 2026-11-01\n\n### Added\n\n---\n\n## [1.1.0] - 2026-01-01\n\n- Old.\n",
])
def test_changelog_fails_on_an_empty_section(tmp_path, section):
    root = _project(tmp_path, changelog=section)
    check = rc.check_changelog(root, "1.2.0")
    assert not check.ok
    assert "empty" in check.detail


def test_changelog_fails_on_placeholder_comments(tmp_path):
    root = _project(tmp_path, changelog=(
        "## [1.2.0] - 2026-11-01\n\n### Added\n- <!-- describe new features -->\n"
    ))
    check = rc.check_changelog(root, "1.2.0")
    assert not check.ok
    assert "placeholder" in check.detail


def test_changelog_entries_of_another_version_do_not_count(tmp_path):
    root = _project(tmp_path, changelog=(
        "## [1.3.0] - 2026-11-02\n\n- New.\n\n"
        "## [1.2.0] - 2026-11-01\n\n## [1.1.0] - 2026-01-01\n\n- Old.\n"
    ))
    assert not rc.check_changelog(root, "1.2.0").ok
    assert rc.check_changelog(root, "1.3.0").ok


# -- scripts/bump_version.sh -------------------------------------------

CHANGELOG = """# Changelog

Intro.

---

## [Unreleased]

### Added

- A thing users want.

---

## [1.2.0] - 2026-01-01

- Older.
"""


@pytest.fixture
def bump_project(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    root = _project(tmp_path)
    (root / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    (root / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "bump_version.sh", root / "scripts")
    (root / "apyrobo").mkdir()
    (root / "apyrobo" / "__version__.py").write_text(
        '__version__ = "1.2.0"\n', encoding="utf-8")
    return root


def _bump(root: Path, version: str = "1.3.0") -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(root / "scripts" / "bump_version.sh"), version],
        capture_output=True, text=True,
    )


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): p.read_text(encoding="utf-8")
        for p in root.rglob("*") if p.is_file()
    }


def test_bump_passes_the_static_release_check(bump_project):
    result = _bump(bump_project)
    assert result.returncode == 0, result.stderr
    assert rc.check_versions(bump_project).ok
    assert rc.check_changelog(bump_project, "1.3.0").ok
    assert '"1.3.0"' in (bump_project / "apyrobo" / "__version__.py").read_text()


def test_bump_moves_the_entries_and_leaves_a_fresh_unreleased(bump_project):
    _bump(bump_project)
    text = (bump_project / "CHANGELOG.md").read_text(encoding="utf-8")
    assert text.index("## [Unreleased]") < text.index("## [1.3.0] - ")
    assert text.index("## [1.3.0] - ") < text.index("A thing users want")
    assert text.index("A thing users want") < text.index("## [1.2.0]")
    assert rc.changelog_section(text, "Unreleased").strip() == "---"
    assert "Intro." in text and "Older." in text


def test_bump_keeps_the_rest_of_server_json(bump_project):
    path = bump_project / "server.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["packages"].append({"registryType": "npm", "version": "0.1.0"})
    data["description"] = "Sûr"
    path.write_text(json.dumps(data), encoding="utf-8")
    _bump(bump_project)
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["description"] == "Sûr"
    assert after["version"] == after["packages"][0]["version"] == "1.3.0"
    assert after["packages"][1]["version"] == "0.1.0"


def test_bump_refuses_an_empty_unreleased_and_changes_nothing(bump_project):
    path = bump_project / "CHANGELOG.md"
    path.write_text(CHANGELOG.replace("- A thing users want.\n", ""), encoding="utf-8")
    before = _snapshot(bump_project)
    result = _bump(bump_project)
    assert result.returncode != 0
    assert "empty" in result.stderr
    assert _snapshot(bump_project) == before


def test_bump_refuses_a_version_that_already_has_a_section(bump_project):
    before = _snapshot(bump_project)
    result = _bump(bump_project, "1.2.0")
    assert result.returncode != 0
    assert "already has" in result.stderr
    assert _snapshot(bump_project) == before

"""Unit tests for scripts/release_check.py (no network, no build)."""

from __future__ import annotations

import importlib.util
import json
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
             pkg_version="1.2.0", changelog="## [1.2.0] - 2026-11-01\n",
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

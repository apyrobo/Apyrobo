"""apyrobo config --file / APYROBO_CONFIG: a file the user named that is
missing or unreadable is one error line and exit 1, never defaults shown as
if they were the user's settings, and never a traceback."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from apyrobo.config import ApyroboConfig


def _run(args: list[str], cwd: Path, env_extra: dict[str, str] | None = None):
    env = {k: v for k, v in os.environ.items() if k != "APYROBO_CONFIG"}
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", "apyrobo", "config", *args],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=60,
    )


def _assert_one_line_error(proc, *needles: str) -> None:
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr
    assert proc.stdout == ""
    err = proc.stderr.strip()
    assert err.startswith("Error:"), err
    assert len(err.splitlines()) == 1, err
    for needle in needles:
        assert needle in err, err


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    proc = _run(["--file", "nope.yaml"], tmp_path)
    _assert_one_line_error(proc, "nope.yaml", "not found")


def test_directory_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "conf").mkdir()
    proc = _run(["--file", "conf"], tmp_path)
    _assert_one_line_error(proc, "conf", "is a directory")


def test_yaml_syntax_error_names_the_line(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("safety:\n  max_speed: [0.5\n")
    proc = _run(["--file", "bad.yaml"], tmp_path)
    _assert_one_line_error(proc, "bad.yaml", "line")


def test_toml_syntax_error(tmp_path: Path) -> None:
    pytest.importorskip("tomllib" if sys.version_info >= (3, 11) else "tomli")
    (tmp_path / "bad.toml").write_text("a = \n")
    proc = _run(["--file", "bad.toml"], tmp_path)
    _assert_one_line_error(proc, "bad.toml", "Invalid value")


def test_top_level_list_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "list.yaml").write_text("- 1\n- 2\n")
    proc = _run(["--file", "list.yaml"], tmp_path)
    _assert_one_line_error(proc, "list.yaml", "mapping", "list")


def test_missing_env_config_is_an_error(tmp_path: Path) -> None:
    proc = _run([], tmp_path, {"APYROBO_CONFIG": "nope.yaml"})
    _assert_one_line_error(proc, "nope.yaml", "APYROBO_CONFIG", "not found")


def test_valid_file_still_shows_its_settings(tmp_path: Path) -> None:
    (tmp_path / "ok.yaml").write_text("safety:\n  max_speed: 0.4\n")
    proc = _run(["--file", "ok.yaml"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "max_speed: 0.4" in proc.stdout


def test_valid_env_config_still_loads(tmp_path: Path) -> None:
    (tmp_path / "ok.yaml").write_text("robot:\n  uri: mock://envbot\n")
    proc = _run([], tmp_path, {"APYROBO_CONFIG": "ok.yaml"})
    assert proc.returncode == 0, proc.stderr
    assert "mock://envbot" in proc.stdout


def test_library_rejects_non_mapping_with_value_error(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n")
    with pytest.raises(ValueError, match="mapping"):
        ApyroboConfig.from_file(path)
    with pytest.raises(ValueError, match="mapping"):
        ApyroboConfig.from_yaml_file(path)


def test_library_missing_file_still_returns_defaults(tmp_path: Path) -> None:
    cfg = ApyroboConfig.from_file(tmp_path / "absent.yaml")
    assert cfg.robot_uri == ApyroboConfig().robot_uri

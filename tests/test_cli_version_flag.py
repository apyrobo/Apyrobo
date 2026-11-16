"""`apyrobo --version` prints the installed version (bug reports ask for it)."""
from __future__ import annotations

import subprocess
import sys

import pytest

import apyrobo
from apyrobo import cli


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_prints_version_and_exits_zero(flag, monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["apyrobo", flag])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"apyrobo {apyrobo.__version__}"


def test_lowercase_v_still_means_verbose(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["apyrobo", "-v"])
    try:
        cli.main()
    except SystemExit as exc:
        assert exc.code in (0, None)
    assert "apyrobo " + apyrobo.__version__ not in capsys.readouterr().out


def test_version_flag_from_a_fresh_process() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "--version"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"apyrobo {apyrobo.__version__}"

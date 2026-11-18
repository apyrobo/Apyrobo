"""apyrobo mcp keepout: a bad --out says which flag and folder, and every flag has help."""

from __future__ import annotations

import subprocess
import sys

import pytest


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "apyrobo", "mcp", "keepout", *args],
        capture_output=True, text=True, timeout=60,
    )


def test_out_that_is_a_file_names_the_flag_and_the_folder(tmp_path):
    blocked = tmp_path / "keepout_mask.pgm"
    blocked.write_text("")
    result = run_cli("--no-go", "0,0,1,1", "--out", str(blocked))
    assert result.returncode == 1
    line = result.stderr.strip()
    assert line.startswith(f"Error: cannot write the mask to the --out folder {blocked} (")
    assert "give a folder you can write to" in line
    assert "[Errno" not in line
    assert "\n" not in line
    assert result.stdout == ""


def test_out_under_a_file_names_the_flag(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("")
    result = run_cli("--no-go", "0,0,1,1", "--out", str(blocked / "masks"))
    assert result.returncode == 1
    assert "the --out folder" in result.stderr
    assert "[Errno" not in result.stderr


def test_missing_out_folder_is_still_created(tmp_path):
    out = tmp_path / "a" / "b"
    result = run_cli("--no-go", "0,0,1,1", "--out", str(out))
    assert result.returncode == 0, result.stderr
    assert (out / "keepout_mask.pgm").is_file()
    assert (out / "keepout_mask.yaml").is_file()


def test_zone_errors_keep_their_own_message(tmp_path):
    result = run_cli("--no-go", "0,x,1,1", "--out", str(tmp_path))
    assert result.returncode == 1
    assert "--out" not in result.stderr


@pytest.mark.parametrize("flag", ["--policy-file", "--no-go", "--resolution", "--out"])
def test_every_keepout_flag_has_help(flag):
    result = run_cli("--help")
    assert result.returncode == 0
    lines = result.stdout.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith(flag))
    # argparse puts the help on the flag's line, or on the next indented line.
    own = lines[start].strip().split("  ", 1)
    following = lines[start + 1].strip() if start + 1 < len(lines) else ""
    has_help = len(own) == 2 and own[1].strip() or (
        following and not following.startswith("-")
    )
    assert has_help, f"{flag} has no help text:\n{result.stdout}"

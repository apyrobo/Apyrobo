"""apyrobo mcp config --policy-file names why a path cannot be read."""
import os
import subprocess
import sys

import pytest

from apyrobo import mcp_clients


def test_folder_is_not_reported_as_missing(tmp_path):
    with pytest.raises(ValueError) as exc:
        mcp_clients.server_args(policy_file=str(tmp_path))
    message = str(exc.value)
    assert "not found" not in message
    assert message.startswith(f"cannot read policy file {tmp_path.resolve()}")


def test_missing_file_still_says_not_found(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(ValueError, match="policy file not found"):
        mcp_clients.server_args(policy_file=str(missing))


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0,
                    reason="file permissions do not stop root or Windows here")
def test_unreadable_file_names_the_cause(tmp_path):
    policy = tmp_path / "lab.yaml"
    policy.write_text("max_speed: 0.5\n")
    policy.chmod(0)
    try:
        with pytest.raises(ValueError, match="cannot read policy file"):
            mcp_clients.server_args(policy_file=str(policy))
    finally:
        policy.chmod(0o644)


def test_config_command_prints_one_line_for_a_folder(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "mcp", "config", "cursor",
         "--policy-file", str(tmp_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert proc.stderr.startswith("Error: cannot read policy file")
    assert "not found" not in proc.stderr

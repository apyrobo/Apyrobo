"""A policy file whose base is not a policy is refused everywhere it is read.

apyrobo mcp refused to start with ``base: bogus``, but apyrobo mcp config
printed a config for it (exit 0), so the client got a server that never
starts; apyrobo mcp keepout's error did not name the file; and ``base: ""``
quietly meant 'default'.
"""
import subprocess
import sys

import pytest

from apyrobo import mcp_clients
from apyrobo.mcp_server import build_server, load_policy_file


@pytest.fixture
def bad_base(tmp_path):
    p = tmp_path / "lab.yaml"
    p.write_text("base: bogus\nmax_speed: 0.3\n")
    return p


@pytest.mark.parametrize("value", ["bogus", "", "Strict", "strict "])
def test_load_refuses_a_base_that_is_not_a_policy(tmp_path, value):
    p = tmp_path / "lab.yaml"
    p.write_text(f"base: {value!r}\n")
    with pytest.raises(ValueError) as exc:
        load_policy_file(p)
    message = str(exc.value)
    assert message.startswith(f"{p}: unknown policy {value!r} in base")
    assert "choose from ['default', 'strict']" in message


@pytest.mark.parametrize("value", ["default", "strict"])
def test_known_bases_still_load(tmp_path, value):
    p = tmp_path / "lab.yaml"
    p.write_text(f"base: {value}\n")
    assert load_policy_file(p)["base"] == value


def test_server_still_refuses(bad_base):
    with pytest.raises(ValueError, match="unknown policy 'bogus'"):
        build_server(policy_file=bad_base)


def test_server_refuses_even_when_policy_flag_is_given(bad_base):
    # --policy strict used to hide a broken file; the file is still wrong.
    with pytest.raises(ValueError, match="unknown policy 'bogus' in base"):
        build_server(policy="strict", policy_file=bad_base)


def test_config_args_refuse(bad_base):
    with pytest.raises(ValueError, match="unknown policy 'bogus' in base"):
        mcp_clients.server_args(policy_file=str(bad_base))


def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "mcp", *args],
        capture_output=True, text=True, timeout=120,
    )


@pytest.mark.parametrize("client", ["claude-desktop", "cursor", "vscode", "codex"])
def test_config_command_prints_no_config(bad_base, client):
    proc = _cli("config", client, "--policy-file", str(bad_base))
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert "unknown policy 'bogus' in base" in proc.stderr
    assert "lab.yaml" in proc.stderr


def test_keepout_names_the_file(bad_base, tmp_path):
    proc = _cli("keepout", "--policy-file", str(bad_base), "--out", str(tmp_path / "m.pgm"))
    assert proc.returncode == 1
    assert "lab.yaml" in proc.stderr
    assert "unknown policy 'bogus' in base" in proc.stderr
    assert not (tmp_path / "m.pgm").exists()


def test_selftest_refuses(bad_base):
    proc = _cli("selftest", "--policy-file", str(bad_base))
    assert proc.returncode == 1
    assert "unknown policy 'bogus' in base" in proc.stderr

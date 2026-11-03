"""docs/first-real-robot.md: its links, policy and commands match the code."""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

import pytest

from apyrobo.cli import main
from apyrobo.mcp_server import build_server

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "first-real-robot.md"
TEXT = DOC.read_text(encoding="utf-8")


def blocks(language):
    return re.findall(rf"```{language}\n(.*?)```", TEXT, flags=re.S)


@pytest.fixture
def policy(tmp_path, monkeypatch):
    path = tmp_path / "first-robot.yaml"
    path.write_text(blocks("yaml")[0], encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return path


def call(server, name, arguments):
    return server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    })["result"]


def test_relative_links_point_at_existing_files():
    for target in re.findall(r"\]\((?!https?:)([^)#]+)", TEXT):
        assert (DOC.parent / target).exists(), target


def test_sample_policy_clamps_and_refuses(policy):
    server = build_server(policy_file=policy)
    moved = call(server, "move_to", {"x": 1, "y": 0, "speed": 2.0})
    assert not moved.get("isError")
    report = json.loads(moved["content"][0]["text"])
    assert report["safety_interventions"][0]["enforced"] == 0.2
    assert call(server, "move_to", {"x": 3, "y": 3})["isError"]
    assert call(server, "move_to", {"x": 6, "y": 0})["isError"]


def commands():
    joined = "\n".join(blocks("bash")).replace("\\\n", " ")
    return [line for line in joined.splitlines() if line.startswith("apyrobo ")]


def run_cli(command, monkeypatch):
    monkeypatch.setattr(sys, "argv", shlex.split(command))
    try:
        main()
    except SystemExit as stopped:
        return stopped.code
    return 0


def test_selftest_command_passes_on_the_sample_policy(policy, monkeypatch, capsys):
    selftest = next(c for c in commands() if " selftest " in c)
    assert run_cli(selftest, monkeypatch) in (0, None)
    assert "checks passed" in capsys.readouterr().out


def test_ssh_config_command_keeps_the_policy_on_the_robot(policy, monkeypatch, capsys):
    config = next(c for c in commands() if " config " in c)
    assert run_cli(config, monkeypatch) in (0, None)
    out = capsys.readouterr().out
    remote = json.loads(out[: out.rindex("}") + 1])["mcpServers"]["apyrobo"]["args"][-1]
    assert "--policy-file /home/ubuntu/first-robot.yaml" in remote
    assert "--confirm-moves" in remote


def test_doc_says_it_is_not_a_certified_system_and_keeps_the_e_stop():
    assert "not a certified safety system" in TEXT
    assert "does not replace a hardware e-stop" in TEXT

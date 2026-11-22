"""Keep docs/mcp.md in step with what apyrobo/mcp_server.py implements."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

from apyrobo import mcp_server

DOC = Path(__file__).resolve().parent.parent / "docs" / "mcp.md"


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def test_every_supported_protocol_version_is_documented():
    text = _doc()
    versions = mcp_server.SUPPORTED_PROTOCOL_VERSIONS + mcp_server.MODERN_PROTOCOL_VERSIONS
    missing = [v for v in versions if v not in text]
    assert not missing, f"docs/mcp.md does not mention protocol versions {missing}"


def test_no_unsupported_protocol_version_is_documented():
    text = _doc()
    supported = set(mcp_server.SUPPORTED_PROTOCOL_VERSIONS + mcp_server.MODERN_PROTOCOL_VERSIONS)
    documented = set(re.findall(r"\b20\d\d-\d\d-\d\d\b", text))
    extra = documented - supported
    assert not extra, f"docs/mcp.md names protocol versions the server does not support: {extra}"


def test_documented_methods_are_handled():
    source = Path(mcp_server.__file__).read_text(encoding="utf-8")
    for method in ("initialize", "ping", "tools/list", "tools/call", "server/discover"):
        assert f"`{method}`" in _doc(), f"docs/mcp.md does not list {method}"
        assert f'"{method}"' in source, f"mcp_server.py does not handle {method}"


def test_relative_links_point_at_existing_files():
    broken = []
    for target in re.findall(r"\]\((?!https?://|#)([^)#]+)\)", _doc()):
        if not (DOC.parent / target).exists():
            broken.append(target)
    assert not broken, f"docs/mcp.md links to missing files: {broken}"


def test_the_uvx_ros2_pin_is_documented():
    """51b: the isolated-environment trap and the verified pin, not just sourcing."""
    text = _doc()
    for needed in (
        "--no-managed-python --python 3.12",
        "--with 'numpy<2' --with transforms3d",
        "/usr/lib/python3/dist-packages",
        "ros-$ROS_DISTRO-nav2-msgs",
    ):
        assert needed in text, f"docs/mcp.md no longer documents {needed!r}"


def test_the_tools_table_is_the_whole_tool_list():
    server = mcp_server.build_server(robot_uri="mock://docs-tools")
    served = {tool["name"] for tool in server.toolbox.list_tools()}
    section = _doc().split("## Tools")[1].split("\n## ")[0]
    documented = set(re.findall(r"^\| `(\w+)", section, flags=re.MULTILINE))
    assert documented == served
    assert not any("topic" in name or "service" in name for name in served)


def test_the_enforcer_reads_no_live_sensors_as_the_docs_say():
    server = mcp_server.build_server(robot_uri="mock://docs-sensors")
    assert server.toolbox.enforcer._world_state is None
    text = " ".join(_doc().split())
    assert "It checks policy, not live sensors." in text
    assert "human-proximity check does not run" in text


def test_docs_say_how_the_speed_limit_reaches_a_nav2_drive():
    text = _doc()
    assert "### Speed on `ros2://`" in text
    assert "`speed_limit` topic" in text
    # The fallback when nothing takes the limit stays documented, and so do
    # the limits of what CI checks.
    assert '"speed_limit_applied": false' in text
    assert "Humble with DWB" in text
    assert "Clamps and no-go refusals happen" not in text
    llms = (DOC.parent.parent / "llms.txt").read_text(encoding="utf-8")
    assert "`speed_limit` topic" in llms
    assert "not applied" in llms
    robot = (DOC.parent / "first-real-robot.md").read_text(encoding="utf-8")
    assert "mcp.md#speed-on-ros2" in robot


# --- Client snippets: a pasted block must match what `apyrobo mcp config` prints.

FENCE = re.compile(r"```(\w*)\n(.*?)```", re.S)


def _fences(lang: str) -> list[str]:
    return [body for tag, body in FENCE.findall(_doc()) if tag == lang]


def _config_args(argv: list[str], monkeypatch):
    """Parse ``apyrobo mcp config ...`` with the real CLI parser."""
    import sys

    import apyrobo.cli as cli

    seen = {}
    monkeypatch.setattr(cli, "cmd_mcp", lambda args: seen.update(args=args))
    monkeypatch.setattr(sys, "argv", ["apyrobo", *argv])
    cli.main()
    return seen["args"]


def _printed(argv: list[str], monkeypatch) -> tuple[str, str]:
    """What ``apyrobo mcp config`` prints, with nothing on PATH (bare ``uvx``)."""
    import io

    from apyrobo import mcp_clients

    out, err = io.StringIO(), io.StringIO()
    code = mcp_clients.main(_config_args(argv, monkeypatch), stdout=out, stderr=err,
                            which=lambda name: None)
    assert code == 0, err.getvalue()
    return out.getvalue(), err.getvalue()


def _documented_config_flags() -> list[str]:
    """Flags of the `apyrobo mcp config claude-desktop ...` example in the doc."""
    for body in _fences("bash"):
        for line in body.splitlines():
            if line.startswith("apyrobo mcp config claude-desktop"):
                return shlex.split(line)[4:]
    raise AssertionError("docs/mcp.md has no `apyrobo mcp config claude-desktop` example")


def _without_uvx_source(args: list[str]) -> list[str]:
    """Drop ``--from <git source>``: both uvx forms are valid in the docs.

    The tool prints the GitHub form before ``apyrobo mcp`` is on PyPI and the
    plain form after, so the docs may show either; the rest must match.
    """
    from apyrobo.mcp_clients import GIT_SOURCE

    if args[:2] == ["--from", GIT_SOURCE]:
        return args[2:]
    return args


def test_every_json_block_parses():
    blocks = _fences("json")
    assert blocks
    for body in blocks:
        try:
            json.loads(body)
        except json.JSONDecodeError:
            # JSON Lines (the audit log example): every line is one object.
            for line in body.strip().splitlines():
                json.loads(line)


def _documented_server() -> dict:
    servers = [json.loads(b)["mcpServers"] for b in _fences("json")
               if '"mcpServers"' in b and '"type"' not in b]
    assert len(servers) == 1, "expected one Claude Desktop mcpServers block in docs/mcp.md"
    return servers[0]["apyrobo"]


def test_the_desktop_block_matches_mcp_config(monkeypatch):
    flags = _documented_config_flags()
    out, _ = _printed(["mcp", "config", "claude-desktop", "--launcher", "uvx", *flags],
                      monkeypatch)
    printed = json.loads(out)["mcpServers"]["apyrobo"]
    documented = _documented_server()
    assert documented["command"] == printed["command"] == "uvx"
    assert _without_uvx_source(documented["args"]) == _without_uvx_source(printed["args"])
    assert _without_uvx_source(documented["args"])[:2] == ["apyrobo", "mcp"]


def test_cursor_takes_the_same_block(monkeypatch):
    flags = _documented_config_flags()
    desktop, _ = _printed(["mcp", "config", "claude-desktop", "--launcher", "uvx", *flags],
                          monkeypatch)
    cursor, _ = _printed(["mcp", "config", "cursor", "--launcher", "uvx", *flags], monkeypatch)
    assert json.loads(cursor) == json.loads(desktop)
    assert "Add the same `mcpServers` block to `~/.cursor/mcp.json`" in _doc()


def test_the_vscode_block_matches_mcp_config(monkeypatch):
    blocks = [json.loads(b)["mcpServers"]["apyrobo"] for b in _fences("json")
              if '"mcpServers"' in b and '"type"' in b]
    assert len(blocks) == 1, "expected one VS Code mcpServers block in docs/mcp.md"
    documented = blocks[0]
    out, _ = _printed(["mcp", "config", "vscode", "--launcher", "uvx",
                       *_documented_config_flags()], monkeypatch)
    printed = json.loads(out)["mcpServers"]["apyrobo"]
    assert documented["type"] == printed["type"] == "stdio"
    assert documented["tools"] == printed["tools"] == ["*"]
    assert documented["command"] == printed["command"] == "uvx"
    assert _without_uvx_source(documented["args"]) == _without_uvx_source(printed["args"])
    assert "~/.copilot/mcp-config.json" in _doc()


def test_the_pip_variant_matches_launcher_apyrobo(monkeypatch):
    match = re.search(r'use `"command": "(\w+)"` and\s+`"args": (\[.*?\])`', _doc(), re.S)
    assert match, "the pip variant of the Claude Desktop block is missing"
    command, args = match.group(1), json.loads(match.group(2))
    out, _ = _printed(
        ["mcp", "config", "claude-desktop", "--launcher", "apyrobo", *_documented_config_flags()],
        monkeypatch,
    )
    printed = json.loads(out)["mcpServers"]["apyrobo"]
    assert (command, args) == (printed["command"], printed["args"])


def test_the_claude_code_command_matches_mcp_config(monkeypatch):
    lines = [line for body in _fences("bash") for line in body.splitlines()
             if line.startswith("claude mcp add apyrobo ")]
    assert lines, "the Claude Code `claude mcp add` example is missing"
    documented = shlex.split(lines[0])
    out, _ = _printed(
        ["mcp", "config", "claude-code", "--launcher", "uvx", *_documented_config_flags()],
        monkeypatch,
    )
    printed = shlex.split(out.strip())
    split = printed.index("--")
    assert documented[:split + 2] == printed[:split + 2]  # claude mcp add apyrobo -- uvx
    assert (_without_uvx_source(documented[split + 2:])
            == _without_uvx_source(printed[split + 2:]))


def test_the_claude_code_pip_variant_matches_launcher_apyrobo(monkeypatch):
    match = re.search(r"If you installed with `pip`, use `-- ([^`]+)`", _doc())
    assert match, "the pip variant of the Claude Code command is missing"
    out, _ = _printed(
        ["mcp", "config", "claude-code", "--launcher", "apyrobo", *_documented_config_flags()],
        monkeypatch,
    )
    printed = shlex.split(out.strip())
    assert shlex.split(match.group(1)) == printed[printed.index("--") + 1:]


def test_the_codex_command_matches_mcp_config(monkeypatch):
    lines = [line for body in _fences("bash") for line in body.splitlines()
             if line.startswith("codex mcp add apyrobo ")]
    assert lines, "the Codex `codex mcp add` example is missing"
    documented = shlex.split(lines[0])
    out, _ = _printed(
        ["mcp", "config", "codex", "--launcher", "uvx", *_documented_config_flags()],
        monkeypatch,
    )
    printed = shlex.split(out.strip())
    split = printed.index("--")
    assert documented[:split + 2] == printed[:split + 2]  # codex mcp add apyrobo -- uvx
    assert (_without_uvx_source(documented[split + 2:])
            == _without_uvx_source(printed[split + 2:]))


def test_the_codex_pip_variant_matches_launcher_apyrobo(monkeypatch):
    match = re.search(r"If you installed with `pip`, end the line with `-- ([^`]+)`", _doc())
    assert match, "the pip variant of the Codex command is missing"
    out, _ = _printed(
        ["mcp", "config", "codex", "--launcher", "apyrobo", *_documented_config_flags()],
        monkeypatch,
    )
    printed = shlex.split(out.strip())
    assert shlex.split(match.group(1)) == printed[printed.index("--") + 1:]


def _gemini_doc_lines() -> list[list[str]]:
    return [shlex.split(line) for body in _fences("bash") for line in body.splitlines()
            if line.startswith("gemini mcp add ")]


def test_the_gemini_command_matches_mcp_config(monkeypatch):
    lines = _gemini_doc_lines()
    assert lines, "the Gemini CLI `gemini mcp add` example is missing"
    documented = lines[0]
    _, err = _printed(
        ["mcp", "config", "gemini", "--launcher", "uvx", *_documented_config_flags()],
        monkeypatch,
    )
    line = next(x for x in err.splitlines() if x.startswith("Or run instead: "))
    printed = shlex.split(line[len("Or run instead: "):])
    split = printed.index("--")
    assert documented[:split + 1] == printed[:split + 1]  # gemini mcp add -s user apyrobo uvx --
    assert (_without_uvx_source(documented[split + 1:])
            == _without_uvx_source(printed[split + 1:]))
    assert "--trust" not in documented


def test_the_gemini_pip_variant_matches_launcher_apyrobo(monkeypatch):
    match = re.search(r"If you installed with\s+`pip`, use `(gemini mcp add [^`]+)`", _doc())
    assert match, "the pip variant of the Gemini CLI command is missing"
    _, err = _printed(
        ["mcp", "config", "gemini", "--launcher", "apyrobo", *_documented_config_flags()],
        monkeypatch,
    )
    line = next(x for x in err.splitlines() if x.startswith("Or run instead: "))
    assert shlex.split(match.group(1)) == shlex.split(line[len("Or run instead: "):])

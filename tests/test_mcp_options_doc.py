"""docs/mcp.md 'Options' lists the flags ``apyrobo mcp`` takes, with their defaults.

The table is what a reader copies flags from, so it is checked against the
parser: every flag the server takes is in the table and every row is a flag the
server takes, the defaults it states match ``--help``, and the speed limits it
gives for ``default`` and ``strict`` are the ones the server enforces.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from apyrobo import cli
from apyrobo.mcp_server import build_server

DOC = Path(__file__).resolve().parent.parent / "docs" / "mcp.md"


def _rows() -> dict[str, tuple[str, str]]:
    text = DOC.read_text(encoding="utf-8")
    start = text.index("## Options")
    end = text.index("\n#", start + 1)
    rows = {}
    for line in text[start:end].splitlines():
        match = re.match(r"\| `(--[a-z-]+)[^`]*` \| (.*?) \| (.*) \|$", line)
        if match:
            rows[match.group(1)] = (match.group(2), match.group(3))
    return rows


@pytest.fixture(scope="module")
def help_text() -> str:
    import contextlib
    import io
    import sys

    out = io.StringIO()
    argv = sys.argv
    sys.argv = ["apyrobo", "mcp", "--help"]
    try:
        with contextlib.redirect_stdout(out), pytest.raises(SystemExit):
            cli.main()
    finally:
        sys.argv = argv
    return out.getvalue()


def _help_flags(help_text: str) -> set[str]:
    options = help_text[help_text.index("options:"):]
    return set(re.findall(r"^\s+(--[a-z-]+)", options, re.M)) - {"--help"}


def test_the_table_lists_every_flag_the_server_takes(help_text):
    documented = set(_rows())
    taken = _help_flags(help_text)
    assert taken, "could not read the flags from apyrobo mcp --help"
    assert documented == taken, (
        f"missing from docs/mcp.md Options: {sorted(taken - documented)}; "
        f"listed but not taken: {sorted(documented - taken)}"
    )


@pytest.mark.parametrize(("flag", "default"), [("--robot", "mock://turtlebot4"),
                                               ("--provider", "rule")])
def test_stated_defaults_match_the_help(help_text, flag, default):
    assert _rows()[flag][0] == f"`{default}`"
    assert f"(default: {default}" in " ".join(help_text.split())


def test_policy_default_says_the_file_base_comes_first(help_text):
    """With no --policy, a policy file's base is used, as --help says."""
    cell = _rows()["--policy"][0]
    assert "base" in cell and "`default`" in cell
    assert "the policy file's base, else default" in " ".join(help_text.split())


def test_policy_file_base_is_used_and_the_flag_wins(tmp_path):
    strict = build_server(policy="strict").toolbox.enforcer.policy.max_speed
    default = build_server().toolbox.enforcer.policy.max_speed
    p = tmp_path / "p.yaml"
    p.write_text("base: strict\n")
    assert build_server(policy_file=p).toolbox.enforcer.policy.max_speed == strict
    flag_wins = build_server(policy="default", policy_file=p)
    assert flag_wins.toolbox.enforcer.policy.max_speed == default


def test_base_policy_speeds_are_the_ones_stated():
    meaning = _rows()["--policy"][1]
    for name in ("default", "strict"):
        limit = build_server(policy=name).toolbox.enforcer.policy.max_speed
        assert f"`{name}` ({limit:g} m/s)" in meaning, (
            f"docs/mcp.md Options gives another speed for {name}; the server enforces {limit:g} m/s"
        )

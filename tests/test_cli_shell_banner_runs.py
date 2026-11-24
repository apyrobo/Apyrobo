"""Every name the apyrobo shell banner lists exists, and every 'Try' line runs.

The banner listed ``skills → 8 built-in skills available`` but the REPL had
no ``skills`` name, so the first thing a new user typed raised NameError.
The 'Try' lines planned a task but never ran it.
"""
from __future__ import annotations

import argparse
import re
from typing import Any
from unittest.mock import patch

from apyrobo.cli import cmd_shell
from apyrobo.skills.executor import SkillStatus


def _shell() -> tuple[str, dict[str, Any]]:
    seen: dict[str, Any] = {}

    def interact(banner: str = "", local: dict | None = None, exitmsg: str = "") -> None:
        seen["banner"] = banner
        seen["ns"] = local if local is not None else {}

    args = argparse.Namespace(robot="mock://turtlebot4", provider="rule", max_speed=1.5)
    with patch("code.interact", side_effect=interact):
        cmd_shell(args)
    return seen["banner"], seen["ns"]


def _listed_names(banner: str) -> list[str]:
    return re.findall(r"^  (\w+)\s+→", banner, re.M)


def _try_lines(banner: str) -> list[str]:
    block = banner.split("Try:\n", 1)[1].split("\n\n", 1)[0]
    return [line.strip() for line in block.splitlines() if line.strip()]


def test_every_listed_name_is_defined() -> None:
    banner, ns = _shell()
    names = _listed_names(banner)
    assert names == ["robot", "raw_robot", "agent", "skills"]
    for name in names:
        assert name in ns, f"banner lists {name!r} but the shell has no such name"


def test_skills_holds_the_builtin_skills() -> None:
    banner, ns = _shell()
    count = int(re.search(r"skills\s+→ (\d+) built-in", banner).group(1))  # type: ignore[union-attr]
    assert len(ns["skills"]) == count
    assert "navigate_to" in ns["skills"]


def test_try_lines_run_in_order_and_execute_through_the_enforcer() -> None:
    banner, ns = _shell()
    lines = _try_lines(banner)
    assert any("execute_graph" in line for line in lines)
    for line in lines:
        target = line if " = " in line else f"_ = {line}"
        exec(target, ns)  # noqa: S102 - the banner's own lines
        if "execute_graph" in line:
            assert ns["_"] == SkillStatus.COMPLETED
            x, y = ns["raw_robot"].get_position()
            assert (round(x, 2), round(y, 2)) == (2.0, 1.0)
    assert ns["graph"].get_execution_order()

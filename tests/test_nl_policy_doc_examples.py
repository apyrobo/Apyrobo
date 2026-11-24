"""docs/concepts/nl_safety_policies.md: the Python blocks run, and the field list matches the code.

Before this fix the "Runnable example" called ``robot.execute(action)`` on a
name it never defined (and ``Robot`` has no ``execute``), the "Policy storage"
block used ``NLPolicyStore``, ``policy`` and ``policy_id`` without defining
them, and the field list said ``severity: hard`` blocks execution and
``soft`` only warns, while ``check_compliance`` reports both the same way and
nothing blocks.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from apyrobo.safety.nl_policy import NLPolicyParser

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "concepts" / "nl_safety_policies.md"


def _section_block(heading: str) -> str:
    text = DOC.read_text(encoding="utf-8")
    section = text.split(heading, 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"```python\n(.*?)```", section, re.S)
    assert blocks, f"no python block under {heading!r}"
    return blocks[0]


def _run(code: str, home: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env={"HOME": str(home), "PATH": "", "PYTHONPATH": str(ROOT)},
    )


def test_runnable_example_runs_and_blocks(tmp_path):
    result = _run(_section_block("## Runnable example"), tmp_path)
    assert result.returncode == 0, result.stderr
    assert "BLOCKED:" in result.stdout
    assert "speed 2.0 m/s exceeds limit 0.5 m/s" in result.stdout


def test_runnable_example_calls_nothing_undefined():
    block = _section_block("## Runnable example")
    assert "robot." not in block


def test_storage_example_runs_on_a_home_path(tmp_path):
    result = _run(_section_block("## Policy storage"), tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / ".apyrobo" / "policies.db").is_file()


@pytest.mark.parametrize("severity", ["hard", "soft"])
def test_severity_does_not_change_what_the_check_reports(severity):
    parser = NLPolicyParser(severity=severity)
    policy = parser.parse("never exceed 0.5 m/s near humans")
    assert policy.severity == severity
    assert parser.check_compliance("navigate at 2.0 m/s to dock", [policy])


def test_doc_does_not_say_severity_blocks():
    text = DOC.read_text(encoding="utf-8")
    assert "block execution" not in text
    assert "warn only" not in text


def test_doc_lists_every_constraint_type_the_parser_makes():
    text = DOC.read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.startswith("- `constraint_type`"))
    for kind in ("speed_limit", "proximity_limit", "battery_reserve", "no_go_zone",
                 "time_window", "load_limit", "custom"):
        assert kind in line


def test_adapter_guide_says_unitree_needs_an_import():
    text = (ROOT / "docs" / "adapter_authoring.md").read_text(encoding="utf-8")
    row = next(ln for ln in text.splitlines() if ln.startswith("| `UnitreeAdapter`"))
    assert "import apyrobo.core.unitree_adapter" in row

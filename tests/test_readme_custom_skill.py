"""The README's "Custom Skills" block must run as written.

It used to register a new skill with no handler, so the task it ran came back
failed ("No handler registered for 'inspect_shelf'"). This test reads the
block from README.md, runs it in a fresh interpreter and checks the outcome
its comment promises.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADING = "### Custom Skills"


def _block() -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert HEADING in text, f"README.md has no {HEADING!r} section"
    section = text.split(HEADING, 1)[1]
    m = re.search(r"```python\n(.*?)```", section, re.S)
    assert m, "the Custom Skills section has no python block"
    return m.group(1)


def test_block_registers_a_handler_for_its_new_skill():
    block = _block()
    ids = re.findall(r'skill_id="([^"]+)"', block)
    assert ids, "the block defines no skill_id"
    for skill_id in ids:
        assert f'@skill_handler("{skill_id}")' in block, (
            f"the block adds skill {skill_id!r} without a handler"
        )


def test_block_runs_and_prints_what_its_comment_says():
    block = _block()
    promised = re.search(r"^# → (.+)$", block, re.M)
    assert promised, "the block no longer shows its expected output"
    proc = subprocess.run(
        [sys.executable, "-c", block],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines, f"the block printed nothing; stderr: {proc.stderr}"
    assert lines[-1] == promised.group(1).strip(), proc.stdout
    assert lines[-1].startswith("completed"), proc.stdout

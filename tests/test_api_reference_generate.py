"""docs/api_reference.md 'Generating Full API Docs' must run as written.

The section used to say ``pdoc apyrobo --html --output-dir docs/api/`` (pdoc3
flags; pdoc exits 2 "unrecognized arguments: --html") and ``mkdocs build``
with no mkdocs.yml in the repo.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "api_reference.md"


def _generate_section() -> str:
    text = DOC.read_text(encoding="utf-8")
    start = text.index("## Generating Full API Docs")
    nxt = text.find("\n## ", start + 1)
    return text[start:] if nxt == -1 else text[start:nxt]


def _bash_commands(section: str) -> list[str]:
    blocks = re.findall(r"```bash\n(.*?)```", section, re.S)
    lines = []
    for block in blocks:
        for line in block.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def test_section_has_one_pdoc_command() -> None:
    commands = _bash_commands(_generate_section())
    assert commands == ["pip install pdoc", "pdoc apyrobo -o docs/api/"]


def test_no_pdoc3_flags_or_missing_mkdocs_config() -> None:
    section = _generate_section()
    assert "--html" not in section
    assert "--output-dir" not in section
    if not (ROOT / "mkdocs.yml").exists():
        assert "mkdocs" not in section


def test_generated_docs_are_gitignored() -> None:
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "docs/api/" in lines


def test_options_are_ones_pdoc_accepts() -> None:
    pdoc = shutil.which("pdoc")
    if pdoc is None:
        pytest.skip("pdoc not installed")
    out = subprocess.run([pdoc, "--help"], capture_output=True, text=True, timeout=60).stdout
    assert "--output-directory" in out
    assert re.search(r"-o\b", out)
    assert "--html" not in out


def test_pdoc_command_writes_the_page(tmp_path: Path) -> None:
    pdoc = shutil.which("pdoc")
    if pdoc is None:
        pytest.skip("pdoc not installed")
    result = subprocess.run(
        [pdoc, "apyrobo", "-o", str(tmp_path / "api")],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    page = (tmp_path / "api" / "apyrobo.html").read_text(encoding="utf-8")
    assert (tmp_path / "api" / "index.html").exists()
    for name in ("Robot", "Agent", "SkillGraph", "SafetyEnforcer"):
        assert f'id="{name}"' in page

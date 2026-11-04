"""
docs/mcp_safety_demo.svg, the animation at the top of the README, is drawn by
scripts/make_safety_demo_svg.py from what the real ``apyrobo mcp`` server
answers. These tests run that capture and hold the committed file to it.
"""
from __future__ import annotations

import importlib.util
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_safety_demo_svg.py"
SVG = ROOT / "docs" / "mcp_safety_demo.svg"
README = ROOT / "README.md"
SVG_NS = "{http://www.w3.org/2000/svg}"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("make_safety_demo_svg", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def session(script):
    return script.load_example().run_demo()


def _shown_text(svg: str) -> list[str]:
    root = ET.fromstring(svg)
    return [t.text or "" for t in root.iter(f"{SVG_NS}text")]


def test_committed_svg_matches_a_fresh_run(script, session):
    assert SVG.read_text(encoding="utf-8") == script.build(session), (
        "docs/mcp_safety_demo.svg is stale; rerun python scripts/make_safety_demo_svg.py"
    )


def test_svg_shows_the_run_the_clamp_and_the_refusal(script, session):
    shown = "\n".join(_shown_text(script.build(session)))
    assert "moved, robot at (1.0, 0.0)" in shown
    assert "speed_clamped: requested 5.0 m/s, enforced 0.5 m/s" in shown
    assert "REFUSED by safety policy" in shown
    assert "stopped, robot at (2.0, 0.0)" in shown


def test_refusal_is_the_servers_own_text(script, session):
    refusal = next(call["text"] for call in session["calls"] if call["is_error"])
    blocks = script.transcript_lines(session, "apyrobo mcp")
    drawn = [text.strip() for block in blocks for kind, text in block if kind == "refused"]
    assert " ".join(drawn) == refusal


def test_a_changed_reply_changes_the_picture(script, session):
    changed = json.loads(json.dumps(session))
    reply = json.loads(changed["calls"][2]["text"])
    reply["safety_interventions"][0]["enforced"] = 0.25
    changed["calls"][2]["text"] = json.dumps(reply)
    assert script.build(changed) != script.build(session)
    assert "enforced 0.25 m/s" in script.build(changed)


def test_no_line_runs_past_the_terminal(script, session):
    for block in script.transcript_lines(session, "apyrobo mcp --robot mock://turtlebot4"):
        for _, text in block:
            assert len(text) <= script.COLUMNS, text


def test_file_is_under_one_megabyte():
    assert SVG.stat().st_size < 1_000_000


def test_readme_shows_it_near_the_top():
    readme = README.read_text(encoding="utf-8")
    assert "docs/mcp_safety_demo.svg" in readme
    position = readme.index("docs/mcp_safety_demo.svg")
    assert position < readme.index("demos/orchestration_flow/demo.gif")
    assert re.search(r"scripts/make_safety_demo_svg\.py", readme)

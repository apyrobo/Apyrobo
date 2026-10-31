"""
examples/05_mcp_safety_demo.py drives the real ``apyrobo mcp`` server over
stdio and shows what it claims: a normal move runs, an over-limit move is
clamped, a no-go target is refused, stop works and safety_report records the
clamp and the refusal. docs/mcp_demo_transcript.md must match its output.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "05_mcp_safety_demo.py"
TRANSCRIPT = ROOT / "docs" / "mcp_demo_transcript.md"


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("mcp_demo_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def session(example):
    return example.run_demo()


def _call(session, i):
    call = session["calls"][i]
    return call, (None if call["is_error"] else json.loads(call["text"]))


def test_handshake_and_tools(session):
    assert session["server_name"] == "apyrobo"
    assert session["protocol_version"] == "2025-06-18"
    for tool in ("robot_status", "move_to", "stop", "safety_report"):
        assert tool in session["tools"]


def test_status_shows_policy(session):
    _, status = _call(session, 0)
    policy = status["safety_policy"]
    assert policy["max_speed_m_s"] == 0.5
    assert policy["no_go_zones"] == [{"x_min": 2.0, "y_min": 2.0, "x_max": 4.0, "y_max": 4.0}]


def test_normal_move_runs_untouched(session):
    call, result = _call(session, 1)
    assert not call["is_error"]
    assert result["position"] == {"x": 1.0, "y": 0.0}
    assert result["safety_interventions"] == []


def test_over_limit_move_is_clamped(session):
    call, result = _call(session, 2)
    assert not call["is_error"]
    assert result["position"] == {"x": 2.0, "y": 0.0}
    [clamp] = result["safety_interventions"]
    assert clamp["type"] == "speed_clamped"
    assert clamp["requested"] == 5.0 and clamp["enforced"] == 0.5


def test_no_go_target_is_refused_and_robot_stays(session):
    call, _ = _call(session, 3)
    assert call["is_error"]
    assert call["text"].startswith("REFUSED by safety policy")
    assert "did not move" in call["text"]
    _, stopped = _call(session, 4)
    assert stopped["status"] == "stopped"
    assert stopped["position"] == {"x": 2.0, "y": 0.0}


def test_safety_report_audits_clamp_and_refusal(session):
    _, report = _call(session, 5)
    assert [i["type"] for i in report["interventions"]] == ["speed_clamped"]
    assert [v["type"] for v in report["violations"]] == ["collision_zone"]
    assert report["violations"][0]["requested_position"] == [3.0, 3.0]


def test_committed_transcript_matches(example, session):
    assert TRANSCRIPT.read_text(encoding="utf-8") == example.to_markdown(session), (
        "docs/mcp_demo_transcript.md is stale; regenerate it with "
        "python examples/05_mcp_safety_demo.py --write docs/mcp_demo_transcript.md --quiet"
    )


def test_mcp_docs_link_the_transcript_and_example():
    docs = (ROOT / "docs" / "mcp.md").read_text(encoding="utf-8")
    assert "](mcp_demo_transcript.md)" in docs
    assert "examples/05_mcp_safety_demo.py" in docs
    assert EXAMPLE.exists() and TRANSCRIPT.exists()

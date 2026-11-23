"""MIGRATION.md says what the code does.

Two claims were wrong: the v0.4.0 section told users to put the REST API key
in apyrobo.yaml (nothing reads it there, so the gateway stayed open), and the
v4.1.0 section said Agent(provider="llm") raises without litellm (it falls
back to the rule planner).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from apyrobo import Agent, Robot
from apyrobo.observability import clear_event_handlers, on_event
from apyrobo.skills.agent import RuleBasedProvider

DOC = Path(__file__).resolve().parents[1] / "MIGRATION.md"


def _section(title: str) -> str:
    text = DOC.read_text()
    start = text.index(title)
    nxt = text.find("\n## ", start + len(title))
    return text[start: nxt if nxt != -1 else len(text)]


# --- v0.4.0: REST API key ---------------------------------------------------

def test_api_key_section_names_the_env_var_not_yaml():
    sec = _section("## Migrating to v0.4.0")
    assert "APYROBO_API_KEY" in sec
    assert not re.search(r"```yaml\s*\napi:\s*\n\s+key:", sec), (
        "MIGRATION.md tells users to set api.key in apyrobo.yaml, "
        "which nothing reads"
    )


def test_no_code_reads_an_api_key_from_config():
    # If someone wires apyrobo.yaml's api.key into the gateway, the doc can
    # say so again; until then this guards the sentence that it does not.
    src = (Path(__file__).resolve().parents[1] / "apyrobo" / "api" / "app.py").read_text()
    assert "APYROBO_API_KEY" in src
    assert "load_config" not in src and "ApyroboConfig" not in src


def test_env_key_is_enforced_and_health_stays_public(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from apyrobo.api.app import create_app

    monkeypatch.setenv("APYROBO_API_KEY", "your-api-key")
    client = TestClient(create_app())
    assert client.get("/health").status_code == 200
    assert client.get("/robots").status_code == 401
    ok = client.get("/robots", headers={"X-API-Key": "your-api-key"})
    assert ok.status_code == 200


def test_no_key_means_no_auth(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from apyrobo.api.app import create_app

    monkeypatch.delenv("APYROBO_API_KEY", raising=False)
    client = TestClient(create_app())
    assert client.get("/robots").status_code == 200


# --- v4.1.0: provider="llm" without litellm ----------------------------------

def test_doc_does_not_say_llm_provider_raises():
    sec = _section("## Migrating to v4.1.0")
    assert "raises `RuntimeError`" not in sec
    assert "falls back" in sec or "rule planner instead" in sec


def test_llm_provider_without_litellm_falls_back(monkeypatch, caplog):
    # A None entry in sys.modules makes `import litellm` raise ImportError,
    # the same as an install without apyrobo[llm].
    monkeypatch.setitem(sys.modules, "litellm", None)
    events = []
    clear_event_handlers()
    on_event(events.append)
    try:
        robot = Robot.discover("mock://tb4")
        agent = Agent(provider="llm")
        with caplog.at_level("WARNING"):
            graph = agent.plan("go to (1, 2)", robot)
    finally:
        clear_event_handlers()

    assert len(graph) >= 1
    assert isinstance(agent.planned_by, RuleBasedProvider)
    assert "pip install 'apyrobo[llm]'" in caplog.text
    assert any(e.event_type == "agent.degraded" for e in events)


def test_auto_without_litellm_emits_documented_reason(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", None)
    events = []
    clear_event_handlers()
    on_event(events.append)
    try:
        agent = Agent(provider="auto")
    finally:
        clear_event_handlers()
    assert isinstance(agent.provider, RuleBasedProvider)
    degraded = [e for e in events if e.event_type == "agent.degraded"]
    assert degraded and degraded[0].data["reason"] == "litellm_not_installed"
    assert "litellm_not_installed" in _section("## Migrating to v4.1.0")

"""docs/deployment.md only promises what the REST gateway does.

The guide used to document APYROBO_REDIS_URL ("shared state"),
APYROBO_AUDIT_DB and APYROBO_LOG_LEVEL, a Prometheus ``GET /metrics`` and a
2-replica Kubernetes deployment sharing state through Redis. apyrobo.api.app
reads only APYROBO_API_KEY, keeps tasks in process memory and serves no
/metrics, so on the default 2 replicas a task lookup could land on the other
pod and answer 404.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "deployment.md"

APP = ROOT / "apyrobo" / "api" / "app.py"


def _client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from apyrobo.api.app import create_app

    return TestClient(create_app(api_key="k"))


def _section(heading: str) -> str:
    text = DOC.read_text()
    start = text.index(heading)
    nxt = text.find("\n## ", start + len(heading))
    return text[start: nxt if nxt != -1 else len(text)]


def _table_first_cells(section: str) -> list[str]:
    cells = []
    for line in section.splitlines():
        if line.startswith("|") and not line.startswith("|--"):
            cells.append(line.split("|")[1].strip())
    return cells[1:]  # drop the header row


def test_every_documented_variable_is_read_by_the_gateway():
    source = "".join(p.read_text() for p in (ROOT / "apyrobo" / "api").glob("*.py"))
    table = _section("## Environment Variables Reference")
    names = [c.strip("`") for c in _table_first_cells(table)]
    assert names, "no variables in the table"
    for name in names:
        assert f'"{name}"' in source, f"docs/deployment.md lists {name}; apyrobo.api never reads it"


def test_every_documented_endpoint_is_served():
    # Read the routes from the source so this runs without fastapi installed.
    routes = set(re.findall(r'@app\.\w+\(\s*"([^"]+)"', APP.read_text()))
    assert "/health" in routes
    for cell in _table_first_cells(_section("## Health Check Endpoints")):
        method, path = cell.strip("`").split()
        assert path in routes, f"docs/deployment.md lists {method} {path}; the gateway lacks it"


def test_metrics_is_not_served_and_the_doc_says_so():
    assert "/metrics" not in APP.read_text()
    assert "serves no `/metrics`" in DOC.read_text()
    assert _client().get("/metrics").status_code == 404


def test_health_reply_matches_the_doc():
    body = _client().get("/health").json()
    assert body == {"status": "ok", "tasks": 0, "robots": 0}
    assert '{"status":"ok","tasks":0,"robots":0}' in DOC.read_text()


def test_kubernetes_runs_one_replica_and_the_doc_says_so():
    deployment = yaml.safe_load((ROOT / "k8s" / "deployment-api.yaml").read_text())
    assert deployment["spec"]["replicas"] == 1
    assert "**1 replica**" in DOC.read_text()


def test_state_is_per_process():
    # Why one replica: a second gateway process does not know the first one's tasks.
    headers = {"X-API-Key": "k"}
    task = _client().post("/tasks", json={"skill": "navigate_to"}, headers=headers).json()
    assert _client().get(f"/tasks/{task['task_id']}", headers=headers).status_code == 404

"""Tests for apyrobo.api.app"""
import pytest

try:
    from fastapi.testclient import TestClient
    from apyrobo.api.app import create_app
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _FASTAPI_AVAILABLE, reason="fastapi not installed"
)


@pytest.fixture
def client():
    app = create_app(api_key="test-key")
    return TestClient(app)


@pytest.fixture
def authed(client):
    """Client with auth header pre-set."""
    client.headers["X-API-Key"] = "test-key"
    return client


class TestHealth:
    def test_health_public(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_no_auth_required(self, client):
        # No auth header — should still return 200
        resp = client.get("/health")
        assert resp.status_code == 200


class TestAuth:
    def test_submit_task_requires_auth(self, client):
        resp = client.post("/tasks", json={"skill": "navigate_to", "params": {}})
        assert resp.status_code == 401

    def test_invalid_key_rejected(self, client):
        resp = client.post(
            "/tasks",
            json={"skill": "navigate_to", "params": {}},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401


class TestTasks:
    def test_submit_task(self, authed):
        resp = authed.post("/tasks", json={"skill": "navigate_to", "params": {"x": 1, "y": 2}})
        assert resp.status_code == 201
        data = resp.json()
        assert "task_id" in data
        assert data["skill"] == "navigate_to"
        assert data["status"] == "queued"

    def test_get_task(self, authed):
        resp = authed.post("/tasks", json={"skill": "stop", "params": {}})
        task_id = resp.json()["task_id"]

        resp2 = authed.get(f"/tasks/{task_id}")
        assert resp2.status_code == 200
        assert resp2.json()["task_id"] == task_id

    def test_get_missing_task(self, authed):
        resp = authed.get("/tasks/nonexistent-id")
        assert resp.status_code == 404

    def test_submit_with_robot_id(self, authed):
        resp = authed.post(
            "/tasks",
            json={"skill": "navigate_to", "params": {}, "robot_id": "tb4_1"},
        )
        assert resp.status_code == 201
        assert resp.json()["robot_id"] == "tb4_1"


class TestRobots:
    def test_list_robots_empty(self, authed):
        resp = authed.get("/robots")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_execute_skill_registers_robot(self, authed):
        resp = authed.post(
            "/robots/tb4_1/skills/navigate_to",
            json={"params": {"x": 0, "y": 0}},
        )
        assert resp.status_code == 200
        assert "task_id" in resp.json()

        # Robot should now be listed
        robots = authed.get("/robots").json()
        robot_ids = [r["robot_id"] for r in robots]
        assert "tb4_1" in robot_ids

    def test_no_auth_key_app(self):
        """App with no API key should allow all requests."""
        app = create_app(api_key="")
        client = TestClient(app)
        resp = client.post("/tasks", json={"skill": "stop", "params": {}})
        assert resp.status_code == 201

import json
import time
import urllib.error
from urllib import request as urllib_request

from apyrobo.auth import AuthManager
from apyrobo.observability import clear_event_handlers, on_event
from apyrobo.operations import (
    BatteryMonitor,
    FleetDashboard,
    MapManager,
    OperationsApiServer,
    ScheduledTaskRunner,
    TeleoperationBridge,
    WebhookEmitter,
    _parse_cron_to_seconds,
)


def test_battery_monitor_return_to_dock_trigger():
    monitor = BatteryMonitor("r1", critical_threshold=10.0)
    triggered = {"value": False}
    monitor.set_return_to_dock_callback(lambda: triggered.__setitem__("value", True))

    monitor.update(percentage=8.0, is_charging=False)
    assert monitor.evaluate_return_to_dock() is True
    assert triggered["value"] is True


def test_teleoperation_bridge_callback_wiring():
    bridge = TeleoperationBridge("r1")
    sent = []
    bridge.set_velocity_callback(lambda l, a: sent.append((l, a)))
    bridge.enable("op")

    assert bridge.send_velocity(0.5, 0.1) is True
    bridge.disable()
    assert sent[0] == (0.5, 0.1)


def test_webhook_emitter_slack_teams_targets_registration():
    emitter = WebhookEmitter(retry_count=1)
    emitter.add_slack_target("slack", "http://localhost:1/slack")
    emitter.add_teams_target("teams", "http://localhost:1/teams")

    assert emitter.target_count == 2


def test_map_manager_register_and_set_active():
    manager = MapManager()
    manager.register("floor1", "/tmp/floor1.yaml", floor=1)
    manager.set_active("floor1")

    assert manager.active_map_name == "floor1"
    assert manager.get_floor_map(1)["name"] == "floor1"


def test_scheduled_task_runner_executes_job():
    runner = ScheduledTaskRunner()
    counter = {"n": 0}
    runner.add_interval_job("job", 0.05, lambda: counter.__setitem__("n", counter["n"] + 1))
    runner.start()
    time.sleep(0.2)
    runner.stop()
    assert counter["n"] >= 1


def test_operations_api_server_basic_routes():
    server = OperationsApiServer(port=18081)
    server.set_robots([{"id": "r1"}])
    server.start()
    time.sleep(0.05)
    try:
        with urllib_request.urlopen("http://127.0.0.1:18081/health", timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            assert payload["status"] == "ok"

        req = urllib_request.Request(
            "http://127.0.0.1:18081/tasks",
            method="POST",
            data=json.dumps({"task": "patrol"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib_request.urlopen(req, timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            assert payload["task_id"]
            assert payload["status"] == "queued"
    finally:
        server.stop()


def test_fleet_dashboard_snapshot():
    dash = FleetDashboard()
    dash.update_robot("r1", {"status": "ok"})
    dash.update_task("t1", {"status": "running"})

    snap = dash.snapshot()
    assert snap["robots"]["r1"]["status"] == "ok"
    assert snap["tasks"]["t1"]["status"] == "running"


# ---- OP-02: ScheduledTaskRunner tests ------------------------------------


def test_parse_cron_expressions():
    assert _parse_cron_to_seconds("*/30 * * * *") == 1800.0   # every 30 min
    assert _parse_cron_to_seconds("*/5 * * * *") == 300.0     # every 5 min
    assert _parse_cron_to_seconds("0 2 * * *") == 86400.0     # daily at 2am
    assert _parse_cron_to_seconds("0 */4 * * *") == 14400.0   # every 4 hours
    assert _parse_cron_to_seconds("* * * * *") == 60.0        # every minute
    assert _parse_cron_to_seconds("invalid") == 3600.0        # fallback


def test_scheduled_task_runner_multiple_intervals():
    runner = ScheduledTaskRunner()
    c1 = {"n": 0}
    c2 = {"n": 0}
    runner.add_interval_job("fast", 0.05, lambda: c1.__setitem__("n", c1["n"] + 1))
    runner.add_interval_job("slow", 0.15, lambda: c2.__setitem__("n", c2["n"] + 1))
    runner.start()
    time.sleep(0.35)
    runner.stop()
    assert c1["n"] >= 3
    assert c2["n"] >= 1
    assert c1["n"] > c2["n"]


def test_scheduled_task_runner_stop_prevents_execution():
    runner = ScheduledTaskRunner()
    counter = {"n": 0}
    runner.add_interval_job("job", 0.05, lambda: counter.__setitem__("n", counter["n"] + 1))
    runner.start()
    time.sleep(0.15)
    runner.stop()
    count_at_stop = counter["n"]
    time.sleep(0.15)
    assert counter["n"] == count_at_stop


def test_scheduled_task_runner_failed_job_doesnt_stop_scheduler():
    runner = ScheduledTaskRunner()
    counter = {"n": 0}

    def fail():
        raise RuntimeError("boom")

    runner.add_interval_job("failing", 0.05, fail)
    runner.add_interval_job("ok", 0.05, lambda: counter.__setitem__("n", counter["n"] + 1))
    runner.start()
    time.sleep(0.2)
    runner.stop()
    assert counter["n"] >= 1


def test_scheduled_task_runner_emit_event():
    clear_event_handlers()
    events = []
    on_event(lambda e: events.append(e))

    runner = ScheduledTaskRunner()
    runner.add_interval_job("job", 0.05, lambda: None)
    runner.start()
    time.sleep(0.15)
    runner.stop()

    clear_event_handlers()

    task_events = [e for e in events if e.event_type == "scheduled_task_run"]
    assert len(task_events) >= 1
    assert task_events[0].data["task_name"] == "job"
    assert task_events[0].data["status"] == "success"


def test_scheduled_task_runner_add_task_agent_mode():
    class FakeResult:
        status = "completed"

    class FakeAgent:
        def execute(self, task, robot, **kw):
            robot["executed"] = True
            return FakeResult()

    robot = {"id": "r1"}
    runner = ScheduledTaskRunner()
    runner.add_task("patrol", "*/5 * * * *", "patrol the area", robot, FakeAgent())
    # Override interval for fast testing
    runner._jobs[-1]["interval_s"] = 0.05
    runner._jobs[-1]["next"] = time.time() + 0.05
    runner.start()
    time.sleep(0.2)
    runner.stop()
    assert robot.get("executed") is True


def test_scheduled_task_runner_emit_event_on_failure():
    clear_event_handlers()
    events = []
    on_event(lambda e: events.append(e))

    runner = ScheduledTaskRunner()
    def _raise_oops():
        raise RuntimeError("oops")

    runner.add_interval_job("fail_job", 0.05, _raise_oops)
    runner.start()
    time.sleep(0.15)
    runner.stop()

    clear_event_handlers()

    err_events = [e for e in events if e.event_type == "scheduled_task_run" and e.data.get("status") == "error"]
    assert len(err_events) >= 1
    assert "oops" in err_events[0].data["error"]


# ---- OP-01: OperationsApiServer tests ------------------------------------


def _api_request(port, path, method="GET", body=None, headers=None):
    """Helper to make HTTP requests to the test server."""
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode("utf-8") if body else None
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib_request.Request(url, method=method, data=data, headers=hdrs)
    try:
        with urllib_request.urlopen(req, timeout=2) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_api_post_tasks_returns_202_with_task_id():
    server = OperationsApiServer(port=18082)
    server.start()
    time.sleep(0.05)
    try:
        code, data = _api_request(18082, "/tasks", method="POST", body={"task": "deliver"})
        assert code == 202
        assert "task_id" in data
        assert data["status"] == "queued"
    finally:
        server.stop()


def test_api_get_task_status():
    server = OperationsApiServer(port=18083)
    server.start()
    time.sleep(0.05)
    try:
        _, data = _api_request(18083, "/tasks", method="POST", body={"task": "patrol"})
        task_id = data["task_id"]
        time.sleep(0.1)  # let background thread complete

        code, info = _api_request(18083, f"/tasks/{task_id}")
        assert code == 200
        assert info["task_id"] == task_id
    finally:
        server.stop()


def test_api_get_robots_returns_list():
    server = OperationsApiServer(port=18084)
    server.set_robots([{"id": "r1", "capabilities": {"name": "bot1"}}])
    server.start()
    time.sleep(0.05)
    try:
        code, data = _api_request(18084, "/robots")
        assert code == 200
        assert len(data["robots"]) == 1
        assert data["robots"][0]["id"] == "r1"
    finally:
        server.stop()


def test_api_invalid_api_key_returns_401():
    auth = AuthManager()
    auth.add_user("admin", role="admin", api_key="secret-key-123")

    server = OperationsApiServer(port=18085, auth_manager=auth)
    server.start()
    time.sleep(0.05)
    try:
        # No key → 401
        code, data = _api_request(18085, "/health")
        assert code == 401

        # Wrong key → 401
        code, data = _api_request(18085, "/health", headers={"X-API-Key": "wrong"})
        assert code == 401

        # Valid key → 200
        code, data = _api_request(18085, "/health", headers={"X-API-Key": "secret-key-123"})
        assert code == 200
        assert data["status"] == "ok"
    finally:
        server.stop()


def test_api_task_background_status_updates():
    class FakeResult:
        status = "completed"
        def to_dict(self):
            return {"status": "completed"}

    class FakeAgent:
        def execute(self, task, robot, **kw):
            return FakeResult()

    class FakeRobot:
        pass

    class FakeBus:
        @property
        def robot_ids(self):
            return ["r1"]
        def get_robot(self, rid):
            return FakeRobot()
        def get_capabilities(self, rid):
            return type("Cap", (), {"model_dump": lambda self: {"name": "bot"}})()

    server = OperationsApiServer(
        port=18086,
        agent=FakeAgent(),
        swarm_bus=FakeBus(),
    )
    server.start()
    time.sleep(0.05)
    try:
        _, data = _api_request(18086, "/tasks", method="POST",
                               body={"task": "deliver", "robot_id": "r1"})
        task_id = data["task_id"]
        time.sleep(0.2)  # let background execution finish

        code, info = _api_request(18086, f"/tasks/{task_id}")
        assert code == 200
        assert info["status"] == "completed"
    finally:
        server.stop()


def test_api_delete_task_cancels():
    server = OperationsApiServer(port=18087)
    server.start()
    time.sleep(0.05)
    try:
        _, data = _api_request(18087, "/tasks", method="POST", body={"task": "patrol"})
        task_id = data["task_id"]

        code, info = _api_request(18087, f"/tasks/{task_id}", method="DELETE")
        assert code == 200
        assert info["status"] == "cancelled"
    finally:
        server.stop()


def test_api_get_nonexistent_task_returns_404():
    server = OperationsApiServer(port=18088)
    server.start()
    time.sleep(0.05)
    try:
        code, data = _api_request(18088, "/tasks/nonexistent")
        assert code == 404
    finally:
        server.stop()

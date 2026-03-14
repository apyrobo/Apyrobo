import json
import time
from urllib import request as urllib_request

from apyrobo.operations import (
    BatteryMonitor,
    FleetDashboard,
    MapManager,
    OperationsApiServer,
    ScheduledTaskRunner,
    TeleoperationBridge,
    WebhookEmitter,
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
            assert payload["accepted"] is True
    finally:
        server.stop()


def test_fleet_dashboard_snapshot():
    dash = FleetDashboard()
    dash.update_robot("r1", {"status": "ok"})
    dash.update_task("t1", {"status": "running"})

    snap = dash.snapshot()
    assert snap["robots"]["r1"]["status"] == "ok"
    assert snap["tasks"]["t1"]["status"] == "running"

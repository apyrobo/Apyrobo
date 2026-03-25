"""Tests for apyrobo.fleet.manager"""
import time
import pytest
from apyrobo.fleet.manager import FleetManager, RobotInfo


def make_robot(robot_id: str, caps: list[str] | None = None, status: str = "idle") -> RobotInfo:
    return RobotInfo(robot_id=robot_id, capabilities=caps or ["move"], status=status)


class TestRobotInfo:
    def test_is_available_idle_with_caps(self):
        r = make_robot("tb4", ["move", "gripper"])
        assert r.is_available_for(["move"]) is True
        assert r.is_available_for(["move", "gripper"]) is True

    def test_is_available_missing_cap(self):
        r = make_robot("tb4", ["move"])
        assert r.is_available_for(["gripper"]) is False

    def test_is_available_busy(self):
        r = make_robot("tb4", ["move"], status="busy")
        assert r.is_available_for(["move"]) is False

    def test_is_available_offline(self):
        r = make_robot("tb4", ["move"], status="offline")
        assert r.is_available_for(["move"]) is False

    def test_no_requirements_always_available(self):
        r = make_robot("tb4")
        assert r.is_available_for([]) is True


class TestFleetManager:
    def setup_method(self):
        self.fm = FleetManager()

    def test_register(self):
        self.fm.register(make_robot("r1"))
        assert len(self.fm) == 1

    def test_heartbeat_updates_timestamp(self):
        self.fm.register(make_robot("r1"))
        old_hb = self.fm.get_robot("r1").last_heartbeat
        time.sleep(0.01)
        self.fm.heartbeat("r1")
        assert self.fm.get_robot("r1").last_heartbeat > old_hb

    def test_heartbeat_unknown_robot_raises(self):
        with pytest.raises(KeyError):
            self.fm.heartbeat("nonexistent")

    def test_assign_task_picks_idle_robot(self):
        self.fm.register(make_robot("r1", ["move"]))
        robot_id = self.fm.assign_task({"required": ["move"], "task_id": "t1"})
        assert robot_id == "r1"
        assert self.fm.get_robot("r1").status == "busy"

    def test_assign_task_no_robot_available(self):
        self.fm.register(make_robot("r1", ["move"], status="busy"))
        robot_id = self.fm.assign_task({"required": ["move"], "task_id": "t1"})
        assert robot_id is None

    def test_assign_task_capability_mismatch(self):
        self.fm.register(make_robot("r1", ["move"]))
        robot_id = self.fm.assign_task({"required": ["gripper"], "task_id": "t1"})
        assert robot_id is None

    def test_assign_task_load_balancing(self):
        # Register two robots with different heartbeat times
        r1 = make_robot("r1", ["move"])
        r2 = make_robot("r2", ["move"])
        r1.last_heartbeat = time.time() - 10
        r2.last_heartbeat = time.time()
        self.fm.register(r1)
        self.fm.register(r2)
        # Should assign to r1 (older heartbeat = least recently used)
        robot_id = self.fm.assign_task({"required": ["move"], "task_id": "t1"})
        assert robot_id == "r1"

    def test_complete_task_sets_idle(self):
        self.fm.register(make_robot("r1", ["move"]))
        self.fm.assign_task({"required": ["move"], "task_id": "t1"})
        self.fm.complete_task("r1")
        assert self.fm.get_robot("r1").status == "idle"

    def test_get_status(self):
        self.fm.register(make_robot("r1"))
        self.fm.register(make_robot("r2", status="busy"))
        status = self.fm.get_status()
        assert status["total"] == 2
        assert status["idle"] == 1
        assert status["busy"] == 1

    def test_offline_robots_detection(self):
        r = make_robot("stale")
        r.last_heartbeat = time.time() - 60  # definitely stale
        self.fm.register(r)
        offline = self.fm.offline_robots(timeout_sec=30.0)
        assert "stale" in offline
        assert self.fm.get_robot("stale").status == "offline"

    def test_heartbeat_brings_offline_robot_online(self):
        r = make_robot("r1")
        self.fm.register(r)
        r.status = "offline"
        self.fm.heartbeat("r1")
        assert self.fm.get_robot("r1").status == "idle"

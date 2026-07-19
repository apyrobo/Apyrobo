"""Tests for the real mujoco:// physics bridge (apyrobo/sim/mujoco_bridge.py).

Everything here steps actual MuJoCo physics — skipped when the mujoco
package isn't installed (`pip install 'apyrobo[mujoco]'`).
"""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("mujoco")

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import CapabilityType, TaskStatus
from apyrobo.sim.mujoco_bridge import MuJoCoBridgeAdapter


@pytest.fixture
def adapter():
    a = MuJoCoBridgeAdapter("test_bot")
    yield a
    a.shutdown()


class TestBridgeBasics:
    def test_discover_via_scheme(self):
        robot = Robot.discover("mujoco://scheme_bot")
        try:
            assert isinstance(robot._adapter, MuJoCoBridgeAdapter)
            assert robot._adapter.get_capabilities().metadata["real_physics"] is True
        finally:
            robot._adapter.shutdown()

    def test_capabilities(self, adapter):
        caps = adapter.get_capabilities()
        types = {c.capability_type for c in caps.capabilities}
        assert {CapabilityType.NAVIGATE, CapabilityType.ROTATE,
                CapabilityType.PICK, CapabilityType.PLACE} <= types
        assert caps.metadata["backend"] == "mujoco"

    def test_move_blocks_until_arrival(self, adapter):
        adapter.move(1.0, 0.5)
        x, y = adapter.get_position()
        assert abs(x - 1.0) < 0.1 and abs(y - 0.5) < 0.1
        assert adapter.is_moving is False

    def test_sim_time_advances_with_travel(self, adapter):
        t0 = adapter.get_health()["sim_time_s"]
        adapter.move(-1.0, 0.0, speed=1.0)
        elapsed = adapter.get_health()["sim_time_s"] - t0
        # 1 m at 1 m/s can't take less than 1 s of *sim* time — physics,
        # not teleportation.
        assert elapsed >= 0.9

    def test_speed_is_capped(self, adapter):
        t0 = adapter.get_health()["sim_time_s"]
        adapter.move(1.0, 0.0, speed=999.0)
        elapsed = adapter.get_health()["sim_time_s"] - t0
        assert elapsed >= 1.0 / adapter.max_speed * 0.8

    def test_rotate(self, adapter):
        yaw0 = adapter.get_orientation()
        adapter.rotate(1.0)
        assert abs(adapter.get_orientation() - yaw0 - 1.0) < 0.1


class TestStopAndLifecycle:
    def test_cross_thread_stop_interrupts_blocking_move(self):
        a = MuJoCoBridgeAdapter("rt_bot", realtime=True)
        try:
            done = {}

            def long_move():
                a.move(4.0, 0.0, speed=0.3)
                done["pos"] = a.get_position()

            t = threading.Thread(target=long_move)
            t.start()
            time.sleep(0.4)
            a.stop()
            t.join(timeout=3.0)
            assert not t.is_alive()
            assert done["pos"][0] < 1.0
        finally:
            a.shutdown()

    def test_commands_fail_fast_when_disconnected(self, adapter):
        adapter.disconnect()
        with pytest.raises(ConnectionError):
            adapter.move(1.0, 1.0)
        with pytest.raises(ConnectionError):
            adapter.gripper_close()

    def test_stop_safe_while_disconnected(self, adapter):
        adapter.disconnect()
        adapter.stop()  # must not raise — the safety-critical path

    def test_reconnect_resumes(self, adapter):
        adapter.disconnect()
        adapter.connect()
        adapter.move(0.3, 0.0)
        assert abs(adapter.get_position()[0] - 0.3) < 0.1


class TestGrasp:
    def test_grasp_refused_out_of_reach(self, adapter):
        assert adapter.gripper_close() is False  # package is ~1.1 m away

    def test_pick_carry_place(self, adapter):
        start = adapter.object_position()
        adapter.move(0.85, 0.42)
        assert adapter.gripper_close() is True
        # Suction lift: the constraint holds the object off the floor.
        adapter.move(0.5, 0.4)
        assert adapter.object_position()[2] > start[2] + 0.03
        adapter.move(-1.0, -0.8)
        assert adapter.gripper_open() is True
        # Let it fall and settle.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            adapter.move(-1.0, -0.8)  # a few extra sim steps
            if abs(adapter.object_position()[2] - start[2]) < 0.01:
                break
        px, py, pz = adapter.object_position()
        assert abs(px - (-1.0)) < 0.5 and abs(py - (-0.8)) < 0.5
        assert abs(pz - start[2]) < 0.01  # back on the floor


class TestFullPipeline:
    def test_nl_task_delivers_package(self):
        from apyrobo.skills.agent import Agent

        robot = Robot.discover("mujoco://pipeline_bot")
        try:
            agent = Agent(provider="rule")
            result = agent.execute(
                task="deliver package from (0.85, 0.42) to (-1.0, -0.8)",
                robot=robot,
            )
            assert result.status == TaskStatus.COMPLETED
            px, py, _ = robot._adapter.object_position()
            assert abs(px - (-1.0)) < 0.5 and abs(py - (-0.8)) < 0.5
        finally:
            robot._adapter.shutdown()


class TestConformance:
    def test_full_conformance_suite(self):
        from apyrobo.conformance.adapter_checks import run_adapter_checks
        from apyrobo.conformance.report import ConformanceReport

        a = MuJoCoBridgeAdapter("conf_bot")
        try:
            report = ConformanceReport(target="mujoco://conf_bot", kind="adapter")
            run_adapter_checks(a, report)
            must_fails = [
                c for c in report.checks
                if c.level == "MUST" and c.status == "fail"
            ]
            assert report.conformant, f"MUST failures: {must_fails}"
            warnings = [c for c in report.checks if c.status == "warn"]
            assert warnings == [], f"expected 0 warnings: {warnings}"
        finally:
            a.shutdown()

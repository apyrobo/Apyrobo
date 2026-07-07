"""Smoke tests for the live fleet view demo (demos/fleet_view/server.py).

The demo's value is visual, but its planning/execution logic is plain
Python and worth guarding: task text must map to the right target, and the
server must actually move the addressed robot.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from apyrobo import Agent
from apyrobo.orchestration.adapter import OrchestrationMessage

DEMO = Path(__file__).resolve().parents[1] / "demos" / "fleet_view" / "server.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fleet_view_server", DEMO)
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves defaults via sys.modules[cls.__module__]; register
    # before exec so SimRobot's field(default=…) doesn't hit a None module.
    sys.modules["fleet_view_server"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fv():
    return _load_module()


class _CollectingAdapter:
    """Stand-in orchestration adapter — the server never calls receive()."""

    def receive(self):
        return None

    def send(self, msg):
        pass

    def startup(self):
        pass

    def shutdown(self):
        pass


@pytest.fixture
def server(fv):
    fleet = fv.build_fleet()
    return fv.FleetViewServer(_CollectingAdapter(), Agent(provider="rule"), fleet), fleet


class TestTargeting:
    def test_named_zone_maps_to_zone_coords(self, fv):
        fleet = fv.build_fleet()
        srv = fv.FleetViewServer(_CollectingAdapter(), Agent(provider="rule"), fleet)
        assert srv._target_for("deliver a package to the dock") == fv.ZONES["dock"]
        assert srv._target_for("inspect the KITCHEN now") == fv.ZONES["kitchen"]

    def test_explicit_coords_are_parsed_and_clamped(self, fv):
        fleet = fv.build_fleet()
        srv = fv.FleetViewServer(_CollectingAdapter(), Agent(provider="rule"), fleet)
        assert srv._target_for("go to (20, 40)") == (20.0, 40.0)
        # Out-of-world coords clamp into bounds, never off-map.
        x, y = srv._target_for("go to (9999, -9999)")
        assert 0 <= x <= fv.WORLD[0] and 0 <= y <= fv.WORLD[1]

    def test_unmatched_task_gets_in_bounds_waypoint(self, fv):
        fleet = fv.build_fleet()
        srv = fv.FleetViewServer(_CollectingAdapter(), Agent(provider="rule"), fleet)
        x, y = srv._target_for("do something whimsical")
        assert 0 <= x <= fv.WORLD[0] and 0 <= y <= fv.WORLD[1]


class TestExecution:
    def test_handle_moves_the_addressed_robot(self, server):
        srv, fleet = server
        target_id = fleet[-1].robot_id  # a ground robot
        response = srv._handle(OrchestrationMessage(
            task="deliver a package to the dock",
            robot_uri=f"mock://{target_id}",
        ))
        assert response.metadata["status"] == "planned"
        sim = srv.fleet[target_id]
        assert sim.target == pytest.approx(srv._target_for("deliver a package to the dock"))
        assert response.metadata["target"] == list(sim.target)

    def test_tick_advances_toward_target_and_arrives(self, fv):
        sim = fv.SimRobot(
            robot_id="t", kind="ground", x=0.0, y=0.0, speed=5.0,
            adapter=None,  # set below
        )
        sim.adapter = fv.MockAdapter("t")
        sim.target = (10.0, 0.0)
        sim.tick(1.0)          # 5 m of a 10 m trip
        assert 4.9 < sim.x < 5.1 and sim.target is not None
        for _ in range(3):     # overshoot guaranteed → snap to target, go idle
            sim.tick(1.0)
        assert sim.x == pytest.approx(10.0)
        assert sim.target is None
        assert sim.state == "idle"
        # Adapter ground truth tracks the sim.
        assert sim.adapter.get_position() == pytest.approx((10.0, 0.0))

    def test_unknown_robot_uri_does_not_crash(self, server):
        srv, _ = server
        response = srv._handle(OrchestrationMessage(
            task="go to (5, 5)", robot_uri="mock://nonexistent",
        ))
        # Planning still succeeds; there's simply no sim robot to move.
        assert response.metadata["status"] in ("planned", "error")

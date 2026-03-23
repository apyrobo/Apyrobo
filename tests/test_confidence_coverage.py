"""
Comprehensive tests for apyrobo/safety/confidence.py — targeting missing coverage lines.

Covers:
- RiskFactor (LOW/MED/HIGH repr)
- ConfidenceReport (all props, to_dict, __repr__)
- LowConfidenceError
- ConfidenceEstimator.assess with a mock robot
- Test with/without lidar/camera sensors
- Missing capabilities, high complexity graphs
- check_speed with speed exceeded
- check_historical_success with mock state_store
- gate() raising LowConfidenceError, gate() passing
- _check_world_state with mock world_state
- _required_capabilities
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import (
    Capability, CapabilityType, RobotCapability, SensorInfo, SensorType,
)
from apyrobo.safety.confidence import (
    ConfidenceEstimator,
    ConfidenceReport,
    LowConfidenceError,
    RiskFactor,
)
from apyrobo.skills.executor import SkillGraph
from apyrobo.skills.skill import Skill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_skill(skill_id: str, cap: CapabilityType = CapabilityType.NAVIGATE,
               parameters: dict | None = None) -> Skill:
    return Skill(
        skill_id=skill_id,
        name=skill_id,
        description="test skill",
        required_capability=cap,
        parameters=parameters or {},
    )


def make_graph(*skills: Skill) -> SkillGraph:
    g = SkillGraph()
    for skill in skills:
        g.add_skill(skill)
    return g


def make_robot_with_caps(
    caps: list[CapabilityType],
    sensors: list[tuple[str, SensorType]] | None = None,
    max_speed: float | None = 1.0,
) -> Robot:
    """Build a mock robot that returns the given capabilities."""
    robot = MagicMock(spec=Robot)
    robot.robot_id = "test_robot"
    capability_list = [Capability(capability_type=c, name=c.value) for c in caps]
    sensor_list = [
        SensorInfo(sensor_id=sid, sensor_type=stype)
        for sid, stype in (sensors or [])
    ]
    robot.capabilities.return_value = RobotCapability(
        robot_id="test_robot",
        name="TestRobot",
        capabilities=capability_list,
        sensors=sensor_list,
        max_speed=max_speed,
    )
    return robot


def full_robot() -> Robot:
    """Real mock://tb4 robot."""
    return Robot.discover("mock://tb4")


# ---------------------------------------------------------------------------
# RiskFactor
# ---------------------------------------------------------------------------

class TestRiskFactor:
    def test_repr_low(self):
        r = RiskFactor(name="test", severity=0.1)
        assert "LOW" in repr(r)
        assert "test" in repr(r)

    def test_repr_medium(self):
        r = RiskFactor(name="test", severity=0.5)
        assert "MED" in repr(r)

    def test_repr_high(self):
        r = RiskFactor(name="test", severity=0.8)
        assert "HIGH" in repr(r)

    def test_repr_boundary_low_to_med(self):
        # severity == 0.3 is MED
        r = RiskFactor(name="boundary", severity=0.3)
        assert "MED" in repr(r)

    def test_repr_boundary_med_to_high(self):
        # severity == 0.7 is HIGH
        r = RiskFactor(name="boundary", severity=0.7)
        assert "HIGH" in repr(r)

    def test_description_stored(self):
        r = RiskFactor(name="x", severity=0.0, description="some risk")
        assert r.description == "some risk"


# ---------------------------------------------------------------------------
# ConfidenceReport
# ---------------------------------------------------------------------------

class TestConfidenceReport:
    def _make_report(self, confidence: float = 0.8, can_proceed: bool = True) -> ConfidenceReport:
        return ConfidenceReport(
            confidence=confidence,
            risks=[RiskFactor("r1", 0.2, "minor")],
            can_proceed=can_proceed,
            details={"foo": "bar"},
        )

    def test_confidence_stored(self):
        rpt = self._make_report(0.75)
        assert rpt.confidence == 0.75

    def test_can_proceed_stored(self):
        rpt = self._make_report(can_proceed=False)
        assert rpt.can_proceed is False

    def test_details_stored(self):
        rpt = self._make_report()
        assert rpt.details == {"foo": "bar"}

    def test_details_defaults_to_empty_dict(self):
        rpt = ConfidenceReport(confidence=0.9, risks=[], can_proceed=True)
        assert rpt.details == {}

    def test_risk_level_low(self):
        rpt = self._make_report(confidence=0.9)
        assert rpt.risk_level == "low"

    def test_risk_level_medium(self):
        rpt = self._make_report(confidence=0.6)
        assert rpt.risk_level == "medium"

    def test_risk_level_high(self):
        rpt = self._make_report(confidence=0.3)
        assert rpt.risk_level == "high"

    def test_risk_level_boundary_08(self):
        rpt = self._make_report(confidence=0.8)
        assert rpt.risk_level == "low"

    def test_risk_level_boundary_05(self):
        rpt = self._make_report(confidence=0.5)
        assert rpt.risk_level == "medium"

    def test_to_dict_structure(self):
        rpt = self._make_report(0.8)
        d = rpt.to_dict()
        assert "confidence" in d
        assert "risk_level" in d
        assert "can_proceed" in d
        assert "risks" in d
        assert "details" in d

    def test_to_dict_risks_list(self):
        rpt = self._make_report()
        d = rpt.to_dict()
        assert len(d["risks"]) == 1
        assert d["risks"][0]["name"] == "r1"

    def test_to_dict_confidence_rounded(self):
        rpt = ConfidenceReport(confidence=0.666666, risks=[], can_proceed=True)
        d = rpt.to_dict()
        assert d["confidence"] == 0.667

    def test_repr_contains_fields(self):
        rpt = self._make_report(0.8)
        r = repr(rpt)
        assert "ConfidenceReport" in r
        assert "proceed=" in r
        assert "risks=" in r


# ---------------------------------------------------------------------------
# LowConfidenceError
# ---------------------------------------------------------------------------

class TestLowConfidenceError:
    def test_error_stores_report(self):
        rpt = ConfidenceReport(confidence=0.2, risks=[], can_proceed=False)
        err = LowConfidenceError(rpt)
        assert err.report is rpt

    def test_error_message_contains_confidence(self):
        rpt = ConfidenceReport(confidence=0.25, risks=[], can_proceed=False)
        err = LowConfidenceError(rpt)
        assert "25%" in str(err)

    def test_error_message_contains_risk_names(self):
        rpt = ConfidenceReport(
            confidence=0.1,
            risks=[RiskFactor("missing_cap", 0.9)],
            can_proceed=False,
        )
        err = LowConfidenceError(rpt)
        assert "missing_cap" in str(err)


# ---------------------------------------------------------------------------
# ConfidenceEstimator — assess()
# ---------------------------------------------------------------------------

class TestConfidenceEstimatorAssess:
    def test_assess_returns_report(self):
        robot = full_robot()
        graph = make_graph(make_skill("nav1", CapabilityType.NAVIGATE))
        est = ConfidenceEstimator()
        report = est.assess(graph, robot)
        assert isinstance(report, ConfidenceReport)

    def test_assess_navigate_with_lidar_no_risks(self):
        robot = make_robot_with_caps(
            caps=[CapabilityType.NAVIGATE],
            sensors=[("lidar0", SensorType.LIDAR)],
        )
        graph = make_graph(make_skill("nav1", CapabilityType.NAVIGATE))
        est = ConfidenceEstimator()
        report = est.assess(graph, robot)
        assert report.can_proceed is True
        # No capability risk and lidar present
        sensor_risks = [r for r in report.risks if "lidar" in r.name]
        assert len(sensor_risks) == 0

    def test_assess_navigate_without_lidar_adds_risk(self):
        robot = make_robot_with_caps(
            caps=[CapabilityType.NAVIGATE],
            sensors=[("imu0", SensorType.IMU)],
        )
        graph = make_graph(make_skill("nav1", CapabilityType.NAVIGATE))
        est = ConfidenceEstimator()
        report = est.assess(graph, robot)
        lidar_risks = [r for r in report.risks if "lidar" in r.name]
        assert len(lidar_risks) == 1

    def test_assess_pick_without_camera_adds_risk(self):
        robot = make_robot_with_caps(
            caps=[CapabilityType.PICK],
            sensors=[("lidar0", SensorType.LIDAR)],
        )
        graph = make_graph(make_skill("pick1", CapabilityType.PICK))
        est = ConfidenceEstimator()
        report = est.assess(graph, robot)
        camera_risks = [r for r in report.risks if "camera" in r.name]
        assert len(camera_risks) == 1

    def test_assess_place_without_camera_adds_risk(self):
        robot = make_robot_with_caps(
            caps=[CapabilityType.PLACE],
            sensors=[],
        )
        graph = make_graph(make_skill("place1", CapabilityType.PLACE))
        est = ConfidenceEstimator()
        report = est.assess(graph, robot)
        # No camera risk AND no sensors risk
        camera_risks = [r for r in report.risks if "camera" in r.name]
        assert len(camera_risks) == 1

    def test_assess_no_sensors_adds_no_sensors_risk(self):
        robot = make_robot_with_caps(caps=[CapabilityType.NAVIGATE], sensors=[])
        graph = make_graph(make_skill("nav1", CapabilityType.NAVIGATE))
        est = ConfidenceEstimator()
        report = est.assess(graph, robot)
        no_sensor_risks = [r for r in report.risks if r.name == "no_sensors"]
        assert len(no_sensor_risks) == 1

    def test_assess_missing_capability_adds_risk(self):
        robot = make_robot_with_caps(caps=[], sensors=[])
        graph = make_graph(make_skill("nav1", CapabilityType.NAVIGATE))
        est = ConfidenceEstimator()
        report = est.assess(graph, robot)
        cap_risks = [r for r in report.risks if "missing_capability" in r.name]
        assert len(cap_risks) == 1

    def test_assess_custom_capability_not_flagged(self):
        robot = make_robot_with_caps(caps=[], sensors=[("lidar0", SensorType.LIDAR)])
        graph = make_graph(make_skill("custom1", CapabilityType.CUSTOM))
        est = ConfidenceEstimator()
        report = est.assess(graph, robot)
        cap_risks = [r for r in report.risks if "missing_capability" in r.name]
        assert len(cap_risks) == 0

    def test_assess_high_complexity_adds_risk(self):
        robot = make_robot_with_caps(
            caps=[CapabilityType.NAVIGATE],
            sensors=[("lidar0", SensorType.LIDAR)],
        )
        skills = [make_skill(f"nav{i}", CapabilityType.NAVIGATE) for i in range(12)]
        graph = make_graph(*skills)
        est = ConfidenceEstimator()
        report = est.assess(graph, robot)
        complexity_risks = [r for r in report.risks if r.name == "high_complexity"]
        assert len(complexity_risks) == 1

    def test_assess_moderate_complexity_adds_risk(self):
        robot = make_robot_with_caps(
            caps=[CapabilityType.NAVIGATE],
            sensors=[("lidar0", SensorType.LIDAR)],
        )
        skills = [make_skill(f"nav{i}", CapabilityType.NAVIGATE) for i in range(7)]
        graph = make_graph(*skills)
        est = ConfidenceEstimator()
        report = est.assess(graph, robot)
        mod_risks = [r for r in report.risks if r.name == "moderate_complexity"]
        assert len(mod_risks) == 1

    def test_assess_details_populated(self):
        robot = full_robot()
        graph = make_graph(make_skill("nav1", CapabilityType.NAVIGATE))
        est = ConfidenceEstimator()
        report = est.assess(graph, robot)
        assert "skill_count" in report.details
        assert "capability_types_needed" in report.details
        assert "robot_capabilities" in report.details
        assert "sensor_count" in report.details

    def test_assess_score_clamped_to_zero_minimum(self):
        robot = make_robot_with_caps(caps=[], sensors=[])
        # Many missing capabilities to drive score negative
        skills = [
            make_skill("nav1", CapabilityType.NAVIGATE),
            make_skill("pick1", CapabilityType.PICK),
            make_skill("place1", CapabilityType.PLACE),
        ]
        graph = make_graph(*skills)
        est = ConfidenceEstimator()
        report = est.assess(graph, robot)
        assert report.confidence >= 0.0

    def test_assess_score_clamped_to_one_maximum(self):
        robot = make_robot_with_caps(
            caps=[CapabilityType.NAVIGATE],
            sensors=[("lidar0", SensorType.LIDAR)],
        )
        graph = make_graph(make_skill("nav1", CapabilityType.NAVIGATE))
        est = ConfidenceEstimator()
        report = est.assess(graph, robot)
        assert report.confidence <= 1.0


# ---------------------------------------------------------------------------
# ConfidenceEstimator — check_speed
# ---------------------------------------------------------------------------

class TestCheckSpeed:
    def test_speed_exceeded_adds_risk(self):
        robot = make_robot_with_caps(
            caps=[CapabilityType.NAVIGATE],
            sensors=[("lidar0", SensorType.LIDAR)],
            max_speed=0.5,
        )
        skill = make_skill("nav1", CapabilityType.NAVIGATE, parameters={"speed": 2.0})
        graph = make_graph(skill)
        est = ConfidenceEstimator()
        report = est.assess(graph, robot)
        speed_risks = [r for r in report.risks if r.name == "speed_exceeded"]
        assert len(speed_risks) == 1

    def test_speed_within_limits_no_risk(self):
        robot = make_robot_with_caps(
            caps=[CapabilityType.NAVIGATE],
            sensors=[("lidar0", SensorType.LIDAR)],
            max_speed=2.0,
        )
        skill = make_skill("nav1", CapabilityType.NAVIGATE, parameters={"speed": 0.5})
        graph = make_graph(skill)
        est = ConfidenceEstimator()
        report = est.assess(graph, robot)
        speed_risks = [r for r in report.risks if r.name == "speed_exceeded"]
        assert len(speed_risks) == 0

    def test_no_max_speed_no_risk(self):
        robot = make_robot_with_caps(
            caps=[CapabilityType.NAVIGATE],
            sensors=[("lidar0", SensorType.LIDAR)],
            max_speed=None,
        )
        skill = make_skill("nav1", CapabilityType.NAVIGATE, parameters={"speed": 99.0})
        graph = make_graph(skill)
        est = ConfidenceEstimator()
        report = est.assess(graph, robot)
        speed_risks = [r for r in report.risks if r.name == "speed_exceeded"]
        assert len(speed_risks) == 0


# ---------------------------------------------------------------------------
# ConfidenceEstimator — check_historical_success
# ---------------------------------------------------------------------------

class TestCheckHistoricalSuccess:
    def _make_tasks(self, n_completed: int, n_total: int):
        tasks = []
        for i in range(n_total):
            t = MagicMock()
            t.status = "completed" if i < n_completed else "failed"
            tasks.append(t)
        return tasks

    def test_no_state_store_no_impact(self):
        robot = make_robot_with_caps(
            caps=[CapabilityType.NAVIGATE],
            sensors=[("lidar0", SensorType.LIDAR)],
        )
        graph = make_graph(make_skill("nav1", CapabilityType.NAVIGATE))
        est = ConfidenceEstimator(state_store=None)
        report = est.assess(graph, robot)
        historical_risks = [r for r in report.risks if "historical" in r.name]
        assert len(historical_risks) == 0

    def test_few_tasks_not_enough_history(self):
        state_store = MagicMock()
        state_store.get_recent_tasks.return_value = self._make_tasks(2, 2)
        robot = make_robot_with_caps(
            caps=[CapabilityType.NAVIGATE],
            sensors=[("lidar0", SensorType.LIDAR)],
        )
        graph = make_graph(make_skill("nav1", CapabilityType.NAVIGATE))
        est = ConfidenceEstimator(state_store=state_store)
        report = est.assess(graph, robot)
        historical_risks = [r for r in report.risks if "historical" in r.name]
        assert len(historical_risks) == 0

    def test_low_success_rate_adds_risk(self):
        state_store = MagicMock()
        state_store.get_recent_tasks.return_value = self._make_tasks(1, 10)
        robot = make_robot_with_caps(
            caps=[CapabilityType.NAVIGATE],
            sensors=[("lidar0", SensorType.LIDAR)],
        )
        graph = make_graph(make_skill("nav1", CapabilityType.NAVIGATE))
        est = ConfidenceEstimator(state_store=state_store)
        report = est.assess(graph, robot)
        low_risks = [r for r in report.risks if r.name == "low_historical_success"]
        assert len(low_risks) == 1

    def test_moderate_success_rate_adds_risk(self):
        state_store = MagicMock()
        state_store.get_recent_tasks.return_value = self._make_tasks(6, 10)
        robot = make_robot_with_caps(
            caps=[CapabilityType.NAVIGATE],
            sensors=[("lidar0", SensorType.LIDAR)],
        )
        graph = make_graph(make_skill("nav1", CapabilityType.NAVIGATE))
        est = ConfidenceEstimator(state_store=state_store)
        report = est.assess(graph, robot)
        mod_risks = [r for r in report.risks if r.name == "moderate_historical_success"]
        assert len(mod_risks) == 1

    def test_high_success_rate_no_risk(self):
        state_store = MagicMock()
        state_store.get_recent_tasks.return_value = self._make_tasks(9, 10)
        robot = make_robot_with_caps(
            caps=[CapabilityType.NAVIGATE],
            sensors=[("lidar0", SensorType.LIDAR)],
        )
        graph = make_graph(make_skill("nav1", CapabilityType.NAVIGATE))
        est = ConfidenceEstimator(state_store=state_store)
        report = est.assess(graph, robot)
        historical_risks = [r for r in report.risks if "historical" in r.name]
        assert len(historical_risks) == 0

    def test_state_store_exception_is_swallowed(self):
        state_store = MagicMock()
        state_store.get_recent_tasks.side_effect = RuntimeError("DB down")
        robot = make_robot_with_caps(
            caps=[CapabilityType.NAVIGATE],
            sensors=[("lidar0", SensorType.LIDAR)],
        )
        graph = make_graph(make_skill("nav1", CapabilityType.NAVIGATE))
        est = ConfidenceEstimator(state_store=state_store)
        # Should not raise
        report = est.assess(graph, robot)
        assert report is not None


# ---------------------------------------------------------------------------
# ConfidenceEstimator — _check_world_state
# ---------------------------------------------------------------------------

class TestCheckWorldState:
    def test_no_world_state_skipped(self):
        robot = make_robot_with_caps(
            caps=[CapabilityType.NAVIGATE],
            sensors=[("lidar0", SensorType.LIDAR)],
        )
        graph = make_graph(make_skill("nav1", CapabilityType.NAVIGATE))
        est = ConfidenceEstimator(world_state=None)
        report = est.assess(graph, robot)
        world_risks = [r for r in report.risks if "obstacle" in r.name or "crowded" in r.name]
        assert len(world_risks) == 0

    def test_many_obstacles_adds_crowded_risk(self):
        ws = MagicMock()
        ws.obstacles_within.return_value = [1, 2, 3, 4, 5]  # > 3
        robot = make_robot_with_caps(
            caps=[CapabilityType.NAVIGATE],
            sensors=[("lidar0", SensorType.LIDAR)],
        )
        graph = make_graph(make_skill("nav1", CapabilityType.NAVIGATE))
        est = ConfidenceEstimator(world_state=ws)
        report = est.assess(graph, robot)
        crowded_risks = [r for r in report.risks if r.name == "crowded_environment"]
        assert len(crowded_risks) == 1

    def test_few_obstacles_adds_nearby_risk(self):
        ws = MagicMock()
        ws.obstacles_within.return_value = [1, 2]  # <= 3 but > 0
        robot = make_robot_with_caps(
            caps=[CapabilityType.NAVIGATE],
            sensors=[("lidar0", SensorType.LIDAR)],
        )
        graph = make_graph(make_skill("nav1", CapabilityType.NAVIGATE))
        est = ConfidenceEstimator(world_state=ws)
        report = est.assess(graph, robot)
        nearby_risks = [r for r in report.risks if r.name == "obstacles_nearby"]
        assert len(nearby_risks) == 1

    def test_no_obstacles_no_world_risk(self):
        ws = MagicMock()
        ws.obstacles_within.return_value = []
        robot = make_robot_with_caps(
            caps=[CapabilityType.NAVIGATE],
            sensors=[("lidar0", SensorType.LIDAR)],
        )
        graph = make_graph(make_skill("nav1", CapabilityType.NAVIGATE))
        est = ConfidenceEstimator(world_state=ws)
        report = est.assess(graph, robot)
        world_risks = [r for r in report.risks
                       if r.name in ("crowded_environment", "obstacles_nearby")]
        assert len(world_risks) == 0


# ---------------------------------------------------------------------------
# ConfidenceEstimator — gate()
# ---------------------------------------------------------------------------

class TestGate:
    def test_gate_passes_when_confidence_sufficient(self):
        robot = make_robot_with_caps(
            caps=[CapabilityType.NAVIGATE],
            sensors=[("lidar0", SensorType.LIDAR)],
        )
        graph = make_graph(make_skill("nav1", CapabilityType.NAVIGATE))
        est = ConfidenceEstimator(block_below=0.4)
        # Should not raise
        report = est.gate(graph, robot)
        assert isinstance(report, ConfidenceReport)

    def test_gate_raises_when_confidence_too_low(self):
        robot = make_robot_with_caps(caps=[], sensors=[])
        # Lots of missing stuff to drive score very low
        skills = [
            make_skill("nav1", CapabilityType.NAVIGATE),
            make_skill("pick1", CapabilityType.PICK),
            make_skill("place1", CapabilityType.PLACE),
        ]
        graph = make_graph(*skills)
        est = ConfidenceEstimator(block_below=0.99)
        with pytest.raises(LowConfidenceError) as exc_info:
            est.gate(graph, robot)
        assert exc_info.value.report is not None

    def test_gate_uses_proceed_threshold_when_block_below_none(self):
        robot = make_robot_with_caps(caps=[], sensors=[])
        skills = [
            make_skill("nav1", CapabilityType.NAVIGATE),
            make_skill("pick1", CapabilityType.PICK),
            make_skill("place1", CapabilityType.PLACE),
        ]
        graph = make_graph(*skills)
        # block_below=None means use PROCEED_THRESHOLD=0.4
        est = ConfidenceEstimator(block_below=None)
        # Score will be very low — should raise
        with pytest.raises(LowConfidenceError):
            est.gate(graph, robot)


# ---------------------------------------------------------------------------
# ConfidenceEstimator — _required_capabilities (static method)
# ---------------------------------------------------------------------------

class TestRequiredCapabilities:
    def test_single_skill_capability(self):
        graph = make_graph(make_skill("nav1", CapabilityType.NAVIGATE))
        caps = ConfidenceEstimator._required_capabilities(graph)
        assert CapabilityType.NAVIGATE in caps

    def test_multiple_skill_capabilities(self):
        graph = make_graph(
            make_skill("nav1", CapabilityType.NAVIGATE),
            make_skill("pick1", CapabilityType.PICK),
        )
        caps = ConfidenceEstimator._required_capabilities(graph)
        assert CapabilityType.NAVIGATE in caps
        assert CapabilityType.PICK in caps

    def test_empty_graph_empty_capabilities(self):
        graph = SkillGraph()
        caps = ConfidenceEstimator._required_capabilities(graph)
        assert len(caps) == 0

"""Tests for multi-agent coordination bus (v7.0.0)."""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from apyrobo.coordination import TaskBus, TaskRequest, TaskResult, MultiAgentCoordinator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill_graph(*skill_names):
    """Return a mock SkillGraph with the given skill names."""
    graph = MagicMock()
    skills = []
    for name in skill_names:
        s = MagicMock()
        s.name = name
        s.skill_id = name
        skills.append(s)
    graph.get_execution_order.return_value = skills
    return graph


def _make_agent(*skill_names):
    """Return a mock Agent that returns a fixed skill plan."""
    agent = MagicMock()
    agent.plan.return_value = _make_skill_graph(*skill_names)
    return agent


def _make_robot(cap_types: list[str] | None = None):
    """Return a mock Robot advertising the given capability types."""
    robot = MagicMock()
    robot.__repr__ = lambda self: "mock://robot"
    if cap_types is not None:
        caps = []
        for c in cap_types:
            cap = MagicMock()
            cap.capability_type.value = c
            caps.append(cap)
        robot.capabilities.return_value = MagicMock(capabilities=caps)
    else:
        robot.capabilities.side_effect = AttributeError("no caps")
    return robot


# ---------------------------------------------------------------------------
# TaskRequest / TaskResult
# ---------------------------------------------------------------------------

class TestTaskRequest:
    def test_auto_request_id(self):
        r1 = TaskRequest(task="a")
        r2 = TaskRequest(task="b")
        assert r1.request_id != r2.request_id

    def test_to_dict(self):
        r = TaskRequest(task="navigate", robot_uri="mock://spot",
                        required_capability="NAVIGATE")
        d = r.to_dict()
        assert d["task"] == "navigate"
        assert d["required_capability"] == "NAVIGATE"
        assert "request_id" in d

    def test_default_fields(self):
        r = TaskRequest(task="x")
        assert r.robot_uri == ""
        assert r.required_capability == ""
        assert r.requester_id == ""


class TestTaskResult:
    def test_to_dict(self):
        r = TaskResult(
            request_id="abc",
            agent_id="arm-bot",
            robot_uri="mock://ur5",
            success=True,
            skills_planned=["pick", "place"],
            elapsed_ms=42.5,
        )
        d = r.to_dict()
        assert d["success"] is True
        assert "pick" in d["skills_planned"]
        assert d["elapsed_ms"] == 42.5

    def test_failure_result(self):
        r = TaskResult(
            request_id="x",
            agent_id="",
            robot_uri="",
            success=False,
            error="no agent",
        )
        assert not r.success
        assert r.error == "no agent"


# ---------------------------------------------------------------------------
# TaskBus — registration
# ---------------------------------------------------------------------------

class TestTaskBusRegistration:
    def test_register_increments_count(self):
        bus = TaskBus()
        coord = MagicMock()
        coord.agent_id = "bot-1"
        coord.capabilities = ["NAVIGATE"]
        bus.register(coord)
        assert bus.agent_count == 1

    def test_unregister_decrements_count(self):
        bus = TaskBus()
        coord = MagicMock()
        coord.agent_id = "bot-1"
        coord.capabilities = []
        bus.register(coord)
        bus.unregister("bot-1")
        assert bus.agent_count == 0

    def test_unregister_unknown_agent_noop(self):
        bus = TaskBus()
        bus.unregister("nonexistent")  # should not raise

    def test_agent_ids(self):
        bus = TaskBus()
        for i in range(3):
            coord = MagicMock()
            coord.agent_id = f"bot-{i}"
            coord.capabilities = []
            bus.register(coord)
        ids = bus.agent_ids()
        assert set(ids) == {"bot-0", "bot-1", "bot-2"}

    def test_agent_capabilities_map(self):
        bus = TaskBus()
        coord = MagicMock()
        coord.agent_id = "arm"
        coord.capabilities = ["PICK", "PLACE"]
        bus.register(coord)
        caps = bus.agent_capabilities()
        assert caps["arm"] == ["PICK", "PLACE"]


# ---------------------------------------------------------------------------
# TaskBus._select_agent
# ---------------------------------------------------------------------------

class TestTaskBusSelectAgent:
    def _bus_with_agents(self, agent_caps: dict[str, list[str]]) -> TaskBus:
        bus = TaskBus()
        for aid, caps in agent_caps.items():
            coord = MagicMock()
            coord.agent_id = aid
            coord.capabilities = caps
            coord.queue_depth = 0
            bus.register(coord)
        return bus

    def test_select_by_capability(self):
        bus = self._bus_with_agents({
            "nav-bot": ["NAVIGATE"],
            "arm-bot": ["PICK", "PLACE"],
        })
        agent = bus._select_agent("PICK")
        assert agent.agent_id == "arm-bot"

    def test_select_fallback_when_no_match(self):
        bus = self._bus_with_agents({"nav-bot": ["NAVIGATE"]})
        agent = bus._select_agent("PICK")
        # Falls back to any agent
        assert agent is not None

    def test_select_none_when_no_agents(self):
        bus = TaskBus()
        assert bus._select_agent("NAVIGATE") is None

    def test_select_least_loaded(self):
        bus = TaskBus()
        for i, depth in enumerate([5, 1, 3]):
            coord = MagicMock()
            coord.agent_id = f"bot-{i}"
            coord.capabilities = ["NAVIGATE"]
            coord.queue_depth = depth
            bus.register(coord)
        agent = bus._select_agent("NAVIGATE")
        assert agent.agent_id == "bot-1"  # depth=1

    def test_case_insensitive_capability_match(self):
        bus = self._bus_with_agents({"arm": ["pick", "place"]})
        agent = bus._select_agent("PICK")
        assert agent is not None
        assert agent.agent_id == "arm"


# ---------------------------------------------------------------------------
# TaskBus.dispatch — no agents registered
# ---------------------------------------------------------------------------

class TestTaskBusDispatchNoAgents:
    def test_returns_failure_when_no_agents(self):
        bus = TaskBus(timeout=1.0)
        result = bus.dispatch("navigate to dock")
        assert result.success is False
        assert "No agent" in result.error


# ---------------------------------------------------------------------------
# MultiAgentCoordinator — lifecycle
# ---------------------------------------------------------------------------

class TestMultiAgentCoordinatorLifecycle:
    def test_start_registers_on_bus(self):
        bus = TaskBus()
        agent = _make_agent("navigate")
        robot = _make_robot(["NAVIGATE"])
        coord = MultiAgentCoordinator(agent, robot, bus, agent_id="nav-bot",
                                      capabilities=["NAVIGATE"])
        coord.start()
        try:
            assert bus.agent_count == 1
            assert "nav-bot" in bus.agent_ids()
        finally:
            coord.stop()

    def test_stop_unregisters_from_bus(self):
        bus = TaskBus()
        agent = _make_agent("navigate")
        robot = _make_robot(["NAVIGATE"])
        coord = MultiAgentCoordinator(agent, robot, bus, agent_id="nav-bot",
                                      capabilities=["NAVIGATE"])
        coord.start()
        coord.stop()
        assert bus.agent_count == 0

    def test_auto_generated_agent_id(self):
        bus = TaskBus()
        agent = _make_agent()
        robot = _make_robot([])
        coord = MultiAgentCoordinator(agent, robot, bus, capabilities=[])
        assert coord.agent_id.startswith("agent-")

    def test_capabilities_auto_discovered_from_robot(self):
        bus = TaskBus()
        agent = _make_agent()
        robot = _make_robot(["NAVIGATE", "SCAN"])
        coord = MultiAgentCoordinator(agent, robot, bus)
        assert "NAVIGATE" in coord.capabilities
        assert "SCAN" in coord.capabilities

    def test_capabilities_override_when_explicit(self):
        bus = TaskBus()
        agent = _make_agent()
        robot = _make_robot(["NAVIGATE"])
        coord = MultiAgentCoordinator(agent, robot, bus, capabilities=["PICK"])
        assert coord.capabilities == ["PICK"]

    def test_capabilities_empty_on_robot_error(self):
        bus = TaskBus()
        agent = _make_agent()
        robot = MagicMock()
        robot.capabilities.side_effect = RuntimeError("no caps")
        coord = MultiAgentCoordinator(agent, robot, bus)
        assert coord.capabilities == []


# ---------------------------------------------------------------------------
# Full integration — dispatch + coordinator
# ---------------------------------------------------------------------------

class TestMultiAgentIntegration:
    def _setup(self, caps: list[str], *skill_names):
        bus = TaskBus(timeout=5.0)
        agent = _make_agent(*skill_names)
        robot = _make_robot(caps)
        coord = MultiAgentCoordinator(agent, robot, bus,
                                      agent_id="test-bot", capabilities=caps)
        coord.start()
        return bus, coord

    def test_dispatch_succeeds(self):
        bus, coord = self._setup(["NAVIGATE"], "move_forward", "dock")
        try:
            result = bus.dispatch("navigate to dock", required_capability="NAVIGATE")
            assert result.success is True
            assert "move_forward" in result.skills_planned
            assert "dock" in result.skills_planned
            assert result.agent_id == "test-bot"
        finally:
            coord.stop()

    def test_dispatch_routes_by_capability(self):
        bus = TaskBus(timeout=5.0)
        nav_agent = _make_agent("move")
        nav_robot = _make_robot(["NAVIGATE"])
        arm_agent = _make_agent("pick", "place")
        arm_robot = _make_robot(["PICK", "PLACE"])

        nav_coord = MultiAgentCoordinator(nav_agent, nav_robot, bus,
                                          agent_id="nav-bot", capabilities=["NAVIGATE"])
        arm_coord = MultiAgentCoordinator(arm_agent, arm_robot, bus,
                                          agent_id="arm-bot", capabilities=["PICK", "PLACE"])
        nav_coord.start()
        arm_coord.start()
        try:
            result = bus.dispatch("pick up the cup", required_capability="PICK")
            assert result.success is True
            assert result.agent_id == "arm-bot"
            assert "pick" in result.skills_planned
        finally:
            nav_coord.stop()
            arm_coord.stop()

    def test_dispatch_returns_failure_on_planning_error(self):
        bus = TaskBus(timeout=5.0)
        agent = MagicMock()
        agent.plan.side_effect = RuntimeError("LLM unavailable")
        robot = _make_robot([])
        coord = MultiAgentCoordinator(agent, robot, bus, agent_id="err-bot", capabilities=[])
        coord.start()
        try:
            result = bus.dispatch("do something")
            assert result.success is False
            assert "LLM unavailable" in result.error
        finally:
            coord.stop()

    def test_dispatch_timeout(self):
        bus = TaskBus(timeout=0.1)
        # Agent that never responds (blocks forever)
        agent = MagicMock()
        done_event = threading.Event()

        def _slow_plan(task, robot):
            done_event.wait(timeout=5)
            return _make_skill_graph()

        agent.plan.side_effect = _slow_plan
        robot = _make_robot([])
        coord = MultiAgentCoordinator(agent, robot, bus, agent_id="slow-bot", capabilities=[])
        coord.start()
        try:
            result = bus.dispatch("do something slowly")
            assert result.success is False
            assert "Timed out" in result.error
        finally:
            done_event.set()
            coord.stop()


# ---------------------------------------------------------------------------
# TaskBus.broadcast
# ---------------------------------------------------------------------------

class TestTaskBusBroadcast:
    def test_broadcast_empty_bus(self):
        bus = TaskBus()
        results = bus.broadcast("hello")
        assert results == []

    def test_broadcast_reaches_all_agents(self):
        bus = TaskBus(timeout=5.0)
        coords = []
        for i in range(3):
            agent = _make_agent(f"skill_{i}")
            robot = _make_robot([])
            coord = MultiAgentCoordinator(agent, robot, bus,
                                          agent_id=f"bot-{i}", capabilities=[])
            coord.start()
            coords.append(coord)

        try:
            results = bus.broadcast("status check")
            assert len(results) == 3
            assert all(r.success for r in results)
        finally:
            for c in coords:
                c.stop()


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestCoordinationModuleExports:
    def test_importable(self):
        import apyrobo.coordination as coord_mod
        assert hasattr(coord_mod, "TaskBus")
        assert hasattr(coord_mod, "TaskRequest")
        assert hasattr(coord_mod, "TaskResult")
        assert hasattr(coord_mod, "MultiAgentCoordinator")

    def test_all_list(self):
        import apyrobo.coordination as coord_mod
        for name in ("TaskBus", "TaskRequest", "TaskResult", "MultiAgentCoordinator"):
            assert name in coord_mod.__all__

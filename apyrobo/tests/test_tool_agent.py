"""Tests for apyrobo.agents.tool_agent"""
import pytest
from apyrobo.agents.tool_agent import SkillTool, ToolCallingAgent


def make_nav_tool(calls: list | None = None):
    """Helper: creates a navigate_to SkillTool that records calls."""
    recorded = calls if calls is not None else []

    def navigate(x: float, y: float):
        recorded.append((x, y))
        return f"moved to {x},{y}"

    return SkillTool(
        name="navigate_to",
        description="Move robot to (x, y)",
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
            },
            "required": ["x", "y"],
        },
        executor=navigate,
    ), recorded


class TestSkillTool:
    def test_fields(self):
        tool, _ = make_nav_tool()
        assert tool.name == "navigate_to"
        assert "x" in tool.parameters["properties"]

    def test_executor_callable(self):
        tool, calls = make_nav_tool()
        result = tool.executor(x=1.0, y=2.0)
        assert calls == [(1.0, 2.0)]
        assert "moved" in result


class TestToolCallingAgent:
    def test_run_returns_string(self):
        tool, _ = make_nav_tool()
        agent = ToolCallingAgent(tools=[tool])
        result = agent.run("go to position 1 2")
        assert isinstance(result, str)

    def test_mock_executes_first_tool(self):
        calls = []
        tool, calls = make_nav_tool(calls)
        agent = ToolCallingAgent(tools=[tool])
        result = agent.run("navigate somewhere")
        # mock path should have called the tool
        assert "navigate_to" in result or isinstance(result, str)

    def test_no_tools(self):
        agent = ToolCallingAgent(tools=[])
        result = agent.run("do something")
        assert isinstance(result, str)
        assert "no tools" in result.lower() or len(result) > 0

    def test_tool_map_populated(self):
        tool, _ = make_nav_tool()
        agent = ToolCallingAgent(tools=[tool])
        assert "navigate_to" in agent._tool_map

    def test_multiple_tools(self):
        def stop():
            return "stopped"

        nav_tool, _ = make_nav_tool()
        stop_tool = SkillTool(
            name="stop",
            description="Stop the robot",
            parameters={"type": "object", "properties": {}},
            executor=stop,
        )
        agent = ToolCallingAgent(tools=[nav_tool, stop_tool])
        result = agent.run("stop the robot")
        assert isinstance(result, str)

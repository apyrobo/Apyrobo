"""ToolCallingAgent never runs a tool the model did not ask for.

Before the fix, the offline fallback (no litellm, or any error from the
LLM call) ran the first tool with zeroed arguments, so "stop the robot"
with a navigate_to tool first drove the robot to (0, 0).
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

from apyrobo.agents.tool_agent import SkillTool, ToolCallingAgent


def _tools(calls: list[Any]) -> list[SkillTool]:
    nav = SkillTool(
        name="navigate_to",
        description="Move robot to (x, y)",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["x", "y"],
        },
        executor=lambda x, y: calls.append(("navigate_to", x, y)) or "moved",
    )
    stop = SkillTool(
        name="stop",
        description="Stop the robot",
        parameters={"type": "object", "properties": {}},
        executor=lambda: calls.append(("stop",)) or "stopped",
    )
    return [nav, stop]


def _ns(**kw: Any) -> types.SimpleNamespace:
    return types.SimpleNamespace(**kw)


def _tool_call_response(name: str, args: dict[str, Any]) -> Any:
    tc = _ns(id="call_1", function=_ns(name=name, arguments=json.dumps(args)))
    msg = _ns(content="", tool_calls=[tc])
    return _ns(choices=[_ns(finish_reason="tool_calls", message=msg)])


def _final_response(text: str) -> Any:
    msg = _ns(content=text, tool_calls=None)
    return _ns(choices=[_ns(finish_reason="stop", message=msg)])


def _fake_litellm(monkeypatch: pytest.MonkeyPatch, replies: list[Any]) -> None:
    mod = types.ModuleType("litellm")

    def completion(**_kw: Any) -> Any:
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    mod.completion = completion  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", mod)


def test_without_litellm_no_tool_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "litellm", None)  # import raises ImportError
    calls: list[Any] = []
    result = ToolCallingAgent(_tools(calls)).run("stop the robot now")
    assert calls == []
    assert "no tool was run" in result
    assert "navigate_to" in result and "stop" in result


def test_without_litellm_and_no_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "litellm", None)
    assert "no tools" in ToolCallingAgent([]).run("anything").lower()


def test_llm_error_runs_no_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_litellm(monkeypatch, [RuntimeError("no API key")])
    calls: list[Any] = []
    result = ToolCallingAgent(_tools(calls)).run("stop the robot now")
    assert calls == []
    assert result.startswith("LLM call failed: no API key")


def test_llm_error_mid_run_does_not_add_a_guessed_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_litellm(
        monkeypatch,
        [_tool_call_response("stop", {}), RuntimeError("connection reset")],
    )
    calls: list[Any] = []
    result = ToolCallingAgent(_tools(calls)).run("stop the robot now")
    assert calls == [("stop",)]
    assert "connection reset" in result


def test_model_requested_tool_still_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_litellm(
        monkeypatch,
        [_tool_call_response("navigate_to", {"x": 3, "y": 5}), _final_response("done")],
    )
    calls: list[Any] = []
    result = ToolCallingAgent(_tools(calls)).run("go to 3, 5")
    assert calls == [("navigate_to", 3, 5)]
    assert result == "done"

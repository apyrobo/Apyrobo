"""
Tool-calling Agent — LLM that directly invokes skills via function-calling.

The agent converts a list of SkillTool definitions into LLM tool/function
specs, sends the task to the LLM, executes any tool calls the model
requests, and returns the final text answer.

Without litellm, or when the LLM call fails, no tool is run: the agent
never invokes a skill the model did not ask for.

Classes:
    SkillTool         — descriptor for one executable skill
    ToolCallingAgent  — agent that uses function-calling to run skills
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SkillTool:
    """A skill exposed to the LLM as a callable tool."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object
    executor: Callable[..., Any]  # sync function called with **params


def _tools_to_litellm(tools: list[SkillTool]) -> list[dict[str, Any]]:
    """Convert SkillTools to litellm-compatible tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _mock_run(task: str, tools: list[SkillTool]) -> str:
    """
    Offline fallback when litellm is not installed.

    Runs no tool: with no model to read the task, any tool call would be a
    guess (a navigate tool called with zeros drives the robot to the
    origin, even for "stop").
    """
    if not tools:
        return f"Mock: no tools available for task {task!r}"
    names = ", ".join(t.name for t in tools)
    return (
        f"Mock: litellm is not installed, so no tool was run for task {task!r} "
        f"(available: {names})"
    )


class ToolCallingAgent:
    """
    Agent that uses LLM function-calling to invoke skills.

    Usage:
        def move_robot(x: float, y: float):
            robot.move(x, y)
            return "moved"

        tool = SkillTool(
            name="navigate_to",
            description="Move robot to (x, y)",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
            executor=move_robot,
        )
        agent = ToolCallingAgent(tools=[tool])
        result = agent.run("go to position 3, 5")
    """

    def __init__(self, tools: list[SkillTool], model: str = "gpt-4o") -> None:
        self.tools = tools
        self.model = model
        self._tool_map: dict[str, SkillTool] = {t.name: t for t in tools}

    def run(self, task: str) -> str:
        """
        Execute a task, invoking tools as needed.

        Returns the final natural-language answer.
        """
        try:
            import litellm  # type: ignore[import]
        except ImportError as exc:
            logger.debug("litellm unavailable (%s), using mock", exc)
            return _mock_run(task, self.tools)
        try:
            return self._run_with_litellm(task, litellm)
        except Exception as exc:
            # Never fall back to the mock here: tools the model asked for
            # may already have run, and guessing another call is unsafe.
            logger.warning("LLM call failed (%s); no further tools run", exc)
            return f"LLM call failed: {exc}. No further tools were run."

    def _run_with_litellm(self, task: str, litellm: Any) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        tool_defs = _tools_to_litellm(self.tools)

        for _ in range(10):  # guard against infinite loops
            response = litellm.completion(
                model=self.model,
                messages=messages,
                tools=tool_defs,
                tool_choice="auto",
            )
            choice = response.choices[0]
            finish = choice.finish_reason
            msg = choice.message

            # Add assistant message to context
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": getattr(msg, "tool_calls", None),
            })

            if finish == "tool_calls" and msg.tool_calls:
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments or "{}")
                    tool = self._tool_map.get(fn_name)
                    if tool is None:
                        result = f"Unknown tool: {fn_name}"
                    else:
                        try:
                            result = str(tool.executor(**fn_args))
                        except Exception as e:
                            result = f"Error: {e}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": fn_name,
                        "content": result,
                    })
            else:
                # Final answer
                return msg.content or ""

        return "Tool-calling agent exceeded iteration limit."

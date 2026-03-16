"""
IN-01: ToolCallingProvider — tool_choice fallback tests.

Tests that ToolCallingProvider gracefully falls back when:
- The model raises BadRequestError on tool_choice
- The model returns text instead of tool calls
- Valid tool calls are extracted correctly
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from apyrobo.skills.agent import ToolCallingProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SKILLS: list[dict[str, Any]] = [
    {
        "skill_id": "navigate_to",
        "description": "Navigate to coordinates",
        "parameters": {"x": 0.0, "y": 0.0},
    },
    {
        "skill_id": "rotate",
        "description": "Rotate the robot",
        "parameters": {"angle_rad": 0.0},
    },
]

CAPABILITIES = ["navigation", "manipulation"]


def _make_tool_call(name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _make_response(tool_calls=None, content=""):
    msg = SimpleNamespace(tool_calls=tool_calls, content=content)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


# ===========================================================================
# Tests
# ===========================================================================


class TestToolCallingProvider:
    """IN-01: tool_choice fallback tests."""

    def test_valid_tool_calls_extracted(self) -> None:
        """Valid tool calls are extracted correctly from the response."""
        provider = ToolCallingProvider(model="gpt-4o")
        tc1 = _make_tool_call("navigate_to", json.dumps({"x": 3, "y": 4}))
        tc2 = _make_tool_call("rotate", json.dumps({"angle_rad": 1.57}))
        response = _make_response(tool_calls=[tc1, tc2])

        with patch("litellm.completion", return_value=response):
            plan = provider.plan("go to 3,4 then rotate", SAMPLE_SKILLS, CAPABILITIES)

        assert len(plan) == 2
        assert plan[0]["skill_id"] == "navigate_to"
        assert plan[0]["parameters"] == {"x": 3, "y": 4}
        assert plan[1]["skill_id"] == "rotate"
        assert plan[1]["parameters"] == {"angle_rad": 1.57}

    def test_bad_request_falls_back_to_text(self) -> None:
        """BadRequestError on tool_choice falls back to text parsing."""
        provider = ToolCallingProvider(model="ollama/llama3")

        # Simulate: first call raises BadRequestError, second returns text JSON
        import litellm.exceptions
        text_response = _make_response(
            content=json.dumps([
                {"skill_id": "navigate_to", "parameters": {"x": 1, "y": 2}},
            ])
        )
        mock_completion = MagicMock(
            side_effect=[
                litellm.exceptions.BadRequestError(
                    message="tool_choice not supported",
                    model="ollama/llama3",
                    llm_provider="ollama",
                ),
                text_response,
            ]
        )

        with patch("litellm.completion", mock_completion):
            plan = provider.plan("go to 1,2", SAMPLE_SKILLS, CAPABILITIES)

        assert len(plan) == 1
        assert plan[0]["skill_id"] == "navigate_to"
        assert plan[0]["parameters"] == {"x": 1, "y": 2}
        # Verify it was called twice (original + fallback)
        assert mock_completion.call_count == 2

    def test_empty_tool_calls_falls_back_to_text_content(self) -> None:
        """Empty tool_calls response falls back to text parsing."""
        provider = ToolCallingProvider(model="gpt-4o")
        response = _make_response(
            tool_calls=None,
            content=json.dumps([
                {"skill_id": "rotate", "parameters": {"angle_rad": 0.5}},
            ]),
        )

        with patch("litellm.completion", return_value=response):
            plan = provider.plan("rotate 0.5 rad", SAMPLE_SKILLS, CAPABILITIES)

        assert len(plan) == 1
        assert plan[0]["skill_id"] == "rotate"

    def test_empty_tool_calls_no_content_returns_empty(self) -> None:
        """Empty tool_calls and empty content returns empty list."""
        provider = ToolCallingProvider(model="gpt-4o")
        response = _make_response(tool_calls=None, content="")

        with patch("litellm.completion", return_value=response):
            plan = provider.plan("do something", SAMPLE_SKILLS, CAPABILITIES)

        assert plan == []

    def test_fallback_does_not_raise(self) -> None:
        """Fallback path returns parsed plan or empty list, never raises."""
        provider = ToolCallingProvider(model="ollama/llama3")

        import litellm.exceptions
        # BadRequest → then text response with unparseable content
        bad_text_response = _make_response(content="I don't understand the task.")
        mock_completion = MagicMock(
            side_effect=[
                litellm.exceptions.BadRequestError(
                    message="tool_choice not supported",
                    model="ollama/llama3",
                    llm_provider="ollama",
                ),
                bad_text_response,
            ]
        )

        with patch("litellm.completion", mock_completion):
            plan = provider.plan("do something weird", SAMPLE_SKILLS, CAPABILITIES)

        assert isinstance(plan, list)
        assert len(plan) == 0

    def test_parse_text_plan_with_markdown_code_block(self) -> None:
        """_parse_text_plan handles JSON embedded in markdown."""
        text = '```json\n[{"skill_id": "navigate_to", "parameters": {"x": 5, "y": 6}}]\n```'
        result = ToolCallingProvider._parse_text_plan(text)
        assert len(result) == 1
        assert result[0]["skill_id"] == "navigate_to"

    def test_parse_text_plan_with_invalid_json(self) -> None:
        """_parse_text_plan returns empty list on invalid JSON."""
        result = ToolCallingProvider._parse_text_plan("this is not json at all")
        assert result == []

    def test_malformed_tool_call_arguments(self) -> None:
        """Malformed JSON in tool call arguments yields empty params."""
        provider = ToolCallingProvider(model="gpt-4o")
        tc = _make_tool_call("navigate_to", "not-valid-json{{{")
        response = _make_response(tool_calls=[tc])

        with patch("litellm.completion", return_value=response):
            plan = provider.plan("go somewhere", SAMPLE_SKILLS, CAPABILITIES)

        assert len(plan) == 1
        assert plan[0]["skill_id"] == "navigate_to"
        assert plan[0]["parameters"] == {}

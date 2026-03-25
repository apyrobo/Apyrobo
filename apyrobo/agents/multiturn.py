"""
Multi-turn Agent — stateful conversation with clarification dialogue.

Maintains conversation history and sends full context to the LLM on each
turn, enabling robots to ask for clarification when a task is ambiguous.

Classes:
    ConversationMessage  — a single turn (role + content + timestamp)
    ConversationHistory  — ordered list of messages with token-aware truncation
    MultiTurnAgent       — agent that answers via LLM or MockLLM fallback
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Rough tokens-per-char estimate for context window truncation.
_CHARS_PER_TOKEN = 4


@dataclass
class ConversationMessage:
    """A single message in a multi-turn conversation."""

    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class ConversationHistory:
    """
    Ordered history of conversation messages.

    Supports token-budget truncation: get_context() drops the oldest
    non-system messages first until the total fits within max_tokens.
    """

    def __init__(self) -> None:
        self._messages: list[ConversationMessage] = []

    def add(self, msg: ConversationMessage) -> None:
        self._messages.append(msg)

    def get_context(self, max_tokens: int = 4096) -> list[dict[str, Any]]:
        """
        Return messages as LLM-compatible dicts, truncated to fit max_tokens.

        System messages are always kept; oldest user/assistant messages are
        dropped first when over budget.
        """
        system_msgs = [m for m in self._messages if m.role == "system"]
        other_msgs = [m for m in self._messages if m.role != "system"]

        # Start with system messages (always included)
        kept = list(system_msgs)
        budget = max_tokens - sum(len(m.content) // _CHARS_PER_TOKEN for m in kept)

        # Add other messages newest-first until budget exhausted
        for msg in reversed(other_msgs):
            cost = len(msg.content) // _CHARS_PER_TOKEN + 1
            if budget >= cost:
                kept.append(msg)
                budget -= cost
            else:
                break

        # Sort by timestamp to restore chronological order
        kept.sort(key=lambda m: m.timestamp)
        return [{"role": m.role, "content": m.content} for m in kept]

    def clear(self) -> None:
        self._messages.clear()

    def to_dict(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self._messages]

    def __len__(self) -> int:
        return len(self._messages)


def _mock_llm_reply(messages: list[dict[str, Any]], context: dict[str, Any] | None) -> str:
    """Deterministic mock LLM for use when litellm is not configured."""
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    ctx_note = f" [context keys: {list(context)}]" if context else ""
    return f"Mock reply to: {last_user!r}{ctx_note}"


class MultiTurnAgent:
    """
    Stateful agent that maintains conversation history across turns.

    Uses litellm if available and configured; falls back to MockLLM for
    offline/test use.

    Usage:
        agent = MultiTurnAgent(system_prompt="You are a robot assistant.")
        reply = agent.chat("navigate to the kitchen")
        follow = agent.chat("actually, go to the living room instead")
    """

    def __init__(self, model: str = "gpt-4o", system_prompt: str = "") -> None:
        self.model = model
        self.history = ConversationHistory()
        if system_prompt:
            self.history.add(
                ConversationMessage(role="system", content=system_prompt)
            )

    def chat(self, message: str, context: dict[str, Any] | None = None) -> str:
        """
        Send a message and return the assistant's reply.

        Args:
            message: User message text.
            context:  Optional dict merged into the user message metadata.

        Returns:
            Assistant reply string.
        """
        self.history.add(
            ConversationMessage(role="user", content=message, metadata=context or {})
        )

        messages = self.history.get_context(max_tokens=4096)
        reply = self._call_llm(messages, context)

        self.history.add(ConversationMessage(role="assistant", content=reply))
        return reply

    def _call_llm(
        self, messages: list[dict[str, Any]], context: dict[str, Any] | None
    ) -> str:
        try:
            import litellm  # type: ignore[import]
            response = litellm.completion(
                model=self.model,
                messages=messages,
                max_tokens=512,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.debug("litellm unavailable (%s), using mock LLM", exc)
            return _mock_llm_reply(messages, context)

    def reset(self) -> None:
        """Clear conversation history."""
        self.history.clear()

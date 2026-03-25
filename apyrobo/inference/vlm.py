"""
VLM (Vision-Language Model) Integration — camera-informed planning.

Provides adapters for routing image + text queries to vision-language models,
enabling robots to reason about their environment from camera feeds.

Classes:
    VLMAdapter       — abstract base
    LiteLLMVLMAdapter — real adapter via litellm (gpt-4o, claude-3, etc.)
    MockVLMAdapter   — deterministic responses for testing
"""

from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class VLMAdapter(ABC):
    """Abstract base class for vision-language model adapters."""

    @abstractmethod
    def describe_scene(self, image_data: bytes) -> str:
        """Return a natural-language description of the scene in the image."""

    @abstractmethod
    def answer_question(self, image_data: bytes, question: str) -> str:
        """Answer a question about the image."""


class LiteLLMVLMAdapter(VLMAdapter):
    """
    VLM adapter that routes requests through litellm.

    Supports any model with vision capability: gpt-4o, claude-3-opus,
    gemini-pro-vision, etc.
    """

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 512) -> None:
        self.model = model
        self.max_tokens = max_tokens

    def _encode_image(self, image_data: bytes) -> str:
        return base64.b64encode(image_data).decode("utf-8")

    def _call(self, image_data: bytes, prompt: str) -> str:
        try:
            import litellm  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("litellm is required for LiteLLMVLMAdapter") from exc

        b64 = self._encode_image(image_data)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        response = litellm.completion(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def describe_scene(self, image_data: bytes) -> str:
        return self._call(
            image_data,
            "Describe this scene in detail, focusing on objects, their positions, "
            "and any potential obstacles or points of interest for a robot.",
        )

    def answer_question(self, image_data: bytes, question: str) -> str:
        return self._call(image_data, question)


class MockVLMAdapter(VLMAdapter):
    """
    Deterministic VLM adapter for testing.

    Returns canned responses so tests never hit a real API.
    """

    def __init__(self, scene_description: str = "", answers: dict[str, str] | None = None) -> None:
        self._scene_description = scene_description or (
            "A room with a table on the left, a chair in the center, "
            "and a clear path to the right. No obstacles detected."
        )
        self._answers: dict[str, str] = answers or {}
        self.describe_calls: list[bytes] = []
        self.question_calls: list[tuple[bytes, str]] = []

    def describe_scene(self, image_data: bytes) -> str:
        self.describe_calls.append(image_data)
        logger.debug("MockVLMAdapter.describe_scene called (image=%d bytes)", len(image_data))
        return self._scene_description

    def answer_question(self, image_data: bytes, question: str) -> str:
        self.question_calls.append((image_data, question))
        logger.debug("MockVLMAdapter.answer_question: %r", question)
        return self._answers.get(question, f"Mock answer to: {question}")


class VLMRouter:
    """
    Thin router that picks a VLM adapter and answers vision queries.

    Integrates with the InferenceRouter by providing a route_vision() entry point.
    """

    def __init__(self, adapter: VLMAdapter | None = None) -> None:
        self._adapter: VLMAdapter = adapter or MockVLMAdapter()

    def set_adapter(self, adapter: VLMAdapter) -> None:
        self._adapter = adapter

    def route_vision(self, image_data: bytes, prompt: str) -> str:
        """
        Route a vision request to the configured VLM adapter.

        Args:
            image_data: Raw image bytes (JPEG, PNG, etc.).
            prompt: Natural-language question or instruction about the image.

        Returns:
            Model response string.
        """
        logger.debug("VLMRouter.route_vision prompt=%r image=%d bytes", prompt, len(image_data))
        return self._adapter.answer_question(image_data, prompt)

    def describe(self, image_data: bytes) -> str:
        """Return a scene description for the given image."""
        return self._adapter.describe_scene(image_data)

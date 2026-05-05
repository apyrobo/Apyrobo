"""VLM-based task verification — camera-informed skill outcome confirmation.

After a skill completes, TaskVerifier uses the robot's camera feed and a
vision-language model to confirm that the expected real-world outcome was
actually achieved (not just reported by the skill handler).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Outcome of a single skill verification check."""

    skill_id: str
    verified: bool
    confidence: float          # 0.0–1.0
    observation: str           # what the VLM saw
    suggested_action: str | None = None  # if not verified, what to try next


class TaskVerifier:
    """Uses a VLM to confirm skill completion from the robot's camera feed.

    Example::

        from apyrobo.inference.vlm import LiteLLMVLMAdapter
        verifier = TaskVerifier(LiteLLMVLMAdapter())
        result = verifier.verify_skill("navigate_to", robot=robot)
    """

    # Default expected outcomes, keyed by skill-name fragment.
    DEFAULT_OUTCOMES: dict[str, str] = {
        "navigate": "The robot has reached the target position",
        "pick":     "The robot is holding the target object",
        "place":    "The object has been placed at the target location",
        "inspect":  "The robot has a clear view of the target area",
    }

    def __init__(self, vlm_adapter: Any, confidence_threshold: float = 0.7) -> None:
        self._vlm = vlm_adapter
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_skill(
        self,
        skill_id: str,
        expected_outcome: str | None = None,
        image: bytes | None = None,
        robot: Any = None,
    ) -> VerificationResult:
        """Verify that a skill's expected outcome was achieved.

        Falls back gracefully when no image is available (returns
        ``confidence=0.0, verified=False``).
        """
        if expected_outcome is None:
            expected_outcome = self._default_expected_outcome(skill_id)

        # Try to capture image from the robot if not provided.
        if image is None and robot is not None:
            capture = getattr(robot, "capture_image", None)
            if callable(capture):
                try:
                    image = capture()
                except Exception as exc:
                    logger.debug("capture_image failed: %s", exc)

        if image is None:
            return VerificationResult(
                skill_id=skill_id,
                verified=False,
                confidence=0.0,
                observation="No image available for verification",
                suggested_action=None,
            )

        prompt = self.generate_verification_prompt(skill_id, expected_outcome)
        raw = self._vlm.answer_question(image, prompt)
        return self._parse_response(skill_id, raw)

    def generate_verification_prompt(self, skill_id: str, expected_outcome: str) -> str:
        """Build the VLM prompt for this skill type."""
        base = _strip_index_suffix(skill_id)
        return (
            f"You are verifying the outcome of a robot skill execution.\n\n"
            f"Skill executed: {base}\n"
            f"Expected outcome: {expected_outcome}\n\n"
            f"Examine the image carefully and respond with ONLY a JSON object "
            f"in this exact format:\n"
            f'{{"verified": true, "confidence": 0.95, '
            f'"observation": "brief description of what you see", '
            f'"suggested_action": null}}\n\n'
            f"Set verified=false and suggest a corrective action if the expected "
            f"outcome has NOT been achieved."
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _default_expected_outcome(self, skill_id: str) -> str:
        base = _strip_index_suffix(skill_id)
        for keyword, outcome in self.DEFAULT_OUTCOMES.items():
            if keyword in base:
                return outcome
        return f"The robot has successfully completed the '{base}' skill"

    def _parse_response(self, skill_id: str, response: str) -> VerificationResult:
        """Parse VLM response. Tries JSON first, falls back to keyword scan."""
        # Try JSON (possibly embedded in prose)
        try:
            m = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                return VerificationResult(
                    skill_id=skill_id,
                    verified=bool(data.get("verified", False)),
                    confidence=float(data.get("confidence", 0.5)),
                    observation=str(data.get("observation", response)),
                    suggested_action=data.get("suggested_action") or None,
                )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fallback: keyword-based
        lower = response.lower()
        verified = any(
            w in lower
            for w in ("yes", "confirmed", "success", "achieved", "completed", "true")
        )
        return VerificationResult(
            skill_id=skill_id,
            verified=verified,
            confidence=0.7 if verified else 0.3,
            observation=response,
            suggested_action=None,
        )


class MockTaskVerifier:
    """Deterministic verifier for tests.

    Configure per-skill results via :meth:`set_verified`; unconfigured
    skills default to *verified=True, confidence=0.9*.
    """

    def __init__(self) -> None:
        self._results: dict[str, VerificationResult] = {}
        self.calls: list[tuple[str, str | None, bytes | None]] = []

    def set_verified(
        self,
        skill_id: str,
        verified: bool,
        confidence: float = 0.9,
        observation: str = "mock observation",
        suggested_action: str | None = None,
    ) -> None:
        """Configure the verification result for a given skill_id."""
        self._results[skill_id] = VerificationResult(
            skill_id=skill_id,
            verified=verified,
            confidence=confidence,
            observation=observation,
            suggested_action=suggested_action,
        )

    def verify_skill(
        self,
        skill_id: str,
        expected_outcome: str | None = None,
        image: bytes | None = None,
        robot: Any = None,
    ) -> VerificationResult:
        self.calls.append((skill_id, expected_outcome, image))
        base = _strip_index_suffix(skill_id)
        result = self._results.get(base) or self._results.get(skill_id)
        if result is not None:
            return result
        return VerificationResult(
            skill_id=skill_id,
            verified=True,
            confidence=0.9,
            observation="Mock: assumed success",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_index_suffix(skill_id: str) -> str:
    """'navigate_to_0' → 'navigate_to'."""
    parts = skill_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return skill_id

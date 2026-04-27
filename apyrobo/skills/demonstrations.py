"""
Learning from demonstrations — record, store, replay, and learn skills from
human teleoperation sessions.

Components:
    - DemonstrationStep: single action snapshot within a demo
    - Demonstration: ordered sequence of steps with metadata
    - DemonstrationRecorder: capture live teleoperation
    - DemonstrationStore: persist/load demos as JSON files
    - DemonstrationReplayer: replay a stored demo via the skill executor
    - SkillLearner: extract reusable skill patterns from demo sets
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class DemonstrationStep:
    """A single action captured during a demonstration."""
    skill_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration_s: float = 0.0
    state_before: dict[str, Any] = field(default_factory=dict)
    state_after: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    notes: str = ""


@dataclass
class Demonstration:
    """An ordered sequence of steps recorded from a single demonstration session."""
    name: str
    steps: list[DemonstrationStep] = field(default_factory=list)
    demo_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    operator: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        if self.end_time is None:
            return 0.0
        return max(0.0, self.end_time - self.start_time)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def skill_sequence(self) -> list[str]:
        return [s.skill_name for s in self.steps]

    def successful_steps(self) -> list[DemonstrationStep]:
        return [s for s in self.steps if s.success]


# ---------------------------------------------------------------------------
# DemonstrationRecorder
# ---------------------------------------------------------------------------

class DemonstrationRecorder:
    """
    Captures a live teleoperation session as a Demonstration.

    Usage::

        recorder = DemonstrationRecorder()
        demo = recorder.start("pick_and_place")
        recorder.record_step("navigate_to", {"x": 1.0, "y": 2.0})
        recorder.record_step("pick_object", {"object_id": "box_1"})
        demo = recorder.stop()
    """

    def __init__(self) -> None:
        self._demo: Optional[Demonstration] = None

    @property
    def is_recording(self) -> bool:
        return self._demo is not None

    def start(self, name: str, operator: str = "", description: str = "") -> Demonstration:
        """Begin a new demonstration recording session."""
        if self._demo is not None:
            raise RuntimeError("Already recording. Call stop() first.")
        self._demo = Demonstration(
            name=name,
            operator=operator,
            description=description,
            start_time=time.time(),
        )
        logger.info("DemonstrationRecorder: started %r (%s)", name, self._demo.demo_id)
        return self._demo

    def record_step(
        self,
        skill_name: str,
        parameters: Optional[dict[str, Any]] = None,
        duration_s: float = 0.0,
        state_before: Optional[dict[str, Any]] = None,
        state_after: Optional[dict[str, Any]] = None,
        success: bool = True,
        notes: str = "",
    ) -> DemonstrationStep:
        """Append a step to the current demonstration."""
        if self._demo is None:
            raise RuntimeError("Not recording. Call start() first.")
        step = DemonstrationStep(
            skill_name=skill_name,
            parameters=parameters or {},
            timestamp=time.time(),
            duration_s=duration_s,
            state_before=state_before or {},
            state_after=state_after or {},
            success=success,
            notes=notes,
        )
        self._demo.steps.append(step)
        logger.debug("DemonstrationRecorder: step %r params=%s", skill_name, parameters)
        return step

    def stop(self) -> Demonstration:
        """Finish recording and return the completed Demonstration."""
        if self._demo is None:
            raise RuntimeError("Not recording. Call start() first.")
        self._demo.end_time = time.time()
        demo, self._demo = self._demo, None
        logger.info(
            "DemonstrationRecorder: stopped %r (%d steps, %.1fs)",
            demo.name, demo.step_count, demo.duration_s,
        )
        return demo

    def current_demo(self) -> Optional[Demonstration]:
        """Return the in-progress Demonstration, or None if not recording."""
        return self._demo


# ---------------------------------------------------------------------------
# DemonstrationStore
# ---------------------------------------------------------------------------

class DemonstrationStore:
    """
    JSON-backed persistence layer for Demonstration objects.

    Each demonstration is saved as ``<demo_id>.json`` under the store directory.
    """

    def __init__(self, directory: str = "demonstrations") -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, demo: Demonstration) -> Path:
        """Persist a demonstration to disk. Returns the file path."""
        path = self._dir / f"{demo.demo_id}.json"
        data = _demo_to_dict(demo)
        path.write_text(json.dumps(data, indent=2))
        logger.info("DemonstrationStore: saved %r to %s", demo.name, path)
        return path

    def delete(self, demo_id: str) -> bool:
        """Delete a demonstration. Returns True if it existed."""
        path = self._dir / f"{demo_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self, demo_id: str) -> Demonstration:
        """Load a demonstration by its ID."""
        path = self._dir / f"{demo_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Demonstration {demo_id!r} not found in {self._dir}"
            )
        return _demo_from_dict(json.loads(path.read_text()))

    def load_by_name(self, name: str) -> list[Demonstration]:
        """Load all demonstrations with the given name."""
        return [d for d in self.list_all() if d.name == name]

    def list_all(self) -> list[Demonstration]:
        """Return all stored demonstrations, sorted by start_time."""
        demos = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                demos.append(_demo_from_dict(json.loads(path.read_text())))
            except Exception as exc:
                logger.warning("DemonstrationStore: skipping %s: %s", path, exc)
        demos.sort(key=lambda d: d.start_time)
        return demos

    def list_ids(self) -> list[str]:
        return [p.stem for p in sorted(self._dir.glob("*.json"))]

    def count(self) -> int:
        return len(list(self._dir.glob("*.json")))


def _demo_to_dict(demo: Demonstration) -> dict[str, Any]:
    return asdict(demo)


def _demo_from_dict(data: dict[str, Any]) -> Demonstration:
    steps_raw = data.pop("steps", [])
    demo = Demonstration(**data)
    demo.steps = [DemonstrationStep(**s) for s in steps_raw]
    return demo


# ---------------------------------------------------------------------------
# DemonstrationReplayer
# ---------------------------------------------------------------------------

class DemonstrationReplayer:
    """
    Replays a Demonstration by dispatching each step through a skill executor.

    The executor must expose a ``dispatch(skill_name, **parameters)`` method.
    """

    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def replay(
        self,
        demo: Demonstration,
        skip_failed: bool = True,
        speed_factor: float = 1.0,
    ) -> list[dict[str, Any]]:
        """
        Execute each step in the demonstration.

        Returns a list of per-step records with keys ``skill``, ``result``, ``error``.
        """
        records: list[dict[str, Any]] = []
        for step in demo.steps:
            if skip_failed and not step.success:
                records.append({
                    "skill": step.skill_name, "result": None, "skipped": True
                })
                continue
            records.append(self.replay_step(step))
        return records

    def replay_step(self, step: DemonstrationStep) -> dict[str, Any]:
        """Replay a single step and return a record dict."""
        try:
            result = self._executor.dispatch(step.skill_name, **step.parameters)
            return {"skill": step.skill_name, "result": result, "error": None}
        except Exception as exc:
            logger.warning(
                "DemonstrationReplayer: step %r failed: %s", step.skill_name, exc
            )
            return {"skill": step.skill_name, "result": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# SkillLearner
# ---------------------------------------------------------------------------

@dataclass
class LearnedPattern:
    """A skill pattern extracted from a set of demonstrations."""
    skill_sequence: list[str]
    frequency: int
    avg_duration_s: float
    common_parameters: dict[str, Any]
    source_demo_ids: list[str]


class SkillLearner:
    """
    Extracts reusable skill patterns from a collection of demonstrations.

    Uses frequency analysis over skill sequences to surface the most common
    sub-sequences that could become standalone reusable skills.
    """

    def __init__(self, min_frequency: int = 2) -> None:
        self.min_frequency = min_frequency
        self._patterns: list[LearnedPattern] = []

    def learn(self, demos: list[Demonstration]) -> list[LearnedPattern]:
        """Analyse demos and return patterns meeting the frequency threshold."""
        if not demos:
            return []

        sequence_map: dict[tuple[str, ...], list[str]] = {}
        for demo in demos:
            seq = demo.skill_sequence
            seen: set[tuple[str, ...]] = set()
            for length in range(2, len(seq) + 1):
                for start in range(len(seq) - length + 1):
                    sub = tuple(seq[start : start + length])
                    if sub not in seen:
                        sequence_map.setdefault(sub, []).append(demo.demo_id)
                        seen.add(sub)

        patterns = []
        for seq, demo_ids in sequence_map.items():
            if len(demo_ids) < self.min_frequency:
                continue
            source_demos = [d for d in demos if d.demo_id in set(demo_ids)]
            patterns.append(LearnedPattern(
                skill_sequence=list(seq),
                frequency=len(demo_ids),
                avg_duration_s=self._avg_duration(seq, source_demos),
                common_parameters=self._common_params(seq, source_demos),
                source_demo_ids=sorted(set(demo_ids)),
            ))

        patterns.sort(key=lambda p: (-p.frequency, -len(p.skill_sequence)))
        self._patterns = patterns
        return patterns

    def most_common_sequence(self, demos: list[Demonstration]) -> list[str]:
        """Return the single most frequently repeated skill sequence."""
        patterns = self.learn(demos)
        return patterns[0].skill_sequence if patterns else []

    def extract_unique_skills(self, demos: list[Demonstration]) -> list[str]:
        """Return all distinct skill names observed across the demo set."""
        skills: set[str] = set()
        for demo in demos:
            skills.update(demo.skill_sequence)
        return sorted(skills)

    def summarise(self, demos: list[Demonstration]) -> dict[str, Any]:
        """Return high-level stats about the demo set."""
        if not demos:
            return {"demos": 0, "total_steps": 0, "unique_skills": 0, "patterns": 0}
        patterns = self.learn(demos)
        total_steps = sum(d.step_count for d in demos)
        return {
            "demos": len(demos),
            "total_steps": total_steps,
            "unique_skills": len(self.extract_unique_skills(demos)),
            "patterns": len(patterns),
            "avg_steps_per_demo": total_steps / len(demos),
        }

    def suggest_next_step(
        self, last_skill: str, demos: list[Demonstration]
    ) -> Optional[str]:
        """Suggest the most likely next skill given the last executed skill."""
        follow_counts: dict[str, int] = defaultdict(int)
        for demo in demos:
            skills = demo.skill_sequence
            for i, skill in enumerate(skills[:-1]):
                if skill == last_skill:
                    follow_counts[skills[i + 1]] += 1
        if not follow_counts:
            return None
        return max(follow_counts, key=lambda k: follow_counts[k])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _avg_duration(
        self, seq: tuple[str, ...], demos: list[Demonstration]
    ) -> float:
        durations: list[float] = []
        for demo in demos:
            demo_seq = demo.skill_sequence
            for start in range(len(demo_seq) - len(seq) + 1):
                if tuple(demo_seq[start : start + len(seq)]) == seq:
                    durations.append(
                        sum(demo.steps[start + i].duration_s for i in range(len(seq)))
                    )
        return sum(durations) / len(durations) if durations else 0.0

    def _common_params(
        self, seq: tuple[str, ...], demos: list[Demonstration]
    ) -> dict[str, Any]:
        param_sets: list[list[dict[str, Any]]] = []
        for demo in demos:
            demo_seq = demo.skill_sequence
            for start in range(len(demo_seq) - len(seq) + 1):
                if tuple(demo_seq[start : start + len(seq)]) == seq:
                    param_sets.append(
                        [demo.steps[start + i].parameters for i in range(len(seq))]
                    )
        if not param_sets:
            return {}
        common: dict[str, Any] = {}
        for step_idx, step_params in enumerate(param_sets[0]):
            for key, val in step_params.items():
                if all(
                    step_idx < len(ps) and ps[step_idx].get(key) == val
                    for ps in param_sets
                ):
                    common[f"{seq[step_idx]}.{key}"] = val
        return common

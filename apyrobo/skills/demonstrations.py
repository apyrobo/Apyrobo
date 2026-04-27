"""
Learning from demonstrations — record, store, replay, and learn from teleoperation.
"""
from __future__ import annotations

import json
import os
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class DemonstrationStep:
    timestamp: float
    skill_name: str
    params: dict
    state_before: dict
    state_after: dict
    success: bool = True


@dataclass
class Demonstration:
    demo_id: str
    robot_id: str
    steps: list[DemonstrationStep]
    metadata: dict
    recorded_at: datetime


class DemonstrationRecorder:
    def __init__(self) -> None:
        self._demo_id: Optional[str] = None
        self._robot_id: Optional[str] = None
        self._steps: list[DemonstrationStep] = []
        self._started_at: Optional[datetime] = None

    def start_recording(self, robot_id: str) -> str:
        self._demo_id = str(uuid.uuid4())
        self._robot_id = robot_id
        self._steps = []
        self._started_at = datetime.now(timezone.utc)
        return self._demo_id

    def record_step(
        self,
        skill_name: str,
        params: dict,
        state_before: dict,
        state_after: dict,
        success: bool = True,
    ) -> None:
        if not self.is_recording():
            raise RuntimeError("Not currently recording. Call start_recording first.")
        import time
        step = DemonstrationStep(
            timestamp=time.time(),
            skill_name=skill_name,
            params=params,
            state_before=state_before,
            state_after=state_after,
            success=success,
        )
        self._steps.append(step)

    def stop_recording(self) -> Demonstration:
        if not self.is_recording():
            raise RuntimeError("Not currently recording.")
        demo = Demonstration(
            demo_id=self._demo_id,
            robot_id=self._robot_id,
            steps=list(self._steps),
            metadata={},
            recorded_at=self._started_at,
        )
        self._demo_id = None
        self._robot_id = None
        self._steps = []
        self._started_at = None
        return demo

    def is_recording(self) -> bool:
        return self._demo_id is not None


class DemonstrationStore:
    def save(self, demo: Demonstration, path: str) -> None:
        data = {
            "demo_id": demo.demo_id,
            "robot_id": demo.robot_id,
            "recorded_at": demo.recorded_at.isoformat(),
            "metadata": demo.metadata,
            "steps": [asdict(s) for s in demo.steps],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> Demonstration:
        with open(path) as f:
            data = json.load(f)
        steps = [DemonstrationStep(**s) for s in data["steps"]]
        return Demonstration(
            demo_id=data["demo_id"],
            robot_id=data["robot_id"],
            steps=steps,
            metadata=data.get("metadata", {}),
            recorded_at=datetime.fromisoformat(data["recorded_at"]),
        )

    def list_demos(self, directory: str) -> list[str]:
        try:
            entries = os.listdir(directory)
        except FileNotFoundError:
            return []
        return sorted(
            os.path.join(directory, e) for e in entries if e.endswith(".json")
        )


class DemonstrationReplayer:
    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def replay(self, demo: Demonstration, speed: float = 1.0) -> list[dict]:
        results = []
        for step in demo.steps:
            result = self.replay_step(step)
            results.append(result)
        return results

    def replay_step(self, step: DemonstrationStep) -> dict:
        return self._executor.execute_skill(step.skill_name, step.params)


class SkillLearner:
    def __init__(self) -> None:
        self._training_demos: list[Demonstration] = []

    def learn_from_demonstrations(self, demos: list[Demonstration]) -> dict:
        self._training_demos = list(demos)

        skill_counts: dict[str, int] = defaultdict(int)
        skill_successes: dict[str, int] = defaultdict(int)
        sequences: list[list[str]] = []

        for demo in demos:
            seq = []
            for step in demo.steps:
                skill_counts[step.skill_name] += 1
                if step.success:
                    skill_successes[step.skill_name] += 1
                seq.append(step.skill_name)
            if seq:
                sequences.append(seq)

        success_rates = {
            skill: skill_successes[skill] / skill_counts[skill]
            for skill in skill_counts
        }

        # Find the most frequent bigrams across all demo sequences.
        bigram_counts: dict[tuple[str, str], int] = defaultdict(int)
        for seq in sequences:
            for a, b in zip(seq, seq[1:]):
                bigram_counts[(a, b)] += 1
        top_sequences = sorted(bigram_counts.items(), key=lambda x: -x[1])[:10]

        return {
            "skill_counts": dict(skill_counts),
            "success_rates": success_rates,
            "top_sequences": [
                {"from": a, "to": b, "count": c} for (a, b), c in top_sequences
            ],
            "demo_count": len(demos),
        }

    def extract_skill_template(
        self, skill_name: str, demos: list[Demonstration]
    ) -> dict:
        all_params: list[dict] = []
        for demo in demos:
            for step in demo.steps:
                if step.skill_name == skill_name:
                    all_params.append(step.params)

        if not all_params:
            return {}

        # Collect all numeric keys and average them; keep last value for non-numeric.
        aggregated: dict = {}
        numeric_sums: dict[str, float] = defaultdict(float)
        numeric_counts: dict[str, int] = defaultdict(int)

        for p in all_params:
            for k, v in p.items():
                if isinstance(v, (int, float)):
                    numeric_sums[k] += v
                    numeric_counts[k] += 1
                else:
                    aggregated[k] = v

        for k in numeric_sums:
            aggregated[k] = numeric_sums[k] / numeric_counts[k]

        return aggregated

    def suggest_next_step(
        self, current_state: dict, history: list[DemonstrationStep]
    ) -> Optional[str]:
        if not history or not self._training_demos:
            return None

        last_skill = history[-1].skill_name

        # Count which skills follow last_skill in training demos.
        follow_counts: dict[str, int] = defaultdict(int)
        for demo in self._training_demos:
            skills = [s.skill_name for s in demo.steps]
            for i, skill in enumerate(skills[:-1]):
                if skill == last_skill:
                    follow_counts[skills[i + 1]] += 1

        if not follow_counts:
            return None
        return max(follow_counts, key=lambda k: follow_counts[k])

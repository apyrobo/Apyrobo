"""apyrobo serve says why a task was not planned.

The README's serve example sent "navigate to dock", which the rule planner
refuses (no coordinates). The response said "planned" with no skills and the
reason went only to the log, so a front-end could not tell a refusal from a
plan. The reason now rides in ``metadata.reason``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from apyrobo.core.robot import Robot
from apyrobo.orchestration import (
    MockOrchestrationAdapter,
    OrchestrationMessage,
    OrchestrationServer,
)
from apyrobo.skills.agent import Agent

README = Path(__file__).resolve().parent.parent / "README.md"


def _respond(task: str, **server_kwargs) -> dict:
    robot = Robot.discover("mock://turtlebot4")
    server = OrchestrationServer(
        MockOrchestrationAdapter([]), Agent(provider="rule"),
        default_robot=robot, **server_kwargs,
    )
    reply = server._handle(OrchestrationMessage(task=task, robot_uri="mock://turtlebot4"))
    return reply.to_dict()


def test_refused_task_says_why() -> None:
    meta = _respond("navigate to dock")["metadata"]
    assert meta["status"] == "planned"
    assert meta["count"] == 0 and meta["skills"] == []
    assert "No target" in meta["reason"]
    assert "Agent(provider=" not in meta["reason"]
    assert "execution" not in meta


def test_planned_task_has_no_reason() -> None:
    meta = _respond("go to (2, 3)")["metadata"]
    assert meta["count"] == 1
    assert "reason" not in meta


def test_execute_mode_refusal_carries_reason_too() -> None:
    meta = _respond("navigate to dock", execute_tasks=True)["metadata"]
    assert meta["reason"] == meta["execution"]["error"]
    assert meta["execution"]["status"] == "failed"


def test_readme_serve_lines_match_the_server() -> None:
    text = README.read_text(encoding="utf-8")
    line_in = re.search(r"Each line in is a JSON task: `(\{.*?\})`", text).group(1)
    line_out = re.search(r"Each line out is a JSON response: `(\{.*?\})`\.", text).group(1)
    refused = '{"task": "navigate to dock", "robot_uri": "mock://turtlebot4"}'
    proc = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "serve",
         "--robot", "mock://turtlebot4", "--provider", "rule"],
        input=line_in + "\n" + refused + "\n",
        capture_output=True, text=True, timeout=60,
    )
    replies = [json.loads(x) for x in proc.stdout.splitlines() if x.strip()]
    assert len(replies) == 2, proc.stdout + proc.stderr
    assert replies[0] == json.loads(line_out)
    assert replies[1]["metadata"]["count"] == 0
    assert "No target" in replies[1]["metadata"]["reason"]

"""The LLM-to-rule fallback warning is logged once per agent and reason.

The CLI plans a task more than once with one Agent (plan, then the rule-plan
check, then execute), so on a bare install 'apyrobo execute ... --provider
llm' printed the same "litellm is required" warning two or three times.
"""

from __future__ import annotations

import logging

from apyrobo.core.robot import Robot
from apyrobo.skills.agent import Agent, AgentProvider

_MSG = "LLM provider failed"


class _FailingProvider(AgentProvider):
    def __init__(self, reasons: list[str]) -> None:
        self._reasons = reasons

    def plan(self, task, available_skills, capabilities, **kwargs):
        raise RuntimeError(self._reasons.pop(0) if len(self._reasons) > 1 else self._reasons[0])


def _agent(reasons: list[str]) -> Agent:
    agent = Agent(provider="rule")
    agent._provider = _FailingProvider(reasons)
    return agent


def _warnings(caplog) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records
        if r.levelno == logging.WARNING and _MSG in r.getMessage()
    ]


def test_same_reason_warns_once_across_plans(caplog):
    robot = Robot.discover("mock://turtlebot4")
    agent = _agent(["litellm is required"])
    with caplog.at_level(logging.DEBUG, logger="apyrobo.skills.agent"):
        for _ in range(3):
            graph = agent.plan("go to (1, 2)", robot)
            assert len(graph) == 1
    assert len(_warnings(caplog)) == 1
    debug = [
        r for r in caplog.records
        if r.levelno == logging.DEBUG and _MSG in r.getMessage()
    ]
    assert len(debug) == 2


def test_new_reason_warns_again(caplog):
    robot = Robot.discover("mock://turtlebot4")
    agent = _agent(["litellm is required", "rate limited"])
    with caplog.at_level(logging.WARNING, logger="apyrobo.skills.agent"):
        agent.plan("go to (1, 2)", robot)
        agent.plan("go to (1, 2)", robot)
        agent.plan("go to (1, 2)", robot)
    reasons = [r.getMessage() for r in _warnings(caplog)]
    assert len(reasons) == 2
    assert "litellm is required" in reasons[0]
    assert "rate limited" in reasons[1]


def test_each_agent_warns_for_itself(caplog):
    robot = Robot.discover("mock://turtlebot4")
    with caplog.at_level(logging.WARNING, logger="apyrobo.skills.agent"):
        _agent(["litellm is required"]).plan("go to (1, 2)", robot)
        _agent(["litellm is required"]).plan("go to (1, 2)", robot)
    assert len(_warnings(caplog)) == 2


def test_degraded_event_still_emitted_every_time(monkeypatch):
    import apyrobo.skills.agent as agent_mod

    events: list[str] = []
    monkeypatch.setattr(agent_mod, "emit_event", lambda name, **kw: events.append(name))
    robot = Robot.discover("mock://turtlebot4")
    agent = _agent(["litellm is required"])
    for _ in range(3):
        agent.plan("go to (1, 2)", robot)
    assert events.count("agent.degraded") == 3

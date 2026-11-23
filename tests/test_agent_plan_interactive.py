"""Agent.plan_interactive() hands the provider's question to answer_callback."""

from __future__ import annotations

from typing import Any

from apyrobo import Agent, Robot
from apyrobo.skills.agent import AgentProvider, ClarificationNeeded


class _AskOnce(AgentProvider):
    """Asks one question, then plans a drive to the answer's point."""

    def __init__(self) -> None:
        self.tasks: list[str] = []

    def plan(self, task: str, available_skills: list[dict[str, Any]],
             capabilities: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        self.tasks.append(task)
        if "(" not in task:
            raise ClarificationNeeded("Where should I go?", ["(3, 3)", "(1, 1)"])
        return [{"skill_id": "navigate_to", "parameters": {"x": 3.0, "y": 3.0}}]


def _agent(provider: AgentProvider) -> Agent:
    agent = Agent(provider="rule")
    agent._provider = provider
    agent.planned_by = provider
    return agent


def _targets(graph: Any) -> list[tuple[float, float]]:
    out = []
    for skill in graph.get_execution_order():
        params = graph.get_parameters(skill.skill_id)
        if "x" in params:
            out.append((float(params["x"]), float(params["y"])))
    return out


def test_callback_gets_the_question_and_the_answer_reaches_the_planner() -> None:
    provider = _AskOnce()
    agent = _agent(provider)
    robot = Robot.discover("mock://tb4")
    asked: list[tuple[str, list[str]]] = []

    def answer(question: str, options: list[str]) -> str:
        asked.append((question, options))
        return "(3, 3)"

    graph = agent.plan_interactive("go", robot, answer_callback=answer)

    assert asked == [("Where should I go?", ["(3, 3)", "(1, 1)"])]
    assert provider.tasks == ["go", "go ((3, 3))"]
    assert _targets(graph) == [(3.0, 3.0)]


def test_multi_turn_over_rule_planner_plans_the_answer() -> None:
    agent = Agent(provider="multi_turn")
    robot = Robot.discover("mock://tb4")
    asked: list[str] = []

    graph = agent.plan_interactive(
        "go", robot, answer_callback=lambda q, o: asked.append(q) or "go to (3, 3)",
    )

    assert asked == ["Where should I go?"]
    assert _targets(graph) == [(3.0, 3.0)]
    assert agent.last_refusal is None


def test_plain_plan_still_falls_back_on_a_question() -> None:
    agent = Agent(provider="multi_turn")
    robot = Robot.discover("mock://tb4")

    graph = agent.plan("go", robot)

    assert len(graph) == 0
    assert agent.last_refusal is not None


def test_no_callback_plans_as_plan_does() -> None:
    agent = Agent(provider="multi_turn")
    robot = Robot.discover("mock://tb4")

    graph = agent.plan_interactive("go", robot)

    assert len(graph) == 0
    assert agent.last_refusal is not None
    assert agent._pass_clarification is False


def test_a_second_question_falls_back_instead_of_raising() -> None:
    agent = Agent(provider="multi_turn")
    robot = Robot.discover("mock://tb4")
    asked: list[str] = []

    graph = agent.plan_interactive(
        "go", robot, answer_callback=lambda q, o: asked.append(q) or "there",
    )

    assert asked == ["Where should I go?"]
    assert len(graph) == 0
    assert agent.last_refusal is not None
    assert agent._pass_clarification is False


def test_multi_turn_over_rule_planner_refuses_a_placeholder_task() -> None:
    agent = Agent(provider="multi_turn")
    robot = Robot.discover("mock://tb4")

    graph = agent.plan("deliver the package to room 3", robot)

    assert len(graph) == 0
    assert agent.last_refusal is not None

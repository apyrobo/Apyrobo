#!/usr/bin/env python3
"""Smoke test for a bare `pip install apyrobo` (no extras).

Guards the core-split contract (docs/core_split_plan.md): the kernel —
adapters, capability model, skill graph, rule-based planning, safety —
must work with no optional dependency installed, and every LLM entry
point must degrade or fail with a `pip install 'apyrobo[llm]'` hint.

Run in CI against a clean venv. Safe to run in a dev venv too: the
degradation assertions are skipped when litellm happens to be installed.
"""
from __future__ import annotations

import sys
import time


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    results: list[bool] = []

    t0 = time.perf_counter()
    import apyrobo  # noqa: F401
    elapsed = time.perf_counter() - t0
    results.append(check("import apyrobo", True, f"{elapsed * 1000:.0f} ms"))
    results.append(check("import under 500 ms", elapsed < 0.5, f"{elapsed * 1000:.0f} ms"))

    from apyrobo.core.robot import Robot

    robot = Robot.discover("mock://smoke-test")
    caps = robot.capabilities()
    results.append(check("mock robot discovery", bool(caps.capabilities),
                         f"{len(caps.capabilities)} capabilities"))
    robot.move(x=1.0, y=1.0)
    robot.stop()
    results.append(check("move/stop commands", True))

    from apyrobo.skills.agent import Agent, LLMProvider, RuleBasedProvider

    agent = Agent(provider="rule")
    graph = agent.plan("navigate to the dock", robot)
    results.append(check("rule-based planning", len(graph.get_execution_order()) > 0,
                         f"{len(graph.get_execution_order())} skills planned"))

    try:
        import litellm  # noqa: F401
        litellm_installed = True
    except ImportError:
        litellm_installed = False

    if litellm_installed:
        print("  SKIP  degradation checks — litellm is installed in this environment")
    else:
        auto_agent = Agent(provider="auto")
        results.append(check(
            "provider='auto' degrades to rule-based",
            isinstance(auto_agent._provider, RuleBasedProvider),
        ))
        graph = auto_agent.plan("navigate to the dock", robot)
        results.append(check("degraded agent still plans", len(graph.get_execution_order()) > 0))

        try:
            LLMProvider().plan("navigate", [], [])
            results.append(check("LLMProvider raises without litellm", False))
        except RuntimeError as exc:
            results.append(check(
                "LLMProvider error carries install hint",
                "apyrobo[llm]" in str(exc),
                str(exc),
            ))

    if all(results):
        print(f"\nOK — {len(results)} checks passed")
        return 0
    print(f"\nFAILED — {results.count(False)} of {len(results)} checks failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())

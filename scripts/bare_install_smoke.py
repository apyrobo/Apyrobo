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

import subprocess
import sys
import time

IMPORT_BUDGET_S = 1.5
IMPORT_RUNS = 3

# Optional dependencies that must never load as a side effect of
# `import apyrobo`: each belongs to an extra and costs real start-up time.
HEAVY_OPTIONAL_MODULES = frozenset({
    "litellm", "openai", "anthropic", "torch", "transformers", "lerobot",
    "numpy", "mujoco", "rclpy", "paho", "fastapi", "uvicorn",
})

_IMPORT_TIMER = (
    "import time; t = time.perf_counter(); import apyrobo; "
    "print(time.perf_counter() - t)"
)


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    return ok


def _fresh_import_seconds() -> float:
    """Time `import apyrobo` in a new interpreter (no modules cached)."""
    out = subprocess.run(
        [sys.executable, "-c", _IMPORT_TIMER],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip().splitlines()[-1])


def main() -> int:
    results: list[bool] = []

    before = set(sys.modules)
    t0 = time.perf_counter()
    import apyrobo  # noqa: F401
    cold = time.perf_counter() - t0
    results.append(check("import apyrobo", True, f"cold first import {cold * 1000:.0f} ms"))

    # The import budget exists to catch a heavyweight dependency entering the
    # import path (litellm-class imports cost seconds on every start). The
    # cold first import above also pays for writing .pyc files and a cold
    # disk cache, which on a shared CI runner swings from ~300 ms to ~1.8 s
    # with no code change, so the budget is checked on the fastest of a few
    # fresh interpreters instead. A heavy dependency is slow in every one of
    # them, so it still fails. The roadmap's warm-import target (500 ms) is
    # tracked in docs/core_split_plan.md.
    best = min(_fresh_import_seconds() for _ in range(IMPORT_RUNS))
    results.append(check(
        f"import under {IMPORT_BUDGET_S * 1000:.0f} ms",
        best < IMPORT_BUDGET_S,
        f"fastest of {IMPORT_RUNS} fresh interpreters: {best * 1000:.0f} ms",
    ))

    pulled = sorted(
        {name.split(".")[0] for name in set(sys.modules) - before} & HEAVY_OPTIONAL_MODULES
    )
    results.append(check(
        "import pulls in no optional heavy dependency",
        not pulled,
        ", ".join(pulled) if pulled else "",
    ))

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

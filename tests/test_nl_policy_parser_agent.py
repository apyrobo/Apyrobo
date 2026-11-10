"""NLPolicyParser's LLM fallback must refuse an agent it can never call.

apyrobo.Agent has no complete() method, so before this check
NLPolicyParser(agent=Agent(...)) built fine and then silently stored every
rule the regex missed as "custom": the documented LLM fallback never ran.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

from apyrobo import Agent
from apyrobo.safety.nl_policy import NLPolicyParser

DOC = Path(__file__).resolve().parent.parent / "docs" / "concepts" / "nl_safety_policies.md"


def test_apyrobo_agent_is_refused_with_a_clear_error() -> None:
    with pytest.raises(TypeError, match=r"complete\(prompt\)"):
        NLPolicyParser(agent=Agent(provider="rule"))


@pytest.mark.parametrize(
    "agent", [object(), "gpt-4o", types.SimpleNamespace(complete="not callable")]
)
def test_objects_without_a_callable_complete_are_refused(agent: object) -> None:
    with pytest.raises(TypeError):
        NLPolicyParser(agent=agent)


def test_no_agent_still_works() -> None:
    p = NLPolicyParser().parse("only operate between 9am and 6pm")
    assert p.constraint_type == "custom"
    assert p.source == "regex"


def test_object_with_complete_is_used_for_unknown_rules() -> None:
    class Fake:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return '{"constraint_type": "time_window", "parameters": {"start": "09:00"}}'

    fake = Fake()
    parser = NLPolicyParser(agent=fake)
    assert parser.parse("never exceed 0.5 m/s").source == "regex"
    assert fake.prompts == []
    p = parser.parse("only operate between 9am and 6pm")
    assert p.constraint_type == "time_window"
    assert p.source == "llm"
    assert len(fake.prompts) == 1


def _doc_python_block() -> str:
    blocks = re.findall(r"```python\n(.*?)```", DOC.read_text(encoding="utf-8"), re.S)
    fallback = [b for b in blocks if "agent=" in b]
    assert len(fallback) == 1, "expected one LLM fallback example in the doc"
    return fallback[0]


def test_doc_example_runs_with_litellm_stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def completion(**kwargs: object) -> object:
        calls.append(kwargs)
        msg = types.SimpleNamespace(
            content='{"constraint_type": "time_window", '
            '"parameters": {"start": "09:00", "end": "18:00"}}'
        )
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion))
    ns: dict = {}
    exec(compile(_doc_python_block(), str(DOC), "exec"), ns)  # noqa: S102
    policy = ns["policy"]
    assert policy.constraint_type == "time_window"
    assert policy.source == "llm"
    assert calls and calls[0]["model"] == "gpt-4o-mini"


def test_doc_no_longer_passes_an_agent_instance() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "agent=Agent(" not in text
    assert 'provider="openai"' not in text

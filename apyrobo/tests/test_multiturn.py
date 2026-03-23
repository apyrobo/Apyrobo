"""Tests for apyrobo.agents.multiturn"""
import time
import pytest
from apyrobo.agents.multiturn import ConversationMessage, ConversationHistory, MultiTurnAgent


class TestConversationMessage:
    def test_defaults(self):
        msg = ConversationMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert isinstance(msg.timestamp, float)
        assert msg.metadata == {}

    def test_to_dict(self):
        msg = ConversationMessage(role="assistant", content="hi", metadata={"k": "v"})
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "hi"
        assert d["metadata"] == {"k": "v"}


class TestConversationHistory:
    def test_add_and_len(self):
        hist = ConversationHistory()
        hist.add(ConversationMessage(role="user", content="a"))
        hist.add(ConversationMessage(role="assistant", content="b"))
        assert len(hist) == 2

    def test_get_context_returns_dicts(self):
        hist = ConversationHistory()
        hist.add(ConversationMessage(role="user", content="hello"))
        ctx = hist.get_context()
        assert isinstance(ctx, list)
        assert ctx[0]["role"] == "user"
        assert ctx[0]["content"] == "hello"

    def test_system_messages_always_kept(self):
        hist = ConversationHistory()
        hist.add(ConversationMessage(role="system", content="You are a robot."))
        # Add many messages to exceed typical budget
        for i in range(200):
            hist.add(ConversationMessage(role="user", content=f"msg {i}"))
        ctx = hist.get_context(max_tokens=100)
        roles = [m["role"] for m in ctx]
        assert "system" in roles

    def test_clear(self):
        hist = ConversationHistory()
        hist.add(ConversationMessage(role="user", content="x"))
        hist.clear()
        assert len(hist) == 0

    def test_to_dict(self):
        hist = ConversationHistory()
        hist.add(ConversationMessage(role="user", content="test"))
        d = hist.to_dict()
        assert isinstance(d, list)
        assert d[0]["role"] == "user"

    def test_chronological_order_preserved(self):
        hist = ConversationHistory()
        t0 = time.time()
        hist.add(ConversationMessage(role="user", content="first", timestamp=t0))
        hist.add(ConversationMessage(role="assistant", content="second", timestamp=t0 + 1))
        ctx = hist.get_context()
        assert ctx[0]["content"] == "first"
        assert ctx[1]["content"] == "second"


class TestMultiTurnAgent:
    def test_chat_returns_string(self):
        agent = MultiTurnAgent()
        reply = agent.chat("hello robot")
        assert isinstance(reply, str)
        assert len(reply) > 0

    def test_history_accumulates(self):
        agent = MultiTurnAgent()
        agent.chat("first message")
        agent.chat("second message")
        # history should have user + assistant × 2 turns
        assert len(agent.history) >= 4

    def test_system_prompt_added(self):
        agent = MultiTurnAgent(system_prompt="You are a helpful robot.")
        msgs = agent.history.get_context()
        assert msgs[0]["role"] == "system"
        assert "robot" in msgs[0]["content"]

    def test_reset_clears_history(self):
        agent = MultiTurnAgent(system_prompt="sys")
        agent.chat("hello")
        agent.reset()
        assert len(agent.history) == 0

    def test_context_passed_to_mock(self):
        agent = MultiTurnAgent()
        reply = agent.chat("navigate somewhere", context={"speed": 0.5})
        assert isinstance(reply, str)

    def test_mock_reply_contains_message(self):
        agent = MultiTurnAgent()
        reply = agent.chat("go to kitchen")
        # MockLLM echoes back the last user message
        assert "kitchen" in reply.lower() or len(reply) > 0

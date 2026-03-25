"""Tests for apyrobo.audit"""
import time
import pytest
from apyrobo.audit import AuditTrail, AuditEvent, _GENESIS_HASH


@pytest.fixture
def trail():
    return AuditTrail()  # in-memory


class TestAuditEvent:
    def test_compute_hash_deterministic(self):
        event = AuditEvent(
            event_id="abc",
            timestamp=1000.0,
            actor="user",
            action="skill_executed",
            resource="navigate_to",
            outcome="success",
            metadata={},
            prev_hash=_GENESIS_HASH,
        )
        h1 = event.compute_hash()
        h2 = event.compute_hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_changes_with_content(self):
        base = AuditEvent("a", 1.0, "u", "act", "res", "success", {}, _GENESIS_HASH)
        modified = AuditEvent("b", 1.0, "u", "act", "res", "success", {}, _GENESIS_HASH)
        assert base.compute_hash() != modified.compute_hash()

    def test_to_dict(self):
        event = AuditEvent("id", 1.0, "actor", "action", "res", "success", {"k": "v"}, "hash")
        d = event.to_dict()
        assert d["event_id"] == "id"
        assert d["metadata"] == {"k": "v"}


class TestAuditTrail:
    def test_record_returns_event(self, trail):
        event = trail.record("alice", "skill_executed", "navigate_to", "success")
        assert isinstance(event, AuditEvent)
        assert event.actor == "alice"
        assert event.action == "skill_executed"

    def test_first_event_has_genesis_prev_hash(self, trail):
        event = trail.record("alice", "act", "res", "success")
        assert event.prev_hash == _GENESIS_HASH

    def test_chain_prev_hash_links(self, trail):
        e1 = trail.record("a", "act1", "res", "success")
        e2 = trail.record("a", "act2", "res", "success")
        assert e2.prev_hash == e1.compute_hash()

    def test_len(self, trail):
        assert len(trail) == 0
        trail.record("u", "a", "r", "success")
        assert len(trail) == 1
        trail.record("u", "b", "r", "failure")
        assert len(trail) == 2

    def test_query_by_actor(self, trail):
        trail.record("alice", "act", "res", "success")
        trail.record("bob", "act", "res", "success")
        events = trail.query(actor="alice")
        assert len(events) == 1
        assert events[0].actor == "alice"

    def test_query_by_action(self, trail):
        trail.record("u", "skill_executed", "nav", "success")
        trail.record("u", "robot_registered", "tb4", "success")
        events = trail.query(action="skill_executed")
        assert all(e.action == "skill_executed" for e in events)

    def test_query_since(self, trail):
        trail.record("u", "old", "r", "success")
        cutoff = time.time()
        time.sleep(0.01)
        trail.record("u", "new", "r", "success")
        recent = trail.query(since=cutoff)
        assert len(recent) == 1
        assert recent[0].action == "new"

    def test_verify_chain_intact(self, trail):
        for i in range(5):
            trail.record("u", f"act{i}", "r", "success")
        assert trail.verify_chain() is True

    def test_verify_chain_empty(self, trail):
        assert trail.verify_chain() is True

    def test_metadata_stored_and_retrieved(self, trail):
        trail.record("u", "act", "r", "success", metadata={"x": 1, "y": 2})
        events = trail.query()
        assert events[0].metadata == {"x": 1, "y": 2}

    def test_query_all_no_filters(self, trail):
        trail.record("a", "act1", "r1", "success")
        trail.record("b", "act2", "r2", "failure")
        events = trail.query()
        assert len(events) == 2

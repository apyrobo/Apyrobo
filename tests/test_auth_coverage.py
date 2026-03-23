"""
Comprehensive tests for apyrobo/auth.py — targeting missing coverage lines.

Covers:
- Role constants
- User creation with/without robots/api_key
- User.can_command for admin/viewer/operator with/without robots
- User.can_view
- User.__repr__
- AuditEntry.to_dict
- AuthManager.add_user, remove_user, get_user, authenticate, check_access
- AuthManager.guard(), audit_log, users property, __repr__
- GuardedRobot: capabilities (allowed/denied), move (allowed/denied), stop,
  robot_id property, __repr__
"""

from __future__ import annotations

import pytest

from apyrobo.auth import (
    AuthError,
    AuthManager,
    AuditEntry,
    GuardedRobot,
    Role,
    User,
)
from apyrobo.core.robot import Robot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_robot(name: str = "tb4") -> Robot:
    return Robot.discover(f"mock://{name}")


# ---------------------------------------------------------------------------
# Role constants
# ---------------------------------------------------------------------------

class TestRole:
    def test_admin_constant(self):
        assert Role.ADMIN == "admin"

    def test_operator_constant(self):
        assert Role.OPERATOR == "operator"

    def test_viewer_constant(self):
        assert Role.VIEWER == "viewer"


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------

class TestUser:
    def test_create_defaults(self):
        u = User(user_id="alice")
        assert u.user_id == "alice"
        assert u.role == Role.OPERATOR
        assert u.allowed_robots is None
        assert len(u.api_key) > 0
        assert u.created_at > 0

    def test_create_with_robots(self):
        u = User(user_id="bob", robots=["tb4", "arm1"])
        assert u.allowed_robots == {"tb4", "arm1"}

    def test_create_without_robots_is_none(self):
        u = User(user_id="carol", role=Role.ADMIN)
        assert u.allowed_robots is None

    def test_create_with_explicit_api_key(self):
        u = User(user_id="dave", api_key="my-secret-key")
        assert u.api_key == "my-secret-key"

    def test_create_auto_generates_api_key(self):
        u1 = User(user_id="u1")
        u2 = User(user_id="u2")
        assert u1.api_key != u2.api_key

    # can_command -----------------------------------------------------------

    def test_admin_can_command_any_robot(self):
        u = User(user_id="admin", role=Role.ADMIN)
        assert u.can_command("any_robot") is True
        assert u.can_command("surgical_bot") is True

    def test_viewer_cannot_command(self):
        u = User(user_id="viewer", role=Role.VIEWER)
        assert u.can_command("tb4") is False

    def test_operator_with_no_restriction_can_command_any(self):
        u = User(user_id="op", role=Role.OPERATOR, robots=None)
        assert u.can_command("tb4") is True
        assert u.can_command("anything") is True

    def test_operator_with_robots_can_command_allowed(self):
        u = User(user_id="op", role=Role.OPERATOR, robots=["tb4"])
        assert u.can_command("tb4") is True

    def test_operator_with_robots_cannot_command_unallowed(self):
        u = User(user_id="op", role=Role.OPERATOR, robots=["tb4"])
        assert u.can_command("surgical_bot") is False

    # can_view --------------------------------------------------------------

    def test_any_role_can_view(self):
        for role in (Role.ADMIN, Role.OPERATOR, Role.VIEWER):
            u = User(user_id="u", role=role)
            assert u.can_view("any_robot") is True

    # __repr__ --------------------------------------------------------------

    def test_repr_with_robots(self):
        u = User(user_id="nurse", role=Role.OPERATOR, robots=["delivery_bot"])
        r = repr(u)
        assert "nurse" in r
        assert "operator" in r

    def test_repr_without_robots_shows_all(self):
        u = User(user_id="admin_user", role=Role.ADMIN)
        r = repr(u)
        assert "admin_user" in r
        assert "all" in r


# ---------------------------------------------------------------------------
# AuditEntry
# ---------------------------------------------------------------------------

class TestAuditEntry:
    def test_to_dict_fields(self):
        entry = AuditEntry(
            user_id="u1", robot_id="tb4", action="move",
            allowed=True, reason="OK"
        )
        d = entry.to_dict()
        assert d["user_id"] == "u1"
        assert d["robot_id"] == "tb4"
        assert d["action"] == "move"
        assert d["allowed"] is True
        assert d["reason"] == "OK"
        assert "timestamp" in d

    def test_to_dict_denied(self):
        entry = AuditEntry(
            user_id="hacker", robot_id="arm1", action="move",
            allowed=False, reason="not authorized"
        )
        d = entry.to_dict()
        assert d["allowed"] is False
        assert d["reason"] == "not authorized"

    def test_to_dict_empty_reason(self):
        entry = AuditEntry("u", "r", "cmd", True)
        d = entry.to_dict()
        assert d["reason"] == ""


# ---------------------------------------------------------------------------
# AuthManager
# ---------------------------------------------------------------------------

class TestAuthManager:
    def test_add_user_returns_user(self):
        auth = AuthManager()
        user = auth.add_user("alice", role=Role.OPERATOR, robots=["tb4"])
        assert user.user_id == "alice"
        assert user.role == Role.OPERATOR

    def test_add_user_with_api_key(self):
        auth = AuthManager()
        user = auth.add_user("bob", api_key="fixed-key-123")
        assert user.api_key == "fixed-key-123"

    def test_remove_user_existing(self):
        auth = AuthManager()
        auth.add_user("alice")
        result = auth.remove_user("alice")
        assert result is True
        assert auth.get_user("alice") is None

    def test_remove_user_nonexistent(self):
        auth = AuthManager()
        result = auth.remove_user("ghost")
        assert result is False

    def test_remove_user_clears_api_key(self):
        auth = AuthManager()
        user = auth.add_user("alice", api_key="key-abc")
        auth.remove_user("alice")
        assert auth.authenticate("key-abc") is None

    def test_get_user_existing(self):
        auth = AuthManager()
        auth.add_user("alice")
        u = auth.get_user("alice")
        assert u is not None
        assert u.user_id == "alice"

    def test_get_user_nonexistent(self):
        auth = AuthManager()
        assert auth.get_user("nobody") is None

    def test_authenticate_valid_key(self):
        auth = AuthManager()
        user = auth.add_user("alice", api_key="correct-key")
        found = auth.authenticate("correct-key")
        assert found is not None
        assert found.user_id == "alice"

    def test_authenticate_invalid_key(self):
        auth = AuthManager()
        auth.add_user("alice", api_key="real-key")
        assert auth.authenticate("wrong-key") is None

    # check_access ----------------------------------------------------------

    def test_check_access_unknown_user_denied(self):
        auth = AuthManager()
        result = auth.check_access("ghost", "tb4", "move")
        assert result is False
        # Should have an audit entry for the denial
        assert len(auth.audit_log) == 1
        assert auth.audit_log[0].allowed is False
        assert "unknown user" in auth.audit_log[0].reason

    def test_check_access_view_action_always_allowed(self):
        auth = AuthManager()
        auth.add_user("viewer", role=Role.VIEWER)
        assert auth.check_access("viewer", "tb4", "view") is True

    def test_check_access_capabilities_action_allowed(self):
        auth = AuthManager()
        auth.add_user("viewer", role=Role.VIEWER)
        assert auth.check_access("viewer", "tb4", "capabilities") is True

    def test_check_access_command_allowed(self):
        auth = AuthManager()
        auth.add_user("op", role=Role.OPERATOR, robots=["tb4"])
        assert auth.check_access("op", "tb4", "move") is True

    def test_check_access_command_denied(self):
        auth = AuthManager()
        auth.add_user("op", role=Role.OPERATOR, robots=["tb4"])
        result = auth.check_access("op", "surgical_bot", "move")
        assert result is False

    def test_check_access_viewer_cannot_command(self):
        auth = AuthManager()
        auth.add_user("viewer", role=Role.VIEWER)
        assert auth.check_access("viewer", "tb4", "move") is False

    def test_audit_log_records_entries(self):
        auth = AuthManager()
        auth.add_user("alice", role=Role.ADMIN)
        auth.check_access("alice", "tb4", "move")
        auth.check_access("alice", "tb4", "view")
        log = auth.audit_log
        assert len(log) == 2

    def test_audit_log_returns_copy(self):
        auth = AuthManager()
        log1 = auth.audit_log
        log2 = auth.audit_log
        assert log1 is not log2

    def test_users_property(self):
        auth = AuthManager()
        auth.add_user("alice")
        auth.add_user("bob")
        users = auth.users
        assert "alice" in users
        assert "bob" in users

    def test_users_returns_copy(self):
        auth = AuthManager()
        auth.add_user("alice")
        u1 = auth.users
        u2 = auth.users
        assert u1 is not u2

    def test_repr(self):
        auth = AuthManager()
        auth.add_user("alice")
        r = repr(auth)
        assert "AuthManager" in r
        assert "users=1" in r

    def test_guard_returns_guarded_robot(self):
        auth = AuthManager()
        auth.add_user("op", role=Role.OPERATOR, robots=["tb4"])
        robot = make_robot("tb4")
        guarded = auth.guard(robot, "op")
        assert isinstance(guarded, GuardedRobot)


# ---------------------------------------------------------------------------
# GuardedRobot
# ---------------------------------------------------------------------------

class TestGuardedRobot:
    def _setup(self, role: str = Role.OPERATOR, robots: list | None = None):
        auth = AuthManager()
        auth.add_user("op", role=role, robots=robots)
        robot = make_robot("tb4")
        guarded = auth.guard(robot, "op")
        return auth, guarded

    def test_robot_id_property(self):
        _, guarded = self._setup()
        assert guarded.robot_id == "tb4"

    def test_repr(self):
        _, guarded = self._setup()
        r = repr(guarded)
        assert "GuardedRobot" in r
        assert "tb4" in r
        assert "op" in r

    # capabilities ----------------------------------------------------------

    def test_capabilities_allowed(self):
        _, guarded = self._setup(role=Role.ADMIN)
        caps = guarded.capabilities()
        assert caps is not None

    def test_capabilities_denied(self):
        # Create a viewer — can_view returns True for all, so let's use
        # a setup where check_access denies. We need a viewer who cannot view.
        # Actually can_view always returns True, so capabilities always works.
        # Test that it raises AuthError when user is unknown.
        auth = AuthManager()
        robot = make_robot("tb4")
        guarded = GuardedRobot(robot, auth, "unknown_user")
        with pytest.raises(AuthError):
            guarded.capabilities()

    # move ------------------------------------------------------------------

    def test_move_allowed_operator_on_assigned_robot(self):
        _, guarded = self._setup(role=Role.OPERATOR, robots=["tb4"])
        # Should not raise
        guarded.move(x=1.0, y=2.0)

    def test_move_allowed_admin(self):
        _, guarded = self._setup(role=Role.ADMIN)
        guarded.move(x=0.5, y=0.5)

    def test_move_denied_operator_on_unassigned_robot(self):
        auth = AuthManager()
        auth.add_user("op", role=Role.OPERATOR, robots=["other_bot"])
        robot = make_robot("tb4")
        guarded = auth.guard(robot, "op")
        with pytest.raises(AuthError):
            guarded.move(x=1.0, y=1.0)

    def test_move_denied_viewer(self):
        _, guarded = self._setup(role=Role.VIEWER)
        with pytest.raises(AuthError):
            guarded.move(x=1.0, y=1.0)

    def test_move_with_speed(self):
        _, guarded = self._setup(role=Role.ADMIN)
        guarded.move(x=1.0, y=1.0, speed=0.5)

    # stop ------------------------------------------------------------------

    def test_stop_always_allowed(self):
        # Stop is always audited but always succeeds (check_access doesn't gate it)
        _, guarded = self._setup(role=Role.VIEWER)
        # Should not raise
        guarded.stop()

    def test_stop_creates_audit_entry(self):
        auth, guarded = self._setup(role=Role.VIEWER)
        guarded.stop()
        log = auth.audit_log
        stop_entries = [e for e in log if e.action == "stop"]
        assert len(stop_entries) >= 1

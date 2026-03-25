"""Tests for RBAC additions in apyrobo.auth"""
import pytest
from apyrobo.auth import RBACManager, RBACRole, ROLE_PERMISSIONS


class TestRBACRole:
    def test_role_values(self):
        assert RBACRole.ADMIN.value == "admin"
        assert RBACRole.OPERATOR.value == "operator"
        assert RBACRole.VIEWER.value == "viewer"

    def test_role_permissions_admin_has_wildcard(self):
        assert "*" in ROLE_PERMISSIONS[RBACRole.ADMIN]

    def test_role_permissions_operator(self):
        ops = ROLE_PERMISSIONS[RBACRole.OPERATOR]
        assert "skill:execute" in ops
        assert "robot:register" in ops
        assert "task:submit" in ops
        assert "task:read" in ops

    def test_role_permissions_viewer(self):
        viewer = ROLE_PERMISSIONS[RBACRole.VIEWER]
        assert "task:read" in viewer
        assert "robot:list" in viewer
        assert "skill:execute" not in viewer


class TestRBACManager:
    def setup_method(self):
        self.rbac = RBACManager()

    def test_assign_and_get_role(self):
        self.rbac.assign_role("alice", RBACRole.OPERATOR)
        assert self.rbac.get_role("alice") == RBACRole.OPERATOR

    def test_get_role_unknown_user(self):
        assert self.rbac.get_role("nobody") is None

    def test_admin_has_all_permissions(self):
        self.rbac.assign_role("admin_user", RBACRole.ADMIN)
        assert self.rbac.check_permission("admin_user", "skill:execute")
        assert self.rbac.check_permission("admin_user", "robot:register")
        assert self.rbac.check_permission("admin_user", "anything:at:all")

    def test_operator_permissions(self):
        self.rbac.assign_role("op", RBACRole.OPERATOR)
        assert self.rbac.check_permission("op", "skill:execute") is True
        assert self.rbac.check_permission("op", "task:submit") is True
        assert self.rbac.check_permission("op", "robot:list") is False

    def test_viewer_permissions(self):
        self.rbac.assign_role("viewer", RBACRole.VIEWER)
        assert self.rbac.check_permission("viewer", "task:read") is True
        assert self.rbac.check_permission("viewer", "robot:list") is True
        assert self.rbac.check_permission("viewer", "skill:execute") is False

    def test_unknown_user_has_no_permission(self):
        assert self.rbac.check_permission("ghost", "task:read") is False

    def test_require_permission_passes(self):
        self.rbac.assign_role("op", RBACRole.OPERATOR)
        self.rbac.require_permission("op", "skill:execute")  # should not raise

    def test_require_permission_raises_for_viewer(self):
        self.rbac.assign_role("viewer", RBACRole.VIEWER)
        with pytest.raises(PermissionError):
            self.rbac.require_permission("viewer", "skill:execute")

    def test_require_permission_raises_for_unknown(self):
        with pytest.raises(PermissionError):
            self.rbac.require_permission("nobody", "task:read")

    def test_assign_role_overwrites_previous(self):
        self.rbac.assign_role("alice", RBACRole.VIEWER)
        self.rbac.assign_role("alice", RBACRole.ADMIN)
        assert self.rbac.get_role("alice") == RBACRole.ADMIN
        assert self.rbac.check_permission("alice", "skill:execute") is True

    def test_permissions_for_returns_set(self):
        self.rbac.assign_role("op", RBACRole.OPERATOR)
        perms = self.rbac.permissions_for("op")
        assert isinstance(perms, set)
        assert "skill:execute" in perms

    def test_permissions_for_unknown_user(self):
        assert self.rbac.permissions_for("ghost") == set()

"""
Authentication and Access Control — who can command which robots.

Enforces that only authorized users/agents can issue commands to specific
robots. Designed for environments where safety requires accountability
(hospitals, warehouses, defense).

Supports:
    - API key authentication
    - Role-based access (operator, viewer, admin)
    - Per-robot permissions
    - Audit trail of all commands

Usage:
    auth = AuthManager()
    auth.add_user("nurse_1", role="operator", robots=["delivery_bot"])
    auth.add_user("admin", role="admin")  # all robots

    # Wrap a robot with access control
    guarded = auth.guard(robot, user_id="nurse_1")
    guarded.move(x=1, y=2)  # OK — nurse_1 can operate delivery_bot

    guarded2 = auth.guard(surgical_robot, user_id="nurse_1")
    guarded2.move(x=1, y=2)  # DENIED — nurse_1 can't operate surgical_robot
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import time
from enum import Enum
from typing import Any

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import RobotCapability

logger = logging.getLogger(__name__)


class AuthError(Exception):
    """Raised when an operation is denied by access control."""
    pass


class Role:
    """A named role with permissions."""
    ADMIN = "admin"       # full access to all robots
    OPERATOR = "operator" # can command assigned robots
    VIEWER = "viewer"     # can query capabilities but not command


class User:
    """A registered user/agent with an API key and role."""

    def __init__(self, user_id: str, role: str = Role.OPERATOR,
                 robots: list[str] | None = None, api_key: str | None = None) -> None:
        self.user_id = user_id
        self.role = role
        self.allowed_robots = set(robots) if robots else None  # None = role-dependent
        self.api_key = api_key or secrets.token_hex(16)
        self.created_at = time.time()

    def can_command(self, robot_id: str) -> bool:
        """Check if this user can send commands to a robot."""
        if self.role == Role.ADMIN:
            return True
        if self.role == Role.VIEWER:
            return False
        # Operator: check robot whitelist
        if self.allowed_robots is None:
            return True  # no restrictions
        return robot_id in self.allowed_robots

    def can_view(self, robot_id: str) -> bool:
        """Check if this user can query a robot's capabilities."""
        return True  # all roles can view

    def __repr__(self) -> str:
        bots = list(self.allowed_robots) if self.allowed_robots else "all"
        return f"<User {self.user_id} role={self.role} robots={bots}>"


class AuditEntry:
    """A record of a command attempt."""

    def __init__(self, user_id: str, robot_id: str, action: str,
                 allowed: bool, reason: str = "") -> None:
        self.user_id = user_id
        self.robot_id = robot_id
        self.action = action
        self.allowed = allowed
        self.reason = reason
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "robot_id": self.robot_id,
            "action": self.action,
            "allowed": self.allowed,
            "reason": self.reason,
        }


class AuthManager:
    """
    Manages users, API keys, and access control.

    Usage:
        auth = AuthManager()
        auth.add_user("operator_1", role="operator", robots=["tb4"])
        guarded = auth.guard(robot, user_id="operator_1")
    """

    def __init__(self) -> None:
        self._users: dict[str, User] = {}
        self._api_keys: dict[str, str] = {}  # key -> user_id
        self._audit: list[AuditEntry] = []

    def add_user(self, user_id: str, role: str = Role.OPERATOR,
                 robots: list[str] | None = None, api_key: str | None = None) -> User:
        """Register a user."""
        user = User(user_id=user_id, role=role, robots=robots, api_key=api_key)
        self._users[user_id] = user
        self._api_keys[user.api_key] = user_id
        logger.info("Auth: registered user %s (role=%s)", user_id, role)
        return user

    def remove_user(self, user_id: str) -> bool:
        user = self._users.pop(user_id, None)
        if user:
            self._api_keys.pop(user.api_key, None)
            return True
        return False

    def get_user(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    def authenticate(self, api_key: str) -> User | None:
        """Look up a user by API key."""
        user_id = self._api_keys.get(api_key)
        if user_id:
            return self._users.get(user_id)
        return None

    def check_access(self, user_id: str, robot_id: str, action: str = "command") -> bool:
        """Check if a user can perform an action on a robot. Records audit."""
        user = self._users.get(user_id)
        if user is None:
            self._audit_log(user_id, robot_id, action, False, "unknown user")
            return False

        if action in ("view", "capabilities"):
            allowed = user.can_view(robot_id)
        else:
            allowed = user.can_command(robot_id)

        reason = "OK" if allowed else f"role={user.role}, not authorized for {robot_id}"
        self._audit_log(user_id, robot_id, action, allowed, reason)
        return allowed

    def _audit_log(self, user_id: str, robot_id: str, action: str,
                   allowed: bool, reason: str) -> None:
        entry = AuditEntry(user_id, robot_id, action, allowed, reason)
        self._audit.append(entry)
        if not allowed:
            logger.warning("AUTH DENIED: user=%s robot=%s action=%s reason=%s",
                           user_id, robot_id, action, reason)

    def guard(self, robot: Robot, user_id: str) -> "GuardedRobot":
        """Wrap a robot with access control for a specific user."""
        return GuardedRobot(robot, self, user_id)

    @property
    def audit_log(self) -> list[AuditEntry]:
        return list(self._audit)

    @property
    def users(self) -> dict[str, User]:
        return dict(self._users)

    def __repr__(self) -> str:
        return f"<AuthManager users={len(self._users)} audit_entries={len(self._audit)}>"


# ---------------------------------------------------------------------------
# RBAC — Role-Based Access Control
# ---------------------------------------------------------------------------


class RBACRole(Enum):
    """Roles for the RBAC system."""

    ADMIN = "admin"       # all permissions
    OPERATOR = "operator" # execute skills, register robots, submit/read tasks
    VIEWER = "viewer"     # read-only


ROLE_PERMISSIONS: dict[RBACRole, set[str]] = {
    RBACRole.ADMIN: {"*"},
    RBACRole.OPERATOR: {
        "skill:execute",
        "robot:register",
        "task:submit",
        "task:read",
    },
    RBACRole.VIEWER: {
        "task:read",
        "robot:list",
    },
}


class RBACManager:
    """
    Simple RBAC manager for granting and checking permissions.

    Usage:
        rbac = RBACManager()
        rbac.assign_role("alice", RBACRole.OPERATOR)
        rbac.check_permission("alice", "skill:execute")   # True
        rbac.require_permission("alice", "admin:delete")  # raises PermissionError
    """

    def __init__(self) -> None:
        self._roles: dict[str, RBACRole] = {}

    def assign_role(self, user_id: str, role: RBACRole) -> None:
        """Assign a role to a user (replaces any previous role)."""
        self._roles[user_id] = role
        logger.info("RBAC: assigned role %s to user %s", role.value, user_id)

    def get_role(self, user_id: str) -> RBACRole | None:
        """Return the role assigned to *user_id*, or None."""
        return self._roles.get(user_id)

    def check_permission(self, user_id: str, permission: str) -> bool:
        """Return True if *user_id* has *permission*."""
        role = self._roles.get(user_id)
        if role is None:
            return False
        allowed = ROLE_PERMISSIONS.get(role, set())
        return "*" in allowed or permission in allowed

    def require_permission(self, user_id: str, permission: str) -> None:
        """Raise PermissionError if *user_id* does not have *permission*."""
        if not self.check_permission(user_id, permission):
            role = self._roles.get(user_id)
            raise PermissionError(
                f"User {user_id!r} (role={role.value if role else 'none'}) "
                f"lacks permission {permission!r}"
            )

    def permissions_for(self, user_id: str) -> set[str]:
        """Return the full set of permissions granted to *user_id*."""
        role = self._roles.get(user_id)
        if role is None:
            return set()
        return set(ROLE_PERMISSIONS.get(role, set()))


class GuardedRobot:
    """
    A robot wrapped with access control.

    Has the same API as Robot but checks permissions before every command.
    """

    def __init__(self, robot: Robot, auth: AuthManager, user_id: str) -> None:
        self._robot = robot
        self._auth = auth
        self._user_id = user_id

    def capabilities(self, **kwargs: Any) -> RobotCapability:
        if not self._auth.check_access(self._user_id, self._robot.robot_id, "capabilities"):
            raise AuthError(f"User {self._user_id!r} cannot view robot {self._robot.robot_id!r}")
        return self._robot.capabilities(**kwargs)

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        if not self._auth.check_access(self._user_id, self._robot.robot_id, "move"):
            raise AuthError(f"User {self._user_id!r} cannot command robot {self._robot.robot_id!r}")
        self._robot.move(x=x, y=y, speed=speed)

    def stop(self) -> None:
        # Stop is always allowed for safety — but still audited
        self._auth.check_access(self._user_id, self._robot.robot_id, "stop")
        self._robot.stop()

    @property
    def robot_id(self) -> str:
        return self._robot.robot_id

    def __repr__(self) -> str:
        return f"<GuardedRobot {self._robot.robot_id} user={self._user_id}>"

"""Role-Based Access Control and Action Authorization.

Specification Section 2, Section 5.4 (EXT-D, Req ID: SEC-1, EXT-D):
Roles:
- clinician_reviewer: Clinical professional performing reviews and revisions.
- admin: Platform administrator with full review, user management, and evaluation rights.
- viewer: Read-only observer; strictly rejected from executing any review actions.
"""

from enum import Enum
import logging
from typing import FrozenSet, Set

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    CLINICIAN_REVIEWER = "clinician_reviewer"
    ADMIN = "admin"
    VIEWER = "viewer"


class UnauthorizedActionError(Exception):
    """Raised when an actor role lacks permission to perform a requested action."""
    def __init__(self, role: str, action: str):
        self.role = role
        self.action = action
        super().__init__(
            f"Role '{role}' is not authorized to execute action '{action}'. "
            f"Review actions require 'clinician_reviewer' or 'admin' role."
        )


# Explicit review actions defined in docs/SPEC.md Section 5.4
REVIEW_ACTIONS: FrozenSet[str] = frozenset({
    "REQUEST_CLARIFICATION",
    "REQUEST_EVIDENCE_REFRESH",
    "REQUEST_REVISION",
    "MARK_REVIEWED",
    "ADD_NOTE",
})

# Admin-only operations
ADMIN_ACTIONS: FrozenSet[str] = frozenset({
    "MANAGE_USERS",
    "CREATE_USER",
    "RUN_EVALUATIONS",
})

# Read-only operations allowed for all roles
READ_ACTIONS: FrozenSet[str] = frozenset({
    "VIEW_CASE",
    "VIEW_REPORT",
    "VIEW_EVIDENCE",
    "VIEW_PROVENANCE",
    "VIEW_AUDIT",
})


def can_perform_action(role: str, action: str) -> bool:
    """Predicate checking whether a role is authorized for a given action."""
    # Normalize role string
    r = role.lower()

    if r == UserRole.ADMIN.value:
        # Admin can perform review, admin, and read actions
        return action in REVIEW_ACTIONS or action in ADMIN_ACTIONS or action in READ_ACTIONS

    elif r == UserRole.CLINICIAN_REVIEWER.value:
        # Clinician reviewer can perform review and read actions, but NOT admin actions
        return action in REVIEW_ACTIONS or action in READ_ACTIONS

    elif r == UserRole.VIEWER.value:
        # Viewer is read-only; strictly prohibited from review and admin actions
        return action in READ_ACTIONS

    return False


def authorize_action(role: str, action: str) -> bool:
    """Enforce authorization for an action.

    Raises:
        UnauthorizedActionError: If role is not permitted to perform the action.
    """
    if not can_perform_action(role, action):
        logger.warning("Access denied: role '%s' attempted unauthorized action '%s'", role, action)
        raise UnauthorizedActionError(role, action)
    return True

"""Append-Only Audit Repository Layer.

Specification Section 5.3, Section 6, Rule R11:
- Every state change, review action, and agent execution appends an immutable AuditEvent.
- The audit repository exposes NO update or delete method.
"""

import json
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session
from backend.app.persistence.models import AuditEvent as AuditEventModel
from backend.app.schemas.executions import AuditEvent as AuditEventSchema


class AuditRepository:
    """Repository for appending and reading immutable audit events.
    
    In adherence to clinical provenance and security specifications,
    this class explicitly exposes NO update or delete methods.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def append_event(self, event: AuditEventSchema) -> AuditEventModel:
        """Append an immutable audit event to the persistent audit log."""
        payload_str = json.dumps(event.payload) if isinstance(event.payload, dict) else str(event.payload)
        db_event = AuditEventModel(
            id=event.event_id,
            event_type=event.event_type,
            actor_id=event.actor_id,
            actor_role=event.actor_role,
            resource_type=event.resource_type,
            resource_id=event.resource_id,
            payload=payload_str,
            timestamp=event.timestamp,
        )
        self.session.add(db_event)
        self.session.commit()
        self.session.refresh(db_event)
        return db_event

    def get_by_id(self, event_id: str) -> Optional[AuditEventModel]:
        """Retrieve a specific audit event by its unique ID."""
        stmt = select(AuditEventModel).where(AuditEventModel.id == event_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def list_by_resource(self, resource_type: str, resource_id: str) -> List[AuditEventModel]:
        """Retrieve all audit events for a given resource in chronological order."""
        stmt = (
            select(AuditEventModel)
            .where(
                AuditEventModel.resource_type == resource_type,
                AuditEventModel.resource_id == resource_id,
            )
            .order_by(AuditEventModel.timestamp.asc())
        )
        return list(self.session.execute(stmt).scalars().all())

    def list_all(self, limit: int = 100, offset: int = 0) -> List[AuditEventModel]:
        """List audit events with pagination."""
        stmt = (
            select(AuditEventModel)
            .order_by(AuditEventModel.timestamp.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.session.execute(stmt).scalars().all())

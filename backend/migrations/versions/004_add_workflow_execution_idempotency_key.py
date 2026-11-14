"""004 Add idempotency_key to workflow_executions

Revision ID: 004_add_idempotency_key
Revises: 003_eval_labels_gold_answers
Create Date: 2026-11-03 08:30:00.000000

Specification Section 7, API-4:
- Adds idempotency_key column to workflow_executions table.
- Enforces unique constraint per case on (case_id, idempotency_key).
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "004_add_idempotency_key"
down_revision: Union[str, None] = "003_eval_labels_gold_answers"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("workflow_executions") as batch_op:
        batch_op.add_column(sa.Column("idempotency_key", sa.String(length=255), nullable=True))
        batch_op.create_unique_constraint(
            "uq_workflow_executions_case_idempotency",
            ["case_id", "idempotency_key"],
        )


def downgrade() -> None:
    with op.batch_alter_table("workflow_executions") as batch_op:
        batch_op.drop_constraint("uq_workflow_executions_case_idempotency", type_="unique")
        batch_op.drop_column("idempotency_key")

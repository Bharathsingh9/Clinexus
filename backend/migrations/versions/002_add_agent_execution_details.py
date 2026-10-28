"""002 Add details and hashes to agent_executions table

Revision ID: 002_add_agent_execution_details
Revises: 001_initial_schema
Create Date: 2026-10-27 16:30:00.000000

Specification Section 3, 6 (DB-1, DB-3).
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "002_add_agent_execution_details"
down_revision: Union[str, None] = "001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("agent_executions", sa.Column("error_code", sa.String(length=50), nullable=True))
    op.add_column("agent_executions", sa.Column("input_hash", sa.String(length=64), nullable=True))
    op.add_column("agent_executions", sa.Column("output_hash", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_executions", "output_hash")
    op.drop_column("agent_executions", "input_hash")
    op.drop_column("agent_executions", "error_code")

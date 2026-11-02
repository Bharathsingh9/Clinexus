"""003 Add gold answers and relevant passages to evaluation_labels

Revision ID: 003_eval_labels_gold_answers
Revises: 002_add_agent_execution_details
Create Date: 2026-11-01 23:55:00.000000

Specification Section 9.4, 9.5, 9.7 (EVL-5, DAT-4, DAT-5).
Supports storage of exact_answer, ideal_answer, gold_pmids, and relevant_passages mapping.
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "003_eval_labels_gold_answers"
down_revision: Union[str, None] = "002_add_agent_execution_details"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("evaluation_labels") as batch_op:
        batch_op.add_column(sa.Column("question_type", sa.String(length=20), nullable=True))
        batch_op.add_column(sa.Column("exact_answer", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("ideal_answer", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("gold_pmids", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("relevant_passages", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("derivation_source", sa.String(length=100), nullable=True, server_default="DERIVED_FROM_SAMPLE_ID"))
        batch_op.add_column(sa.Column("confirmed_by_human", sa.Boolean(), nullable=False, server_default=sa.text("false")))
        batch_op.alter_column("corpus_id", nullable=True)
        batch_op.create_unique_constraint("uq_evaluation_labels_sample_id", ["sample_id"])


def downgrade() -> None:
    with op.batch_alter_table("evaluation_labels") as batch_op:
        batch_op.drop_constraint("uq_evaluation_labels_sample_id", type_="unique")
        batch_op.alter_column("corpus_id", nullable=False)
        batch_op.drop_column("confirmed_by_human")
        batch_op.drop_column("derivation_source")
        batch_op.drop_column("relevant_passages")
        batch_op.drop_column("gold_pmids")
        batch_op.drop_column("ideal_answer")
        batch_op.drop_column("exact_answer")
        batch_op.drop_column("question_type")

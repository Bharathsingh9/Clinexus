"""Unit tests for Data Leakage Prevention and WorkflowInput Schema Strictness.

Specification Section 9.7 (Req ID: EVL-5):
- WorkflowInput has ONLY sample_id, question, and question_type with extra="forbid".
- Rejects any extra fields (e.g. exact_answer, ideal_answer, gold_pmids, context).
- Leakage test: serializes WorkflowInput for every synthetic question and asserts that
  gold answer strings, sample_ids of other questions, and context snippet texts never appear in it.
"""

import json
import pytest
from pydantic import ValidationError

from backend.app.schemas.evaluation import BioASQGoldLabel, WorkflowInput


def test_workflow_input_forbids_extra_fields():
    """Verify WorkflowInput rejects any extra field that could carry gold data."""
    # Valid construction
    wf = WorkflowInput(
        sample_id="SYNTH_001",
        question="[SYNTHETIC] Is drug X indicated for condition Y?",
        question_type="yesno",
    )
    assert wf.sample_id == "SYNTH_001"
    assert wf.question == "[SYNTHETIC] Is drug X indicated for condition Y?"
    assert wf.question_type == "yesno"

    # Reject extra="forbid" fields
    with pytest.raises(ValidationError):
        WorkflowInput(
            sample_id="SYNTH_001",
            question="[SYNTHETIC] Question text",
            question_type="yesno",
            exact_answer="yes",  # Forbidden extra field!
        )

    with pytest.raises(ValidationError):
        WorkflowInput(
            sample_id="SYNTH_001",
            question="[SYNTHETIC] Question text",
            question_type="factoid",
            ideal_answer=["[SYNTHETIC] Ideal answer"],  # Forbidden!
        )

    with pytest.raises(ValidationError):
        WorkflowInput(
            sample_id="SYNTH_001",
            question="[SYNTHETIC] Question text",
            question_type="summary",
            snippets=["[SYNTHETIC] Snippet text"],  # Forbidden!
        )


def test_leakage_search_in_serialized_workflow_inputs():
    """Serialize workflow input for every synthetic question and assert gold answers,
    other question IDs, and snippet texts do not appear in the serialized representation (SPEC 9.7).
    """
    synthetic_records = [
        {
            "sample_id": "SYNTH_LEAK_01",
            "question": "[SYNTHETIC] Question 1 regarding therapy A?",
            "question_type": "yesno",
            "exact_answer": "DEFINITIVE_GOLD_YES_VERDICT",
            "ideal_answer": "[SYNTHETIC] Ground truth ideal paragraph for question 1.",
            "snippets": ["[SYNTHETIC] Answer-bearing snippet alpha for question 1."],
        },
        {
            "sample_id": "SYNTH_LEAK_02",
            "question": "[SYNTHETIC] Question 2 regarding enzyme B?",
            "question_type": "factoid",
            "exact_answer": "[SYNTHETIC] SuperSpecificGoldTargetEnzyme2026",
            "ideal_answer": "[SYNTHETIC] Ground truth ideal paragraph for question 2.",
            "snippets": ["[SYNTHETIC] Answer-bearing snippet beta for question 2."],
        },
        {
            "sample_id": "SYNTH_LEAK_03",
            "question": "[SYNTHETIC] Question 3 regarding receptor C?",
            "question_type": "list",
            "exact_answer": "[SYNTHETIC] ReceptorGammaSubunit",
            "ideal_answer": "[SYNTHETIC] Ground truth ideal paragraph for question 3.",
            "snippets": ["[SYNTHETIC] Answer-bearing snippet gamma for question 3."],
        },
    ]

    all_sample_ids = {r["sample_id"] for r in synthetic_records}

    for rec in synthetic_records:
        sid = rec["sample_id"]
        wf_input = WorkflowInput(
            sample_id=sid,
            question=rec["question"],
            question_type=rec["question_type"],
        )

        # Serialize in multiple formats: dict, json, str
        serialized_json = wf_input.model_dump_json()
        serialized_str = str(wf_input)
        serialized_dict = str(wf_input.model_dump())

        for serialized in (serialized_json, serialized_str, serialized_dict):
            # 1. Assert gold exact_answer does not appear
            assert rec["exact_answer"] not in serialized, (
                f"Data leakage detected! exact_answer '{rec['exact_answer']}' found in workflow input for {sid}"
            )

            # 2. Assert ideal_answer does not appear
            assert rec["ideal_answer"] not in serialized, (
                f"Data leakage detected! ideal_answer found in workflow input for {sid}"
            )

            # 3. Assert snippet texts do not appear
            for snip in rec["snippets"]:
                assert snip not in serialized, (
                    f"Data leakage detected! snippet '{snip}' found in workflow input for {sid}"
                )

            # 4. Assert other question IDs do not appear
            other_ids = all_sample_ids - {sid}
            for other_id in other_ids:
                assert other_id not in serialized, (
                    f"Cross-question contamination detected! '{other_id}' found in workflow input for {sid}"
                )

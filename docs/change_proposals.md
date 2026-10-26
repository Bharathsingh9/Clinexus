# Change Proposals (Phase 0)

In accordance with Rule **R10**, any high-impact detail, specification difference, or un-specified repository asset is documented here with: Problem, Options, Recommendation, Decision Status, and Impact.

---

## CP-01: Integration and Usage Proposal for MedQA Dataset
- **Status**: DEFERRED (Date: 2026-10-26, Decision Pointer: `SPEC.md` Appendix A, Item A4)
- **Role in Manifest**: `AUDITED_NOT_SPECIFIED` (`confirmed_by_human: false`)
- **Files**:
  - `datasets/medqa/US/train.jsonl` (10,178 records)
  - `datasets/medqa/US/dev.jsonl` (1,272 records)
  - `datasets/medqa/US/test.jsonl` (1,273 records)
  - `datasets/processed/medqa/medqa_normalized.csv` (12,723 records)
- **What the Data Contains**:
  Multiple-choice board examination questions from the US Medical Licensing Examination (USMLE) in English, with 4 or 5 options, clinical vignettes describing patient presentations, and gold answer keys.
- **Problem**:
  SPEC Section 1 states that out-of-scope items include "any dataset not listed in Section 9. A MedQA adapter is built only if MedQA files are found in datasets/medqa/ during the Phase 0 audit; otherwise it is BLOCKED and not built." The base paper used MedQA as its primary benchmark (50 samples, reporting 94% accuracy).
- **Options**:
  1. *Option A (Recommended)*: Keep MedQA un-ingested in Phase 1–4. If human reviewer explicitly approves, implement a separate MedQA benchmark adapter in Phase 5 alongside BioASQ for comparative reproduction against base paper Table 4.
  2. *Option B*: Do not build any MedQA adapter or evaluation code; retain MedQA strictly as an audited, inactive repository asset.
- **Recommendation**:
  Option A. Maintain MedQA files as `AUDITED_NOT_SPECIFIED` and do not build loaders or adapters until human approval.
- **Decision (SPEC Appendix A, Item A4)**:
  **DEFERRED**. Build nothing for MedQA until a SPEC section is added.
- **Impact**:
  Prevents premature scope expansion while preserving reproducibility options for future evaluation phases.

---

## CP-02: Integration and Usage Proposal for MedBullets Dataset
- **Status**: DEFERRED (Date: 2026-10-26, Decision Pointer: `SPEC.md` Appendix A, Item A4)
- **Role in Manifest**: `AUDITED_NOT_SPECIFIED` (`confirmed_by_human: false`)
- **Files**:
  - `datasets/medbullets/op4_eval-00000-of-00001.parquet` (62 records)
  - `datasets/medbullets/op4_test-00000-of-00001.parquet` (308 records)
  - `datasets/medbullets/op4_train-00000-of-00001.parquet` (246 records)
  - `datasets/medbullets/op5_eval-00000-of-00001.parquet` (62 records)
  - `datasets/medbullets/op5_test-00000-of-00001.parquet` (308 records)
  - `datasets/medbullets/op5_train-00000-of-00001.parquet` (246 records)
  - `datasets/processed/medbullets/medbullets_op4_normalized.csv` (616 records)
- **What the Data Contains**:
  Step 1 and Step 2/3 clinical vignette practice questions and bullet explanations from MedBullets, with 4-choice (`op4`) and 5-choice (`op5`) splits.
- **Problem**:
  The base paper evaluated 50 questions from MedBullets (reporting 84% accuracy via semantic similarity $\theta=0.80$). However, SPEC Section 9 specifies only BioASQ.
- **Options**:
  1. *Option A (Recommended)*: Exclude MedBullets from all pipelines, loaders, and evaluation harnesses for the current release. Keep as `AUDITED_NOT_SPECIFIED`.
  2. *Option B*: Author an additional benchmark evaluator in Phase 5 for MedBullets.
- **Recommendation**:
  Option A. MedBullets is out of scope per SPEC Section 1. Do not build loaders or evaluation code.
- **Decision (SPEC Appendix A, Item A4)**:
  **DEFERRED**. Build nothing for MedBullets until a SPEC section is added.
- **Impact**:
  Ensures the platform focuses strictly on the specified BioASQ evaluation pipeline.

---

## CP-03: Integration and Usage Proposal for PubMedQA Dataset
- **Status**: DEFERRED (Date: 2026-10-26, Decision Pointer: `SPEC.md` Appendix A, Item A4)
- **Role in Manifest**: `AUDITED_NOT_SPECIFIED` (`confirmed_by_human: false`)
- **Files**:
  - `datasets/pubmedqa/ori_pqal.json` (1,000 labeled records)
  - `datasets/processed/pubmedqa/pubmedqa_normalized.csv` (1,000 records)
- **What the Data Contains**:
  Biomedical question-answering pairs derived from PubMed abstracts, each containing research question, abstract contexts, Mesh terms, and yes/no/maybe label (`final_decision`).
- **Problem**:
  The base paper evaluated 50 PubMedQA questions (reporting 88% accuracy). SPEC Section 9 specifies only BioASQ.
- **Options**:
  1. *Option A (Recommended)*: Exclude PubMedQA from pipelines, loaders, and evaluation harnesses. Keep as `AUDITED_NOT_SPECIFIED`.
  2. *Option B*: Construct a PubMedQA adapter for ablation benchmarking.
- **Recommendation**:
  Option A. Respect SPEC Section 1 out-of-scope boundaries. Stop work on PubMedQA.
- **Decision (SPEC Appendix A, Item A4)**:
  **DEFERRED**. Build nothing for PubMedQA until a SPEC section is added.
- **Impact**:
  Avoids unapproved evaluation scope creep.

---

## CP-04: Confidence Calculation Formula Specification
- **Status**: APPROVED (Date: 2026-10-26, Decision Pointer: `SPEC.md` Appendix A, Item A1)
- **Problem**:
  In the base paper (Section II-A.4, Section II-D), the confidence score is introduced as a mechanism to trigger the adaptive feedback loop when falling below threshold $\tau = 0.65$. However, the paper is **completely silent on the mathematical formula**, providing only qualitative factors ("similarity score of RAG, evidence score of web information, and degree of concordance"). SPEC Section 2 provides a fully specified engineering formula:
  $$\text{confidence} = 0.50 \cdot \text{claim\_support} + 0.20 \cdot \text{retrieval\_strength} + 0.15 \cdot \text{source\_availability} + 0.15 \cdot \text{info\_completeness}$$
- **Options**:
  1. *Option A (Recommended)*: Confirm and implement the SPEC Section 2 formula as the authoritative engineering definition, tagged `[ENGINEERING-DECISION: CONF-FORMULA]`.
  2. *Option B*: Attempt to reconstruct an uncalibrated LLM confidence prompt mimicking the paper's vague phrasing.
- **Recommendation**:
  Option A. The SPEC Section 2 formula is deterministic, reproducible, auditable, and prevents LLM self-delusion from dominating the score.
- **Decision (SPEC Appendix A, Item A1)**:
  **APPROVED** as the documented alternative, tagged `[PAPER-GAP]` and `[ENGINEERING-DECISION: CONF-FORMULA]`. The paper names three qualitative factors (RAG similarity, external evidence score, concordance) and the threshold 0.65 [PAPER-VERIFIED: II-A.4, II-D]; the formula is not from the paper. In the baseline pipeline (no claim verifier), omit `claim_support` and use weights $0.40 \cdot \text{retrieval\_strength} + 0.30 \cdot \text{source\_availability} + 0.30 \cdot \text{info\_completeness}$ `[ENGINEERING-DECISION]`.
- **Impact**:
  Provides a transparent, auditable confidence metric for Agent 4, Agent 5, and the Evidence Safety Gate across both baseline and extended pipelines.

---

## CP-05: Workflow Graph Topology and Stage Ordering
- **Status**: APPROVED (Date: 2026-10-26, Decision Pointer: `SPEC.md` Appendix A, Item A2)
- **Problem**:
  The base paper workflow is inherently sequential: its Scanner consumes the RAG Analyzer's candidate hypotheses ("preliminary diagnosis list") to formulate external search queries (`Clarifier -> RAG Analyzer -> Scanner -> Fusion -> Optimizer`). Although Section II-A (p. 167394) uses the colloquial phrase *"In parallel, the Evidence-Based Scanner Agent acts to validate each hypothesis in this preliminary diagnosis list"*, the Scanner cannot operate in parallel without first receiving the hypotheses generated by the RAG Analyzer. SPEC Section 3 defines a strict 10-node deterministic LangGraph:
  `validate_case -> clinical_text_clarifier -> clarification_check -> symptom_rag_analyzer -> evidence_based_scanner -> clinical_data_fusion -> claim_verification -> evidence_safety_gate -> (adaptive_optimizer | finalize_ready | finalize_abstain) -> persist_and_finish`.
- **Options**:
  1. *Option A (Recommended)*: Confirm the SPEC Section 3 sequential LangGraph topology, incorporating Extension A (`claim_verification`) and Extension B (`evidence_safety_gate`, `finalize_abstain`), running the scanner sequentially after the RAG analyzer so it consumes both candidate hypotheses and local retrieval gaps (`gap_hints`).
  2. *Option B*: Attempt an unconstrained parallel execution which would fail because the Scanner depends on the RAG Analyzer's hypotheses.
- **Recommendation**:
  Option A. The sequential execution accurately reflects the data dependency in the paper while directly enforcing safety invariant R11, clinical reliability, and structured gap passing.
- **Decision (SPEC Appendix A, Item A2)**:
  **APPROVED**. Build the baseline first (SPEC nodes without `claim_verification` and `evidence_safety_gate`), then the extended graph. The node order stays sequential (scanner uses RAG gaps). Quoting base paper Section II-A: *"In parallel, the Evidence-Based Scanner Agent acts to validate each hypothesis in this preliminary diagnosis list"*; the sequential order in SPEC is recorded as `[ENGINEERING-DECISION: AGT-3-SEQ]`.
- **Impact**:
  Guarantees that unverified or contradicted claims cannot reach clinicians, allows scanner to focus on candidate hypotheses and RAG gaps, and provides clean phased delivery from baseline (Phase 3) to extensions (Phase 4).

---

## CP-06: Handling of Reached Iteration Limit
- **Status**: APPROVED (Date: 2026-10-26, Decision Pointer: `SPEC.md` Appendix A, Item A3)
- **Problem**:
  The base paper (Section II-D, Table 6) caps optimizer iterations at 3, but treats the resulting output after 3 iterations as final regardless of confidence. SPEC Section 3 defines the gate logic and budget handling:
  *"The iteration counter increments once per optimizer call; at the maximum the optimizer is not called again and the gate must finalize. Reaching the budget is never success. The gate is one pure function evaluate_gate(state, settings) -> GateDecision with an exhaustive truth-table test."*
  Under SPEC Section 3, `evidence_safety_gate` takes the first matching rule:
  1. Out-of-scope request -> `finalize_abstain, OUT_OF_SCOPE`
  2. Verifier error on any material or safety-critical claim -> `finalize_abstain, VERIFICATION_FAILED`
  3. Any safety-critical claim not SUPPORTED, or any material claim CONTRADICTED -> `optimizer if iterations remain, else finalize_abstain`
  4. Fewer than MIN_RELEVANT_PASSAGES relevant passages -> `optimizer if iterations remain and an alternative query exists, else finalize_abstain, NO_RELEVANT_EVIDENCE`
  5. Confidence below threshold -> `optimizer if iterations remain, else finalize_abstain, LOW_CONFIDENCE`
  6. Otherwise -> `finalize_ready`
- **Options**:
  1. *Option A (Recommended)*: In the full system, reaching the limit is never success: the gate decides (`ready` or `abstain`). If rules 1–5 trigger and no iterations remain, route to `finalize_abstain` with `ITERATION_BUDGET_EXHAUSTED` plus the specific reason codes; if rule 6 is met, route to `finalize_ready`. In the baseline system, follow the paper: output the iteration-3 result labeled `"BASELINE: budget reached, not verified"`.
  2. *Option B*: Treat reaching budget 3 as automatic success or output unverified reports as ready in all configurations.
- **Recommendation**:
  Option A. In a clinical decision-support tool, reaching the budget must force gate evaluation rather than automatic success or unconditional termination without reason codes.
- **Decision (SPEC Appendix A, Item A3)**:
  **APPROVED**. In the full system, reaching the limit is never success: the gate decides (`ready` or `abstain`). When abstaining after the budget is spent, include `ITERATION_BUDGET_EXHAUSTED` plus the specific reason codes. The baseline follows the paper: it outputs the iteration-3 result labeled `"BASELINE: budget reached, not verified"`.
- **Impact**:
  Upholds Rule R11 (Safety Invariants) in the full system while enabling faithful base-paper reproduction in Phase 3 baseline evaluation.

---

## CP-07: Candidate Hypotheses Generation, Validation, and Downstream Flow
- **Status**: APPROVED (Date: 2026-10-26, Decision Pointer: `SPEC.md` Appendix A, Item A5)
- **Problem**:
  In the base paper (Section II-A.2, Section II-A.3), the Symptom RAG Analyzer proposes candidate diagnostic hypotheses that the Evidence-Based Scanner then validates. However, SPEC Section 4.2 originally listed only `RetrievedEvidenceSet` as output, without an explicit `CandidateHypothesis` schema, validation rules for hypothesis citations, or guidance for non-diagnostic benchmark runs.
- **Options**:
  1. *Option A (Recommended)*: The Symptom RAG Analyzer outputs a list of `CandidateHypothesis` objects (`label`, `supporting_evidence_ids`, `retrieval_scores`). Code retains a hypothesis only if at least one retrieved passage supports it, and rejects any cited ID not in the retrieved set. Hypotheses are defined as "candidate considerations", never diagnoses. The Scanner builds one PubMed query per hypothesis plus queries for gaps. Fusion consumes the hypotheses. In benchmark mode, where the question is not diagnostic, the list may be empty and the Scanner uses gap queries. Add `CandidateHypothesis` to the schema list.
  2. *Option B*: Let the LLM generate ungrounded diagnostic strings without passage support checks or schema validation.
- **Recommendation**:
  Option A. Aligns with base paper multi-agent hypothesis-testing mechanics while strictly enforcing passage grounding and clinical safety invariants (SAF-1).
- **Decision (SPEC Appendix A, Item A5)**:
  **APPROVED**. The Symptom RAG Analyzer also outputs a list of `CandidateHypothesis` objects (`label`, supporting `evidence_ids`, `retrieval_scores`). Code keeps a hypothesis only if at least one retrieved passage supports it, and rejects any cited ID not in the retrieved set. Hypotheses are "candidate considerations", never diagnoses. The Scanner builds one PubMed query per hypothesis plus queries for gaps. Fusion consumes the hypotheses. In benchmark mode, where the question is not diagnostic, the list may be empty and the Scanner uses gap queries. Add the `CandidateHypothesis` schema to the schema list.
- **Impact**:
  Establishes the formal contract between RAG Analyzer, Scanner, and Fusion, guarantees that all considerations are grounded in retrieved evidence, and adheres to non-diagnostic safety constraints.

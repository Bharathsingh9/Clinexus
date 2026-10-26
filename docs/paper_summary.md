# Base Paper Summary & Specification Fidelity Analysis

## 1. Paper Identification & Metadata
- **Title**: An Adaptive Multi-Agent LLM-Based Clinical Decision Support System Integrating Biomedical RAG and Web Intelligence
- **Authors**: Çağatay Umut Öğdü, Kübra Arslanoğlu (Member, IEEE), Mehmet Karaköse (Senior Member, IEEE)
- **Affiliation**: Department of Computer Engineering, Fırat University, 23119 Elâzığ, Türkiye
- **Publication**: *IEEE Access*, vol. 13, pp. 167390–167404, September 2025
- **DOI**: `10.1109/ACCESS.2025.3613340`
- **License**: Creative Commons Attribution 4.0 (CC BY 4.0)
- **Local Path**: `docs/paper/base_paper.pdf` [PAPER-VERIFIED: p. 167390]

---

## 2. Extraction of the Five Original Agents

### Agent 1: Clinical Text Clarifier Agent
- **Paper Section**: Section II-A.1 (pp. 167393–167394) [PAPER-VERIFIED: II-A.1]
- **Role**: Normalizes unstructured, disorganized, panicked clinical narratives. Performs text normalization (spelling, abbreviations such as "HT", punctuation), Named Entity Recognition (NER: Symptom, Disease, Medication, Test Result), and association extraction (linking symptoms to anatomical regions, e.g., "numbness" in "left arm"). Mentions mapping clinical concepts to SNOMED CT or ICD-10.
- **Inputs**: Raw unstructured clinical narrative (plain text).
- **Outputs**: Structured JSON object with predefined schema containing clinical entities and relationships.
- **Model Selected in Paper**: DeepSeek-R1 (zero-shot inference; chosen for high schema concordance and low latency).
- **Fidelity & Gaps**:
  - [PAPER-VERIFIED: II-A.1]: Clinical Text Clarifier processes raw unstructured text into structured JSON entities and relationships using DeepSeek-R1.
  - [PAPER-GAP]: Paper claims mapping concepts to international medical ontologies (SNOMED CT, ICD-10), but provides no mapping table, ontology database, API, or license.
  - [ENGINEERING-DECISION: AGT-1-VOCAB]: SPEC 4.1 defines a local symptom vocabulary `backend/app/agents/data/symptom_vocab.json` (vocabulary=LOCAL) without claiming unlicensed SNOMED CT or ICD-10.
  - [PAPER-GAP]: Paper describes no clarification question generation for missing required fields (age, sex, duration) or contradictions.
  - [ENGINEERING-DECISION: AGT-1-CLARIFY]: SPEC 4.1 defines up to 5 clarification questions (blocking vs non-blocking) and a deterministic `clarification_check` node.

### Agent 2: Symptom RAG Analyzer Agent
- **Paper Section**: Section II-A.2, Section II-B.1, Section II-B.2 (pp. 167394–167398) [PAPER-VERIFIED: II-A.2, II-B]
- **Role**: Converts structured symptom inputs into a semantic query vector. Executes cosine-similarity search over a Qdrant vector database pre-indexed with biomedical literature. Proposes candidate diagnostic hypotheses retained only when explicitly supported by at least one retrieved passage.
- **Inputs**: Structured clinical symptom data from Agent 1.
- **Outputs**: Ranked list of top-$k$ pertinent document passages and candidate diagnostic hypotheses (`CandidateHypothesis`) with citations.
- **Embedding Model in Paper**: `BioBERT-mnli-snli-scinli-scitail-mednli-stsb` (768-dimensional embeddings; resolved Hugging Face ID `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`).
- **LLM Selected in Paper**: Gemini-2.0-Flash (chosen for long-context semantic alignment and fast function calling).
- **Fidelity & Gaps**:
  - [PAPER-VERIFIED: II-A.2, II-B.1]: Vector database uses Qdrant with 768-dimensional embeddings, cosine similarity search, and top-$k$ context passage retrieval.
  - [PAPER-GAP]: Paper built its vector database from 889 PubMed PDFs (downloaded via Entrez) and 5,630 patient texts from a HuggingFace symptom-disease dataset (`dux-tecblic/symptom-disease-dataset`), neither of which are provided in the repository. The repository instead provides the BioASQ snippets corpus (75,919 rows).
  - [ENGINEERING-DECISION: RET-EMBED / SPEC App A6]: Default embedding model is the paper's model `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` [PAPER-VERIFIED: II-B.1]. `NeuML/pubmedbert-base-embeddings` is retained as an explicit fallback and never the silent default.
  - [ENGINEERING-DECISION: AGT-2-HYP / SPEC App A5]: Agent 2 outputs structured `CandidateHypothesis` objects (`label`, `supporting_evidence_ids`, `retrieval_scores`). Hypotheses are candidate considerations, never diagnoses (SAF-1); code rejects any cited ID not in the retrieved set.
  - [ENGINEERING-DECISION: AGT-LLM / SPEC App A7]: Uses single LLM adapter with per-agent model setting `LLM_MODEL_RAG` (defaulting to `LLM_MODEL`).
  - [PAPER-GAP]: Paper describes "medical term boosting" and multi-level query fallback (lower similarity threshold $\to$ simplify query $\to$ re-cluster symptoms), but omits mathematical formulations and deterministic rule definitions.

### Agent 3: Evidence-Based Scanner Agent
- **Paper Section**: Section II-A.3 (p. 167395) [PAPER-VERIFIED: II-A.3]
- **Role**: Mitigates risk of static vector database staleness by actively scanning medical literature for each diagnostic hypothesis and RAG gap. Submits programmatic searches to scholarly indexes. Categorizes evidence by publication type.
- **Inputs**: Candidate hypotheses from Agent 2, structured case data, and gap hints.
- **Outputs**: Current external evidence summaries and citations weighted by evidence level.
- **Model Selected in Paper**: Qwen2-72B (chosen for search query generation and dense text summarization).
- **Fidelity & Gaps**:
  - [PAPER-VERIFIED: II-A.3]: Scans external literature (PubMed) sequentially for candidate hypotheses to fetch up-to-date guidelines and evidence summaries.
  - [DEVIATION-FROM-PAPER: AGT-3-SOURCES / SPEC App A9]: Base paper specifies web search APIs (Serper), Google Scholar, and NICE in addition to PubMed. Clinexus uses PubMed only via NCBI E-utilities HTTPS. Where EFetch returns publication types (e.g., systematic review, practice guideline), store them as `evidence_level`; do not invent hierarchy weights.
  - [ENGINEERING-DECISION: AGT-3-NCBI]: SPEC 4.3 and Section 2 mandate direct HTTPS integration with official NCBI E-utilities (ESearch, EFetch) with strict rate limits (3 req/s without key, 10 req/s with key) and an offline mode (`EXTERNAL_MODE=offline`) to prevent non-deterministic web scraping failure.
  - [ENGINEERING-DECISION: AGT-3-SCAN / SPEC App A5]: Builds one PubMed query per candidate hypothesis plus queries for RAG gaps. In benchmark mode, candidate hypothesis list may be empty and Scanner uses gap queries.
  - [ENGINEERING-DECISION: AGT-LLM / SPEC App A7]: Uses single LLM adapter with per-agent model setting `LLM_MODEL_SCANNER` (defaulting to `LLM_MODEL`).

### Agent 4: Clinical Data Fusion Agent
- **Paper Section**: Section II-A.4, Section II-C (pp. 167395, 167398) [PAPER-VERIFIED: II-A.4]
- **Role**: Core synthesizer. Combines static RAG passages, dynamic web evidence, and candidate hypotheses. Executes a conflict detection protocol to identify inconsistencies. Evaluates evidence concordance and calculates a confidence score.
- **Inputs**: Static RAG evidence set, candidate hypotheses, dynamic external evidence set, and structured patient case.
- **Outputs**: Structured, synthesized clinical report with source metadata and confidence levels.
- **Model Selected in Paper**: GPT-4o-mini (chosen for scalable clinical data synthesis).
- **Fidelity & Gaps**:
  - [PAPER-VERIFIED: II-A.4]: Combines multi-source evidence, performs cross-checking, identifies contradictions, and synthesizes a structured report with confidence evaluation.
  - [PAPER-GAP]: The paper provides **no mathematical formula** for the "calibrated confidence score". It states only that confidence depends on "similarity score of RAG, evidence score of web information, and degree of concordance".
  - [ENGINEERING-DECISION: CONF-FORMULA]: SPEC Section 2 specifies an exact, deterministic formula:
    $$\text{confidence} = 0.50 \cdot \text{claim\_support} + 0.20 \cdot \text{retrieval\_strength} + 0.15 \cdot \text{source\_availability} + 0.15 \cdot \text{info\_completeness}$$
    In baseline pipeline (no claim verifier), omit `claim_support` and use weights: $0.40 \cdot \text{retrieval\_strength} + 0.30 \cdot \text{source\_availability} + 0.30 \cdot \text{info\_completeness}$ [ENGINEERING-DECISION / SPEC App A1].
  - [DEVIATION-FROM-PAPER: SAFETY]: Base paper Fusion agent generates treatment guidance and protocols [PAPER-VERIFIED: II-A.4, II-C]. This build explicitly omits all treatment recommendations, prescriptions, and dosing orders (SPEC Section 1, Appendix A8, SAF-1, R11). The baseline is "base-paper-style, safety-restricted" and also omits treatment.
  - [ENGINEERING-DECISION: AGT-LLM / SPEC App A7]: Uses single LLM adapter with per-agent model setting `LLM_MODEL_FUSION` (defaulting to `LLM_MODEL`).

### Agent 5: Adaptive Optimizer Agent
- **Paper Section**: Section II-A.5, Section II-D (pp. 167395, 167398–167399) [PAPER-VERIFIED: II-A.5, II-D]
- **Role**: Quality check and meta-controller. If confidence falls below acceptance threshold ($\tau = 0.65$), activates a feedback loop. Identifies untrustworthy or low-confidence data, re-organizes query parameters (keyword adjustments, expansion/reduction), and issues feedback guidance to earlier agents. Halts when confidence $\ge 0.65$ or when maximum iterations (3) is reached.
- **Inputs**: Synthesized report, confidence evaluation, previous iteration history, and gate findings.
- **Outputs**: Optimization guidance report, modified search queries, or termination decision.
- **Model Selected in Paper**: Gemini-2.0-Flash (chosen for low latency and high call throughput).
- **Fidelity & Gaps**:
  - [PAPER-VERIFIED: II-A.5, Table 2, Table 6]: Optimization loop operates on confidence threshold $\tau = 0.65$ with a hard ceiling of 3 iterations.
  - [PAPER-GAP]: The paper relies on LLM-driven feedback guidance without deterministic routing rules or progress-tracking termination conditions.
  - [ENGINEERING-DECISION: AGT-5-ROUTING]: SPEC 4.5 specifies deterministic routing rules (`evaluate_gate`), machine-checkable completion conditions, and automatic stop with `NO_PROGRESS` when two consecutive iterations leave non-SUPPORTED material claims unchanged.
  - [ENGINEERING-DECISION: AGT-5-BUDGET / SPEC App A3]: In full system, reaching budget is never success: safety gate decides (`ready` or `abstain`). When abstaining, includes `ITERATION_BUDGET_EXHAUSTED` plus specific reason codes. Baseline outputs iteration-3 result labeled `"BASELINE: budget reached, not verified"`.
  - [ENGINEERING-DECISION: AGT-LLM / SPEC App A7]: Uses single LLM adapter with per-agent model setting `LLM_MODEL_OPTIMIZER` (defaulting to `LLM_MODEL`).

---

## 3. The Confidence Mechanism Comparison

| Feature | Base Paper | SPEC.md | Classification |
| :--- | :--- | :--- | :--- |
| **Threshold** | $\tau = 0.65$ | `CONFIDENCE_THRESHOLD = 0.65` | `[PAPER-VERIFIED: II-A.5]` |
| **Formula** | **None** (stated textually as dependent on RAG score, web score, concordance) | Explicit deterministic formula: $0.50 \cdot \text{claim\_support} + 0.20 \cdot \text{retrieval\_strength} + 0.15 \cdot \text{source\_availability} + 0.15 \cdot \text{info\_completeness}$ | `[PAPER-GAP]` resolved by `[ENGINEERING-DECISION: CONF-FORMULA]` |
| **Nature of Value** | Described as "calibrated confidence" | Explicitly uncalibrated engineering score; never called a probability | `[ENGINEERING-DECISION]` |
| **Override Power** | Low confidence triggers loop; no safety gate defined | High confidence can **never** override failed verification or abstention conditions | `[ENGINEERING-DECISION: EXT-B]` |

---

## 4. The Iteration Mechanism Comparison

| Feature | Base Paper | SPEC.md | Classification |
| :--- | :--- | :--- | :--- |
| **Max Iterations** | 3 cycles (Table 2, Section II-D) | `MAX_OPTIMIZER_ITERATIONS = 3` (hard ceiling in code) | `[PAPER-VERIFIED: II-D]` |
| **Convergence** | Notes convergence in 2–3 cycles on MedQA (mean 3.7 min) | Enforces bounded execution; increments counter deterministically | `[PAPER-VERIFIED: II-D]` |
| **Termination on Budget** | Reaching 3 iterations halts loop; paper outputs final result | In full system, reaching budget is **never** success: the safety gate decides (`ready` or `abstain`) per SPEC Section 3 and Appendix A3. When abstaining after budget is spent, includes `ITERATION_BUDGET_EXHAUSTED` plus specific reason codes. Baseline outputs iteration-3 result labeled `"BASELINE: budget reached, not verified"`. | `[PAPER-VERIFIED: II-D]` / `[ENGINEERING-DECISION: EXT-B, App A3]` |
| **Stagnation Guard** | Paper notes feedback loop repeats suggestions on cycles 4–5 | Hard stop with `NO_PROGRESS` if non-SUPPORTED claims unchanged over 2 cycles | `[ENGINEERING-DECISION: AGT-5]` |

---

## 5. Evaluation Protocol Comparison

### Datasets Evaluated in Paper
- **MedQA**: 50 questions evaluated (out of USMLE benchmark). Exact-match accuracy on options A–E. Micro-averaged P/R/F1/specificity. Reported accuracy: 94.0% [PAPER-VERIFIED: Section III, Table 3, Table 4].
- **PubMedQA**: 50 questions evaluated. Semantic similarity threshold $\theta = 0.80$ using sentence embeddings. Reported accuracy: 88.0% [PAPER-VERIFIED: Section III, Table 3, Table 4].
- **MedBullets**: 50 questions evaluated. Semantic similarity threshold $\theta = 0.80$. Reported accuracy: 84.0% [PAPER-VERIFIED: Section III, Table 3, Table 4].
- **Ablation Study**: 20 questions from MedQA evaluated across 5 agent removal conditions [PAPER-VERIFIED: Section III, Table 7].

### Datasets Specified in SPEC Section 9
- **BioASQ**: 5,729 questions and 75,919 evidence snippets. Deterministic 20% dev / 80% eval split stratified by question type (seed 42, saved to `datasets/processed/splits_v1.json`, SPEC 9.6; tuning on dev only, reported numbers come only from eval, labeled `NON_OFFICIAL_SPLIT`). Smoke30 subset for regression and latency checks. Scorers in `evaluation/bioasq_scoring.py` follow published BioASQ evaluation guidelines with the version recorded; if guidelines cannot be accessed, metrics are labeled approximations with their formulas explicitly stated (SPEC 9.9) [ENGINEERING-DECISION: DAT-1..8].
- **Status of MedQA / MedBullets / PubMedQA**: Present in repository under `datasets/`, audited in Phase 0 with `role: AUDITED_NOT_SPECIFIED` (`confirmed_by_human: false`). No adapters, loaders, or evaluators built in Phase 0; deferred per SPEC Section 1 and Appendix A4 (CP-01, CP-02, CP-03).

---

## 6. Approved Extensions (A, B, C, D) Gap Analysis

### Extension A: Claim Verification Agent (EXT-A)
- **Paper**: [PAPER-GAP]. No separate verification agent exists. Fusion outputs diagnostic recommendations directly.
- **SPEC 5.1**: [ENGINEERING-DECISION: EXT-A]. Independent verification agent performing:
  1. Deterministic citation presence check.
  2. Entity, unit, and number presence check in cited text.
  3. Model-based entailment judgment with exact character offset spans.
  4. Contradiction scanning across retrieved passages.

### Extension B: Explicit Abstention & Safety Gate (EXT-B)
- **Paper**: [PAPER-GAP]. The paper does not implement abstention; it always attempts to generate diagnostic results and treatment plans.
- **SPEC 3 & 5.2**: [ENGINEERING-DECISION: EXT-B]. Six-rule deterministic `evidence_safety_gate` (`evaluate_gate`), abstention decision schema (`AbstentionDecision`), 10 explicit abstention reason codes, and strict redaction of unverified clinical considerations upon abstention.

### Extension C: Evidence-Level Provenance & Audit Trail (EXT-C)
- **Paper**: [PAPER-GAP]. The paper mentions traceable citations in text, but lacks relational provenance graphs or immutable audit logs.
- **SPEC 5.3**: [ENGINEERING-DECISION: EXT-C]. Relational provenance edge tables (`SUPPORTS`, `CONTRADICTS`, `DERIVED_FROM_PATIENT_INPUT`), immutable append-only `AuditEvent` table, and `assert_provenance_complete` verification.

### Extension D: Human-in-the-Loop Review (EXT-D)
- **Paper**: [PAPER-GAP]. Section IV acknowledges that physicians make final decisions, but defines no interactive HITL workflow or interface.
- **SPEC 5.4**: [ENGINEERING-DECISION: EXT-D]. Strict state transition machine (8 states, pure transition function in `orchestration/states.py`), 5 clinician review actions, append-only clinician notes, and RBAC authorization (`clinician_reviewer`, `admin`, `viewer`).

---

## 7. Items for Change Proposals (`docs/change_proposals.md`)
The following items represent design gaps or differences between the paper and the specification that affect agent roles, workflow order, confidence calculation, or iteration limits:
1. **Confidence Calculation Formula**: Paper provides no formula; SPEC defines deterministic 4-part weighted formula.
2. **Workflow Graph Ordering with Extensions**: Paper uses a sequential 5-agent loop (Scanner consumes hypotheses); SPEC places Clarification Check, Claim Verification, and Safety Gate into deterministic LangGraph workflow.
3. **Evaluation Datasets (BioASQ vs MedQA/MedBullets/PubMedQA)**: Paper evaluated on 50-sample subsets of MedQA, PubMedQA, MedBullets; repository contains full raw/normalized sets for all three plus BioASQ, while SPEC Section 9 specifies BioASQ.
4. **Candidate Hypotheses Contract**: Agent 2 outputs `CandidateHypothesis` objects supported by passages; Scanner conducts per-hypothesis PubMed queries; Fusion synthesizes them.

---

## 8. Summary of Approved Baseline and Extended Differences (SPEC Appendix A1 to A10)

| Item | Topic | Base Paper Behavior | Clinexus Implementation & Fidelity Classification |
| :--- | :--- | :--- | :--- |
| **A1** | **Confidence Formula** | Stated qualitatively without formula [PAPER-GAP]. | Deterministic 4-part formula in full pipeline (`[ENGINEERING-DECISION: CONF-FORMULA]`). In baseline (no verifier), weights are 0.40 retrieval strength, 0.30 source availability, 0.30 info completeness. |
| **A2** | **Workflow Topology** | Mentions parallel scanning in Section II-A, but workflow is sequential as Scanner depends on preliminary hypotheses. | Sequential node ordering: `symptom_rag_analyzer -> evidence_based_scanner` (`[ENGINEERING-DECISION: AGT-3-SEQ]`). Baseline runs 5 agents; extended graph adds verification and gate. |
| **A3** | **Budget Termination** | Reaches 3 iterations and outputs result as final [PAPER-VERIFIED: II-D]. | Reaching budget is never success: safety gate evaluates rules (`ready` vs `abstain`). If abstaining, records `ITERATION_BUDGET_EXHAUSTED` + reasons. Baseline outputs iteration-3 result labeled `"BASELINE: budget reached, not verified"` (`[ENGINEERING-DECISION: EXT-B, App A3]`). |
| **A4** | **Unspecified Datasets** | Evaluated on MedQA (94%), PubMedQA (88%), MedBullets (84%). | All three datasets deferred per SPEC Section 1 and Appendix A4 (`AUDITED_NOT_SPECIFIED`). BioASQ is the authoritative evaluation benchmark. |
| **A5** | **Candidate Hypotheses** | Agent 2 proposes preliminary hypotheses that Agent 3 scans to validate [PAPER-VERIFIED: II-A.2, II-A.3]. | Symptom RAG Analyzer outputs `CandidateHypothesis` objects (`label`, supporting `evidence_ids`, `retrieval_scores`). Hypotheses are candidate considerations, never diagnoses (SAF-1). Code retains a hypothesis only if supported by $\ge 1$ retrieved passage, and rejects any cited ID not in retrieved set. Scanner builds 1 PubMed query per hypothesis plus queries for gaps. Fusion consumes hypotheses. In benchmark mode, list may be empty and Scanner uses gap queries (`[ENGINEERING-DECISION: AGT-2-HYP, AGT-3-SCAN]`). |
| **A6** | **Embedding Model Resolution** | Uses `BioBERT-mnli-snli-scinli-scitail-mednli-stsb` (768-dim) [PAPER-VERIFIED: II-B.1]. | Resolved exact Hugging Face ID: `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` [PAPER-VERIFIED: II-B.1]. Default is this resolved model. `NeuML/pubmedbert-base-embeddings` is an explicit fallback and never the silent default (`[ENGINEERING-DECISION: RET-EMBED, App A6]`). |
| **A7** | **LLM Architecture & Models** | Heterogeneous multi-provider setup (DeepSeek-R1, Gemini-2.0-Flash, Qwen2-72B, GPT-4o-mini). | Single LLM provider adapter with per-agent model settings (`LLM_MODEL_CLARIFIER`, `LLM_MODEL_RAG`, `LLM_MODEL_SCANNER`, `LLM_MODEL_FUSION`, `LLM_MODEL_OPTIMIZER`, `VERIFIER_MODEL`), all defaulting to `LLM_MODEL` (`[ENGINEERING-DECISION / DEVIATION-FROM-PAPER: AGT-LLM]`). |
| **A8** | **Treatment Guidance** | Generates treatment guidance and protocols [PAPER-VERIFIED: II-A.4, II-C]. | **Omitted entirely**: Clinexus prohibits all treatment instructions, prescriptions, and dosing orders (`[DEVIATION-FROM-PAPER: SAFETY]`, SAF-1, R11). The baseline is "base-paper-style, safety-restricted" and also omits treatment. |
| **A9** | **External Evidence Sources** | Uses PubMed, Google Scholar, NICE, and Serper API [PAPER-VERIFIED: II-A.3]. | **PubMed only** via official NCBI E-utilities HTTPS (`[DEVIATION-FROM-PAPER]`, AGT-3-NCBI). When EFetch returns publication types, store as `evidence_level` without inventing hierarchy weights. |
| **A10** | **Baseline Definition** | N/A | Where documentation refers to "paper-faithful baseline", all deviations from the paper must be explicitly stated (A8 safety-restricted treatment omission, A9 PubMed-only external scanning, A6 fallback if activated). |

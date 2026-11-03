# BioASQ Evaluation Protocol and Metrics Specification

Specification Section 9.8, 9.9, 9.10, 10, Appendix A11.

---

## 1. Overview and Scoring Basis

All metric scoring implementations in Clinexus for BioASQ questions and retrieval benchmarking operate strictly under the designation:
```python
scoring_basis = "APPROXIMATION"
```
**Rationale (DEC-049)**: The official BioASQ evaluation measures portal (`http://participants-area.bioasq.org/general_information/eval_measures/` and `https://participants-area.bioasq.org/`) was accessed on 2026-11-02 via read-only web lookup and returned HTTP 404 / required participant credentials. Consequently, all metric implementations represent documented engineering approximations implemented from published challenge literature; official challenge equivalence is **never claimed**.

---

## 2. Retrieval Evaluation Protocol (SPEC Appendix A11)

Retrieval evaluation is performed at the **corpus-row level** (using the BioASQ corpus source row `legacy_id`):

1. **Evidence ID Mapping**: Each retrieved passage chunk carrying platform evidence ID `ev:bioasq:<legacy_id>:<chunk_index>` is mapped to its underlying corpus row (`legacy_id`).
2. **Chunk-to-Row Score Collapsing**: When multiple chunks belonging to the same corpus row are retrieved, they are collapsed into a single row entry retaining the **maximum (best) retrieval score**.
3. **Row Ranking**: Collapsed rows are ranked in descending order of their best retrieval score, using stable tie-breaking on `legacy_id`.
4. **Duplicate-Group Relevance Rule**: If duplicate groups are present, a retrieved row is scored as relevant if the row itself or **any member of its duplicate group** is in the ground-truth relevant set.
5. **Retrieval Depth Cap**:
   > **Note on Retrieval Depth**: Retrieval evaluation is bounded by `RETRIEVAL_TOP_K` (default 10). When a question has more relevant corpus rows than the `RETRIEVAL_TOP_K` threshold, maximum achievable recall is mathematically capped by $\frac{\text{RETRIEVAL\_TOP\_K}}{|R|}$.
6. **Average Precision Normalization**:
   > **[ENGINEERING-DECISION: RETRIEVAL-AP-NORMALIZATION] (DEC-051)**: Average Precision (AP) for each question is normalized by the **total number of ground-truth relevant rows** $|R|$ rather than the number of retrieved relevant rows:
   > $$AP = \frac{1}{|R|} \sum_{k=1}^N P@k \cdot \text{rel}(k)$$
   > where $N$ is the number of retrieved rows, $P@k$ is precision at rank $k$, and $\text{rel}(k) \in \{0, 1\}$.
7. **Exclusion of Questions without Relevant Rows**: Questions that have 0 relevant corpus rows in the dataset are strictly excluded from retrieval metric computation and counted separately as `excluded_no_relevant_rows`. The exact number of evaluated questions is returned alongside every metric (`num_questions`).

---

## 3. Answer Extraction and Question-Level Metrics

Answer extraction is performed by the `AnswerExtractor` agent (`backend/app/agents/answer_extractor.py`) using only `SUPPORTED` and `PARTIALLY_SUPPORTED` claims from verified clinical reports.

### 3.1 Dual-View Aggregation
To evaluate performance transparently in the presence of explicit abstention, every question-level metric returns two distinct perspectives:
* **Answered-Only View**: Metrics calculated strictly over questions that the workflow attempted to answer (excluding workflow abstentions and extractor abstentions).
* **All-Questions View**: Metrics calculated across all submitted questions, where abstained and failed questions receive zero credit (lowering overall accuracy, recall, and F1).

### 3.2 Question Types
* **Yes/No (`yesno`)**:
  - `accuracy`: Fraction of correct predictions over evaluated questions.
  - `macro_f1`: Unweighted arithmetic mean of F1 for class "yes" and F1 for class "no":
    $$\text{Macro-F1} = \frac{F_{1,\text{yes}} + F_{1,\text{no}}}{2}$$
* **Factoid (`factoid`)**:
  - Ground-truth exact answers are treated as acceptable surface forms of the one correct answer (`[ENGINEERING-DECISION: FACTOID-GOLD-SURFACE-FORMS]` from SPEC 9.9).
  - `strict_accuracy`: Rank 1 prediction matches any gold surface form after text normalization.
  - `lenient_accuracy`: Any prediction in the top 5 matches any gold surface form.
  - `mrr`: Mean Reciprocal Rank ($1 / \text{rank}$ for first matching prediction in top 5, 0 if none).
* **List (`list`)**:
  - Gold ground truth consists of entities with synonym sets ($[[\text{syn}_1, \text{syn}_2], \dots]$).
  - A predicted item matches an entity if normalized prediction matches any synonym of that entity.
  - Each gold entity may be matched at most once; duplicate predictions count as extra, penalizing precision.
  - Computes per-question Precision, Recall, and F1, reporting the mean across questions.
* **Summary (`summary`)**:
  - Per DEC-050 and SPEC 9.9, no unpinned external ROUGE dependency is added in Phase 5 Stage C. Summary questions are designated `evaluated: false` with the explicit rationale recorded in the output payload.

### 3.3 Text Normalization
All answer string matching is preceded by deterministic normalization (`normalize_text`):
1. Unicode NFKC normalization (`unicodedata.normalize("NFKC", text)`).
2. Case folding to lowercase.
3. Complete stripping of ASCII punctuation (`string.punctuation`).
4. Collapsing of contiguous whitespace.

---

## 4. Grounding Metrics

* **`gold_document_citation_rate`**:
  Calculated strictly over answered questions:
  $$\text{gold\_document\_citation\_rate} = \frac{\text{Answered questions citing } \ge 1 \text{ gold PMID}}{\text{Total answered questions}}$$
  Pure function over PMID sets; reported separately from claim-level citation correctness.

---

## 5. Limitations (SPEC Section 9.10)

1. **Closed-World Snippet Corpus**: The reference corpus consists of annotator-selected snippets from BioASQ, meaning answer-bearing passages are contained within the search space. Performance is not comparable to open-corpus retrieval.
2. **Potential Pretraining Contamination**: BioASQ training sets are publicly available on the internet; model performance may partially reflect memorization from pretraining rather than pure clinical reasoning.
3. **Data Provenance**: The raw files under `datasets/` represent a processed snapshot of unknown processing provenance.
4. **Research Tool**: Evaluation results represent benchmark measurements on retrospective academic datasets and must **never** be interpreted as evidence of clinical safety or diagnostic efficacy.

---

## 6. Score-Range Recalibration Proposal (Phase 5 Stage D) [PROPOSAL - NOT APPLIED]

### 6.1 Empirical Distribution on `retrieval_dev` (n=1,133)
Using `retrieval_only` (BioBERT question-as-query embeddings against closed-world BioASQ snippet collection `evidence_biobert_mnli_snli_scinli_scitail_mednli_stsb_v1`), retrieval scores were recorded across all 1,133 dev questions. Questions were partitioned into those with at least one relevant passage row in the top 10 (`relevant_in_top_k`, n=979) versus those with no relevant row in top 10 (`no_relevant_in_top_k`, n=154).

| Group | Metric | Count | p5 | p25 | p50 (Median) | p75 | p95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`relevant_in_top_k`** | Top-1 Row Score | 979 | 0.5189 | 0.6304 | 0.7075 | 0.7680 | 0.8565 |
| **`relevant_in_top_k`** | Mean Top-3 Row Score | 979 | 0.4963 | 0.6025 | 0.6749 | 0.7399 | 0.8190 |
| **`no_relevant_in_top_k`** | Top-1 Row Score | 154 | 0.4155 | 0.5495 | 0.6131 | 0.6817 | 0.7999 |
| **`no_relevant_in_top_k`** | Mean Top-3 Row Score | 154 | 0.4088 | 0.5273 | 0.5916 | 0.6651 | 0.7743 |

### 6.2 Proposed Settings and Rationale
* **`RETRIEVAL_SCORE_FLOOR`**: Propose **0.50** (Current default: 0.30).
  * *Rationale*: In questions where relevant evidence is found, the 5th percentile of `mean_top3` is 0.4963 (~0.50). Any retrieval set scoring below 0.50 has less than a 5% probability of containing relevant evidence. Setting the floor at 0.50 anchors 0.0 retrieval strength where evidence is empirically absent, avoiding artificial inflation of confidence scores.
* **`RETRIEVAL_SCORE_CEILING`**: Propose **0.82** (Current default: 0.90).
  * *Rationale*: The 95th percentile of `mean_top3` among successful retrievals is 0.8190 (~0.82). BioBERT cosine similarities rarely exceed 0.86; a ceiling of 0.90 forces genuinely strong retrievals to peak around ~0.70 retrieval strength. A ceiling of 0.82 properly maps top-tier evidence grounding to 1.0 retrieval strength.
* **`RETRIEVAL_MIN_SCORE`**: Propose **0.50** (Current default: 0.30).
  * *Rationale*: Prevents noise passages scoring between 0.30 and 0.49 from satisfying the `MIN_RELEVANT_PASSAGES` constraint for gate rule 4.
* **Limitation**:
  * End-to-end workflow queries originate from the Clinical Text Clarifier and Symptom RAG Analyzer (structured symptom terms and refined query strings), which may yield different similarity score distributions than direct question-as-query evaluation. Human confirmation is required prior to applying these thresholds to evaluation or production runs.
* **Status**: **PROPOSAL ONLY**. No settings in code or environment configuration have been changed.


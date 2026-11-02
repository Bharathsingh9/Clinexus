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

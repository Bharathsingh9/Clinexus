# Clinexus Major Project Report (\LaTeX{})

This directory contains the complete \LaTeX{} source code for the Major Project Part 1 Report:

- **Project Title**: **CLINEXUS: An Explainable Multi-Agent Clinical Intelligence Platform for Clinical Decision Assistance**
- **Institution**: Department of Artificial Intelligence and Data Science, Chaitanya Bharathi Institute of Technology (A), Hyderabad – 500075.
- **Academic Year**: 2026–2027 (November 2026).
- **Target Length**: 45–50 pages (12pt, 1.5 line spacing, standard university margins).

## Files
- `clinexus_report.tex`: The master standalone \LaTeX{} source file containing all front matter, chapters, tables, figures, algorithms, code listings, and bibliography.

## How to Compile

### Option 1: Overleaf
1. Create a new blank project on [Overleaf](https://www.overleaf.com/).
2. Upload `clinexus_report.tex`.
3. Set the compiler to **pdfLaTeX** (default) or **XeLaTeX**.
4. Click **Recompile** (compile twice to ensure Table of Contents and cross-references resolve).

### Option 2: Local \LaTeX{} Distribution (TeX Live / MiKTeX)
Run the following commands in PowerShell or Terminal:
```bash
pdflatex -interaction=nonstopmode clinexus_report.tex
bibtex clinexus_report   # (if using external bib file; embedded thebibliography compiles directly with pdflatex)
pdflatex -interaction=nonstopmode clinexus_report.tex
pdflatex -interaction=nonstopmode clinexus_report.tex
```

## Report Structure
1. **Title Page**: Formatted according to CBIT Department of AI&DS guidelines.
2. **Institute & Department Vision and Mission**: Exact institutional text.
3. **Declaration & Bonafide Certificates**: Ready for project associate signatures and faculty guide approval.
4. **Acknowledgements**: Includes explicit AI tool disclosure note.
5. **Table of Contents, List of Tables, List of Figures**.
6. **Abbreviations & Mathematical Symbols**.
7. **Sustainable Development Goals (SDGs) Mapping**: SDGs 3, 9, 10, and 16.
8. **Abstract & Keywords**.
9. **Chapter 1: Introduction**: Background, Problem Statement, Motivation, Objectives O1–O6, Research Questions RQ1–RQ4, Scope \& Delimitations.
10. **Chapter 2: Literature Survey**: PRISMA 2020 Review, Biomedical RAG, Multi-Agent CDSS, Claim Verification, Provenance, Systematic Comparative Capability Table (24 systems), Research Gaps G1–G6.
11. **Chapter 3: Methodology and Architecture**: Layered multi-agent architecture, the 5 core agents, the 4 approved extensions (EXT-A to EXT-D), Safety Gate Truth Table, Relational Provenance, Mathematical Confidence formulation, Algorithms 1 & 2.
12. **Chapter 4: Performance Evaluation and Results**: Experimental setup, BioASQ benchmark (75.9k corpus, 5.7k questions, smoke30), Golden Clinical Case trace, Ablation studies (5 pipeline variants), retrieval metrics (Recall@k, MAP, MRR), system latency profiles, adversarial security suite.
13. **Chapter 5: Conclusion**.
14. **Chapter 6: Future Scope**: Part 2 roadmap, EHR/FHIR integration, multimodal imaging agents, prospective clinical trials.
15. **Appendix A: Code Attachments**: Key excerpts from `safety/gate.py`, `claim_verification_agent.py`, and `provenance/service.py`.
16. **References**: 28 comprehensive peer-reviewed references matching IEEE/ACM standards.

"""Static Structural Leakage Guard Test (Phase 5 Stage C).

Specification Section 9.7 (Req ID: EVL-5):
- Statically inspects the Abstract Syntax Tree (AST) of every module under:
    * backend/app/agents
    * backend/app/retrieval
    * backend/app/orchestration
    * backend/app/safety
    * backend/app/ingestion
    * backend/app/provenance
- Fails if ANY module imports from backend.app.evaluation or references
  EvaluationLabel (the DB model) or BioASQGoldLabel (the gold label schema).
- Invariant: The ONLY allowed consumers of gold labels are modules under
  backend/app/evaluation and scripts/evaluate_bioasq.py.
"""

import ast
from pathlib import Path
from typing import List, Tuple
import pytest


PROTECTED_DIRECTORIES = [
    Path("backend/app/agents"),
    Path("backend/app/retrieval"),
    Path("backend/app/orchestration"),
    Path("backend/app/safety"),
    Path("backend/app/ingestion"),
    Path("backend/app/provenance"),
]

FORBIDDEN_MODULES = [
    "backend.app.evaluation",
    "backend.app.evaluation.bioasq_scoring",
    "backend.app.evaluation.bioasq_questions",
    "backend.app.evaluation.splits",
]

FORBIDDEN_SYMBOLS = {
    "EvaluationLabel",
    "BioASQGoldLabel",
}


def scan_file_for_leakage_violations(file_path: Path) -> List[str]:
    """Parse a python file with AST and detect unauthorized imports or symbol references."""
    violations: List[str] = []
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))

    for node in ast.walk(tree):
        # 1. Direct imports: import backend.app.evaluation
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden_mod in FORBIDDEN_MODULES:
                    if alias.name == forbidden_mod or alias.name.startswith(forbidden_mod + "."):
                        violations.append(
                            f"{file_path}:{node.lineno} - Disallowed direct import of '{alias.name}'"
                        )
                if alias.name in FORBIDDEN_SYMBOLS:
                    violations.append(
                        f"{file_path}:{node.lineno} - Disallowed import of symbol '{alias.name}'"
                    )

        # 2. From imports: from backend.app.evaluation import ... or from ... import EvaluationLabel
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for forbidden_mod in FORBIDDEN_MODULES:
                if mod == forbidden_mod or mod.startswith(forbidden_mod + "."):
                    violations.append(
                        f"{file_path}:{node.lineno} - Disallowed from-import from '{mod}'"
                    )
            for alias in node.names:
                if alias.name in FORBIDDEN_SYMBOLS:
                    violations.append(
                        f"{file_path}:{node.lineno} - Disallowed import of symbol '{alias.name}' from '{mod}'"
                    )

        # 3. Symbol usage: Name nodes
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_SYMBOLS:
                violations.append(
                    f"{file_path}:{node.lineno} - Disallowed reference to symbol '{node.id}'"
                )

        # 4. Attribute usage: e.g. models.EvaluationLabel
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_SYMBOLS:
                violations.append(
                    f"{file_path}:{node.lineno} - Disallowed attribute reference to '{node.attr}'"
                )

    return violations


def test_structural_leakage_guard_all_protected_modules():
    """Statically verify zero evaluation imports or gold-label references in operational packages."""
    all_violations: List[str] = []
    scanned_file_count = 0

    for directory in PROTECTED_DIRECTORIES:
        assert directory.exists(), f"Protected directory does not exist: {directory}"
        py_files = list(directory.rglob("*.py"))
        for py_file in py_files:
            # Skip compiled or cache files
            if "__pycache__" in py_file.parts:
                continue
            scanned_file_count += 1
            file_violations = scan_file_for_leakage_violations(py_file)
            all_violations.extend(file_violations)

    # Must scan a non-trivial number of operational modules
    assert scanned_file_count >= 15, f"Expected at least 15 files scanned; found {scanned_file_count}"

    # Assert ZERO structural leakage violations
    if all_violations:
        err_msg = "Structural leakage guard failed:\n" + "\n".join(all_violations)
        pytest.fail(err_msg)


def test_structural_leakage_guard_synthetic_violation_detection(tmp_path: Path):
    """Verify that the scanner positively detects synthetic violations."""
    bad_code = """
from backend.app.evaluation.bioasq_scoring import score_yesno
from backend.app.persistence.models import EvaluationLabel

def helper(label: EvaluationLabel):
    return BioASQGoldLabel(sample_id="123")
"""
    test_file = tmp_path / "bad_agent.py"
    test_file.write_text(bad_code, encoding="utf-8")

    violations = scan_file_for_leakage_violations(test_file)
    assert len(violations) >= 3
    assert any("bioasq_scoring" in v for v in violations)
    assert any("EvaluationLabel" in v for v in violations)
    assert any("BioASQGoldLabel" in v for v in violations)

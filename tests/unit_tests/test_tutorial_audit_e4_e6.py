"""Unit tests for the static E.4 (engagement) and E.6 (Diataxis) validators.

The tests use small inline tutorial sources written to a tmp_path so the
validators see real ``plot_*.py`` files as in production. Each test exercises
either a positive (no finding emitted) or negative (expected finding emitted)
case for one rule. Reviewer-only stubs are exercised separately to confirm
they always return a single ``info`` Finding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest

from scripts.tutorial_audit.api import Finding
from scripts.tutorial_audit.static import e4_engagement, e6_diataxis

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_tutorial(tmp_path: Path, name: str, source: str) -> Path:
    """Persist ``source`` as a ``plot_*.py`` file under ``tmp_path``."""
    target = tmp_path / name
    target.write_text(source, encoding="utf-8")
    return target


def _has_rule(
    findings: Iterable[Finding], rule_id: str, level: str | None = None
) -> bool:
    for f in findings:
        if f.rule_id != rule_id:
            continue
        if level is not None and f.level != level:
            continue
        return True
    return False


# Reusable building blocks -- combined per test to keep fixtures explicit.
TITLE_AND_OPENING_QUESTION = '''"""=========================
Title placeholder
=========================

Why does subject leakage in EEG decoding matter?
"""
'''

TITLE_AND_OPENING_NO_QUESTION = '''"""=========================
Title placeholder
=========================

This example shows how to load a recording.
"""
'''

CODE_CELL = "\n# %%\nimport numpy as np\n"


# ---------------------------------------------------------------------------
# E4.32 -- check_dataset_citation
# ---------------------------------------------------------------------------


def test_e4_32_doi_present_no_finding(tmp_path: Path) -> None:
    src = (
        TITLE_AND_OPENING_QUESTION
        + "\n# %% [markdown]\n"
        + "# This tutorial follows Cisotto & Chicco 2024 (doi:10.7717/peerj-cs.2271).\n"
        + CODE_CELL
    )
    path = _write_tutorial(tmp_path, "plot_with_doi.py", src)
    findings = e4_engagement.check_dataset_citation(path, spec={})
    assert findings == []


def test_e4_32_no_doi_warns_when_spec_has_no_related_papers(tmp_path: Path) -> None:
    src = TITLE_AND_OPENING_QUESTION + CODE_CELL
    path = _write_tutorial(tmp_path, "plot_no_doi.py", src)
    findings = e4_engagement.check_dataset_citation(path, spec={})
    assert _has_rule(findings, "E4.32", level="warn")
    assert not _has_rule(findings, "E4.32", level="error")


def test_e4_32_no_doi_errors_when_spec_lists_related_papers(tmp_path: Path) -> None:
    src = TITLE_AND_OPENING_QUESTION + CODE_CELL
    path = _write_tutorial(tmp_path, "plot_no_doi_related.py", src)
    spec = {
        "links": {
            "related_papers": ["Cisotto & Chicco 2024, PeerJ CS"],
        }
    }
    findings = e4_engagement.check_dataset_citation(path, spec=spec)
    assert _has_rule(findings, "E4.32", level="error")


def test_e4_32_duplicate_dois_warn_for_boilerplate(tmp_path: Path) -> None:
    src = (
        TITLE_AND_OPENING_QUESTION
        + "\n# %% [markdown]\n"
        + "# Cite: 10.7717/peerj-cs.2271\n"
        + "#\n"
        + "# See also 10.7717/peerj-cs.2271 for the full method.\n"
        + CODE_CELL
    )
    path = _write_tutorial(tmp_path, "plot_dup_doi.py", src)
    findings = e4_engagement.check_dataset_citation(path, spec={})
    assert _has_rule(findings, "E4.32", level="warn")
    # And the warning evidence should record the boilerplate detection.
    boilerplate = [
        f
        for f in findings
        if f.rule_id == "E4.32" and "boilerplate" in f.message.lower()
    ]
    assert boilerplate
    assert boilerplate[0].evidence["unique_dois"] == ["10.7717/peerj-cs.2271"]


def test_e4_32_two_distinct_dois_no_finding(tmp_path: Path) -> None:
    src = (
        TITLE_AND_OPENING_QUESTION
        + "\n# %% [markdown]\n"
        + "# Dataset: 10.18112/openneuro.ds002778.v1.0.5\n"
        + "# Method:  10.7717/peerj-cs.2271\n"
        + CODE_CELL
    )
    path = _write_tutorial(tmp_path, "plot_two_doi.py", src)
    findings = e4_engagement.check_dataset_citation(path, spec={})
    assert findings == []


# ---------------------------------------------------------------------------
# E4.34 -- check_extensions_section
# ---------------------------------------------------------------------------


def test_e4_34_extensions_with_three_bullets_no_finding(tmp_path: Path) -> None:
    src = (
        TITLE_AND_OPENING_QUESTION
        + CODE_CELL
        + "\n# %% [markdown]\n"
        + "# ## Try it yourself\n"
        + "#\n"
        + "# - Swap the dataset to ds002778.\n"
        + "# - Reduce the window to 1 second.\n"
        + "# - Increase the number of folds to 10.\n"
    )
    path = _write_tutorial(tmp_path, "plot_extensions_ok.py", src)
    findings = e4_engagement.check_extensions_section(path, spec={})
    assert findings == []


def test_e4_34_missing_section_warns(tmp_path: Path) -> None:
    src = TITLE_AND_OPENING_QUESTION + CODE_CELL
    path = _write_tutorial(tmp_path, "plot_no_extensions.py", src)
    findings = e4_engagement.check_extensions_section(path, spec={})
    assert _has_rule(findings, "E4.34", level="warn")
    assert any("missing" in f.message.lower() for f in findings)


def test_e4_34_too_few_bullets_warns(tmp_path: Path) -> None:
    src = (
        TITLE_AND_OPENING_QUESTION
        + CODE_CELL
        + "\n# %% [markdown]\n"
        + "# ## Extensions\n"
        + "#\n"
        + "# - Try a different dataset.\n"
        + "# - Try a longer window.\n"
    )
    path = _write_tutorial(tmp_path, "plot_two_bullets.py", src)
    findings = e4_engagement.check_extensions_section(path, spec={})
    assert _has_rule(findings, "E4.34", level="warn")
    bullets = [f for f in findings if f.rule_id == "E4.34"]
    assert bullets[0].evidence["bullet_count"] == 2


def test_e4_34_rest_underline_header_accepted(tmp_path: Path) -> None:
    src = (
        TITLE_AND_OPENING_QUESTION
        + CODE_CELL
        + "\n# %% [markdown]\n"
        + "# Extensions\n"
        + "# ----------\n"
        + "#\n"
        + "# - Modification one.\n"
        + "# - Modification two.\n"
        + "# - Modification three.\n"
    )
    path = _write_tutorial(tmp_path, "plot_rest_extensions.py", src)
    findings = e4_engagement.check_extensions_section(path, spec={})
    assert findings == []


# ---------------------------------------------------------------------------
# E4.31 -- check_motivating_question_present (machine-checkable subset)
# ---------------------------------------------------------------------------


def test_e4_31_question_present_emits_only_reviewer_info(tmp_path: Path) -> None:
    src = TITLE_AND_OPENING_QUESTION + CODE_CELL
    path = _write_tutorial(tmp_path, "plot_question.py", src)
    findings = e4_engagement.check_motivating_question_present(path, spec={})
    # Exactly one finding -- the reviewer-only info notice.
    assert len(findings) == 1
    assert findings[0].rule_id == "E4.31"
    assert findings[0].level == "info"
    assert findings[0].evidence.get("reviewer_only") is True


def test_e4_31_no_question_emits_warn_and_info(tmp_path: Path) -> None:
    src = TITLE_AND_OPENING_NO_QUESTION + CODE_CELL
    path = _write_tutorial(tmp_path, "plot_no_question.py", src)
    findings = e4_engagement.check_motivating_question_present(path, spec={})
    assert _has_rule(findings, "E4.31", level="warn")
    assert _has_rule(findings, "E4.31", level="info")


# ---------------------------------------------------------------------------
# E4.33 / E4.35 -- reviewer-only stubs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fn", "rule_id"),
    [
        (e4_engagement.check_result_meaningful, "E4.33"),
        (e4_engagement.check_tone, "E4.35"),
    ],
)
def test_e4_reviewer_stubs_emit_single_info(tmp_path: Path, fn, rule_id: str) -> None:
    src = TITLE_AND_OPENING_QUESTION + CODE_CELL
    path = _write_tutorial(tmp_path, "plot_stub.py", src)
    findings = fn(path, spec={})
    assert len(findings) == 1
    assert findings[0].rule_id == rule_id
    assert findings[0].level == "info"
    assert findings[0].evidence.get("reviewer_only") is True
    # Reviewer stubs must point reviewers at the strategy doc.
    assert "reviewer" in findings[0].message.lower()


# ---------------------------------------------------------------------------
# E6.48 -- check_concept_link_present
# ---------------------------------------------------------------------------


def test_e6_48_doc_role_link_no_finding(tmp_path: Path) -> None:
    src = (
        TITLE_AND_OPENING_QUESTION
        + CODE_CELL
        + "\n# %% [markdown]\n"
        + "# See :doc:`/concepts/leakage_and_evaluation` for theory.\n"
    )
    path = _write_tutorial(tmp_path, "plot_concept_doc.py", src)
    findings = e6_diataxis.check_concept_link_present(path, spec={})
    assert findings == []


def test_e6_48_explanation_role_link_no_finding(tmp_path: Path) -> None:
    src = (
        TITLE_AND_OPENING_QUESTION
        + CODE_CELL
        + "\n# %% [markdown]\n"
        + "# Read more in :doc:`/explanation/why_montage_matters`.\n"
    )
    path = _write_tutorial(tmp_path, "plot_explanation_doc.py", src)
    findings = e6_diataxis.check_concept_link_present(path, spec={})
    assert findings == []


def test_e6_48_markdown_link_no_finding(tmp_path: Path) -> None:
    src = (
        TITLE_AND_OPENING_QUESTION
        + CODE_CELL
        + "\n# %% [markdown]\n"
        + "# See [the leakage explainer](docs/source/concepts/leakage.rst).\n"
    )
    path = _write_tutorial(tmp_path, "plot_md_concept.py", src)
    findings = e6_diataxis.check_concept_link_present(path, spec={})
    assert findings == []


def test_e6_48_missing_link_warns(tmp_path: Path) -> None:
    src = (
        TITLE_AND_OPENING_QUESTION
        + CODE_CELL
        + "\n# %% [markdown]\n"
        + "# Just prose; no cross-reference to an explanation page.\n"
    )
    path = _write_tutorial(tmp_path, "plot_no_concept.py", src)
    findings = e6_diataxis.check_concept_link_present(path, spec={})
    assert _has_rule(findings, "E6.48", level="warn")


def test_e6_48_unrelated_doc_role_does_not_satisfy(tmp_path: Path) -> None:
    """A :doc: link into ``how_to/`` or ``api/`` must not satisfy E6.48."""
    src = (
        TITLE_AND_OPENING_QUESTION
        + CODE_CELL
        + "\n# %% [markdown]\n"
        + "# See :doc:`/how_to/handle_bad_records` for a recipe.\n"
    )
    path = _write_tutorial(tmp_path, "plot_how_to_only.py", src)
    findings = e6_diataxis.check_concept_link_present(path, spec={})
    assert _has_rule(findings, "E6.48", level="warn")


# ---------------------------------------------------------------------------
# E6.47 / E6.49 -- reviewer-only stubs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fn", "rule_id"),
    [
        (e6_diataxis.check_diataxis_purity, "E6.47"),
        (e6_diataxis.check_how_to_extraction, "E6.49"),
    ],
)
def test_e6_reviewer_stubs_emit_single_info(tmp_path: Path, fn, rule_id: str) -> None:
    src = TITLE_AND_OPENING_QUESTION + CODE_CELL
    path = _write_tutorial(tmp_path, "plot_stub_e6.py", src)
    findings = fn(path, spec={})
    assert len(findings) == 1
    assert findings[0].rule_id == rule_id
    assert findings[0].level == "info"
    assert findings[0].evidence.get("reviewer_only") is True
    assert "reviewer" in findings[0].message.lower()


# ---------------------------------------------------------------------------
# Cross-cutting -- every active finding carries valid citations
# ---------------------------------------------------------------------------


def test_active_findings_carry_required_citations(tmp_path: Path) -> None:
    """Every Finding must carry ``cite_rubric`` and ``cite_plan`` strings."""
    src = TITLE_AND_OPENING_NO_QUESTION + CODE_CELL  # triggers many findings
    path = _write_tutorial(tmp_path, "plot_citations.py", src)
    fns = [
        e4_engagement.check_dataset_citation,
        e4_engagement.check_extensions_section,
        e4_engagement.check_motivating_question_present,
        e4_engagement.check_result_meaningful,
        e4_engagement.check_tone,
        e6_diataxis.check_concept_link_present,
        e6_diataxis.check_diataxis_purity,
        e6_diataxis.check_how_to_extraction,
    ]
    seen_any = False
    for fn in fns:
        for finding in fn(path, spec={}):
            seen_any = True
            assert finding.cite_rubric.startswith("compass_artifact.md#")
            assert finding.cite_plan.startswith("tutorial_restructure_plan.md#")
            # Forbidden citation per task spec.
            assert "data-viz-design.md#viz_compliance" not in finding.cite_plan
    assert seen_any, "Expected at least one finding from the negative-case fixture"

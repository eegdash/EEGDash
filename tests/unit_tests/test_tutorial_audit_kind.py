"""Unit tests for the ``spec.kind``-aware branching of the tutorial audit.

Tutorials and how-tos live in different Diataxis quadrants and have
different layout contracts. The static validators in
``scripts.tutorial_audit.static`` branch on ``spec.kind`` so a how-to is
not penalised by tutorial-only rules (PRIMM scaffolding, "Extensions"
wrap-up, motivating question, learning-objective bullet list, and the
``plot_*.py`` filename). These tests cover the four most load-bearing
branches:

* E1.1.howto -- the how-to filename rule.
* E2.20.howto -- the how-to ``## Goal`` section rule (Python source).
* E4.34.howto -- the how-to ``## Common pitfalls`` wrap-up rule.
* Markdown source handling -- ``output_kind: markdown`` how-tos must not
  crash the AST-based validators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from scripts.tutorial_audit.api import Finding
from scripts.tutorial_audit.static import (
    e1_structural,
    e2_pedagogical,
    e4_engagement,
)


def _write(tmp_path: Path, name: str, source: str) -> Path:
    """Persist ``source`` to ``tmp_path/name`` and return the path."""
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


HOW_TO_PY_HEADER = '''"""How-to: stage data on a fast filesystem
=========================================

Recipe for staging EEGDash data onto a per-node SSD before training.
"""

# %% [markdown]
# ## Goal
#
# Resolve ``cache_dir`` from a SLURM env var and stage data once.
'''


HOW_TO_PY_NO_GOAL = '''"""How-to: stage data on a fast filesystem
=========================================

Recipe for staging EEGDash data onto a per-node SSD before training.
"""

# %% [markdown]
# ## Overview
#
# Resolve ``cache_dir`` from a SLURM env var and stage data once.
'''


HOW_TO_PY_WITH_PITFALLS = HOW_TO_PY_HEADER + (
    "\n# %% [markdown]\n"
    "# ## Common pitfalls\n"
    "#\n"
    "# - Hard-coded paths break across nodes.\n"
    "# - Filtering after download wastes bandwidth.\n"
    "# - Stale partial caches must be purged before re-run.\n"
)


HOW_TO_MD = """# How to run a SLURM job

**Goal.** Copy the .slurm template, edit four variables, and submit.

## Prerequisites

- A SLURM cluster.

## Common pitfalls

- Hard-coded ``$HOME`` paths.
- Missing module activation inside the job script.
- Logs colliding when ``%j`` is used instead of ``%A_%a``.
"""


# ---------------------------------------------------------------------------
# E1.1.howto -- filename branch
# ---------------------------------------------------------------------------


def test_e1_1_howto_python_filename_passes(tmp_path: Path) -> None:
    """A how-to .py file named ``how_to_*.py`` is accepted."""
    path = _write(tmp_path, "how_to_stage_cache.py", HOW_TO_PY_HEADER)
    spec = {"kind": "how-to", "output_kind": "python"}
    assert e1_structural.check_filename(path, spec) == []


def test_e1_1_howto_markdown_filename_passes(tmp_path: Path) -> None:
    """A how-to .md file named ``how_to_*.md`` is accepted."""
    path = _write(tmp_path, "how_to_run_on_slurm.md", HOW_TO_MD)
    spec = {"kind": "how-to", "output_kind": "markdown"}
    assert e1_structural.check_filename(path, spec) == []


def test_e1_1_tutorial_still_fails_on_how_to_filename(tmp_path: Path) -> None:
    """A spec with ``kind: tutorial`` still fails E1.1 when the file is
    ``how_to_*.py`` -- the legacy rule fires unchanged for tutorials.
    """
    path = _write(tmp_path, "how_to_misclassified.py", HOW_TO_PY_HEADER)
    spec = {"kind": "tutorial"}
    findings = e1_structural.check_filename(path, spec)
    assert _has_rule(findings, "E1.1", level="error")
    assert not _has_rule(findings, "E1.1.howto")


def test_e1_1_howto_wrong_extension_fails(tmp_path: Path) -> None:
    """A how-to whose ``output_kind`` is ``markdown`` must end in ``.md``."""
    path = _write(tmp_path, "how_to_wrong.py", HOW_TO_PY_HEADER)
    spec = {"kind": "how-to", "output_kind": "markdown"}
    findings = e1_structural.check_filename(path, spec)
    assert _has_rule(findings, "E1.1.howto", level="error")


# ---------------------------------------------------------------------------
# E2.20.howto -- Goal section branch
# ---------------------------------------------------------------------------


def test_e2_20_howto_with_goal_section_passes(tmp_path: Path) -> None:
    """A how-to .py file with a ``## Goal`` markdown header passes."""
    path = _write(tmp_path, "how_to_stage_cache.py", HOW_TO_PY_HEADER)
    spec = {"kind": "how-to", "output_kind": "python"}
    findings = e2_pedagogical.check_learning_objectives(path, spec)
    assert findings == []


def test_e2_20_howto_without_goal_section_fails(tmp_path: Path) -> None:
    """A how-to .py file with no ``## Goal`` (or equivalent) header fails."""
    path = _write(tmp_path, "how_to_stage_cache.py", HOW_TO_PY_NO_GOAL)
    spec = {"kind": "how-to", "output_kind": "python"}
    findings = e2_pedagogical.check_learning_objectives(path, spec)
    assert _has_rule(findings, "E2.20.howto", level="error")
    assert not _has_rule(findings, "E2.20")


def test_e2_20_howto_markdown_with_inline_goal_label_passes(tmp_path: Path) -> None:
    """A markdown how-to with an inline ``**Goal.** ...`` paragraph is accepted."""
    path = _write(tmp_path, "how_to_run_on_slurm.md", HOW_TO_MD)
    spec = {"kind": "how-to", "output_kind": "markdown"}
    findings = e2_pedagogical.check_learning_objectives(path, spec)
    assert findings == []


def test_e2_20_tutorial_still_demands_learning_objectives(tmp_path: Path) -> None:
    """A spec with ``kind: tutorial`` still fires E2.20 when the LO header is
    absent. The new ``E2.20.howto`` rule must not fire for tutorials.
    """
    src = '''"""Tutorial title
================

Why this matters.
"""

# %% [markdown]
# ## Setup
#
# - Step one.
# - Step two.
'''
    path = _write(tmp_path, "plot_no_lo.py", src)
    spec = {"kind": "tutorial"}
    findings = e2_pedagogical.check_learning_objectives(path, spec)
    assert _has_rule(findings, "E2.20", level="error")
    assert not _has_rule(findings, "E2.20.howto")


# ---------------------------------------------------------------------------
# E4.31 -- motivating question check is skipped for how-tos
# ---------------------------------------------------------------------------


def test_e4_31_skipped_for_how_to(tmp_path: Path) -> None:
    """How-tos open with a Goal, not a question, so E4.31 is info-only."""
    path = _write(tmp_path, "how_to_stage_cache.py", HOW_TO_PY_HEADER)
    spec = {"kind": "how-to", "output_kind": "python"}
    findings = e4_engagement.check_motivating_question_present(path, spec)
    # Exactly one info finding noting the deliberate skip.
    assert len(findings) == 1
    assert findings[0].rule_id == "E4.31"
    assert findings[0].level == "info"
    assert findings[0].evidence.get("skipped") == "how-to"


# ---------------------------------------------------------------------------
# E4.34.howto -- Common pitfalls branch
# ---------------------------------------------------------------------------


def test_e4_34_howto_with_three_pitfalls_passes(tmp_path: Path) -> None:
    """A how-to with three pitfall bullets satisfies E4.34.howto."""
    path = _write(tmp_path, "how_to_stage_cache.py", HOW_TO_PY_WITH_PITFALLS)
    spec = {"kind": "how-to", "output_kind": "python"}
    findings = e4_engagement.check_extensions_section(path, spec)
    assert findings == []


def test_e4_34_howto_missing_pitfalls_warns(tmp_path: Path) -> None:
    """A how-to with no ``## Common pitfalls`` section warns on E4.34.howto."""
    path = _write(tmp_path, "how_to_stage_cache.py", HOW_TO_PY_HEADER)
    spec = {"kind": "how-to", "output_kind": "python"}
    findings = e4_engagement.check_extensions_section(path, spec)
    assert _has_rule(findings, "E4.34.howto", level="warn")
    assert not _has_rule(findings, "E4.34")


def test_e4_34_tutorial_still_demands_extensions(tmp_path: Path) -> None:
    """Tutorials still fire E4.34 (not E4.34.howto) when Extensions is absent."""
    src = '''"""Tutorial title
================

Why this matters?
"""

# %% [markdown]
# ## Setup
#
# Some prose.
'''
    path = _write(tmp_path, "plot_no_extensions.py", src)
    spec = {"kind": "tutorial"}
    findings = e4_engagement.check_extensions_section(path, spec)
    assert _has_rule(findings, "E4.34", level="warn")
    assert not _has_rule(findings, "E4.34.howto")


# ---------------------------------------------------------------------------
# Markdown how-to source: no AST parser crash
# ---------------------------------------------------------------------------


def test_markdown_howto_does_not_crash_ast_validators(tmp_path: Path) -> None:
    """``output_kind: markdown`` how-tos must not crash the AST-based checks.

    The Markdown source is not valid Python, so naive ``ast.parse`` would
    raise. The kind-aware short-circuit returns an empty (or info-only)
    finding list instead.
    """
    path = _write(tmp_path, "how_to_run_on_slurm.md", HOW_TO_MD)
    spec = {"kind": "how-to", "output_kind": "markdown"}

    # E1.2 docstring check skips with an info-level finding.
    docstring_findings = e1_structural.check_docstring_header(path, spec)
    assert len(docstring_findings) == 1
    assert docstring_findings[0].level == "info"
    assert docstring_findings[0].rule_id == "E1.2"

    # E1.4 block-delimiter check is exempt for markdown sources.
    assert e1_structural.check_block_delimiters(path, spec) == []

    # E2.20.howto Goal check works on Markdown sources.
    assert e2_pedagogical.check_learning_objectives(path, spec) == []

    # E4.31 emits a single info-level "skipped: how-to" finding.
    e4_31 = e4_engagement.check_motivating_question_present(path, spec)
    assert len(e4_31) == 1
    assert e4_31[0].level == "info"

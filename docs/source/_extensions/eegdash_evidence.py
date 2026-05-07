"""Sphinx extension exposing the ``eegdash-evidence-dashboard`` directive.

The directive aggregates the per-tutorial evidence dossiers stored in
``docs/evidence/tutorials/<id>/evidence.json`` (and the optional
``reviewer_score.json``) and renders them as a styled HTML table on the
public Tutorial Evidence page.

The extension intentionally has no third-party dependencies beyond Sphinx
itself: it reads YAML by hand (only a tiny subset of fields are needed —
``id``, ``title``, ``category``, ``difficulty``, ``state``) and JSON via
the standard library so the docs build never has to import the runtime
``eegdash`` package machinery.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util import logging

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #

# This file lives at ``docs/source/_extensions/eegdash_evidence.py``; the
# repository root is therefore three directories up.
EXT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXT_DIR.parents[2]
EVIDENCE_ROOT = REPO_ROOT / "docs" / "evidence" / "tutorials"
SPEC_ROOT = REPO_ROOT / "docs" / "tutorials" / "_spec"


# Map the seven category codes (A through I, with letters G/J skipped) to
# the gallery subdirectory the rendered tutorial lives in. Categories are
# stable, defined by ``docs/tutorials/_spec/<id>.yaml``.
CATEGORY_TO_GALLERY: dict[str, str] = {
    "A-start-here": "tutorials/00_start_here",
    "B-core-workflow": "tutorials/10_core_workflow",
    "C-event-related": "tutorials/20_event_related",
    "D-resting-state": "tutorials/30_resting_state",
    "E-feature-engineering": "tutorials/40_features",
    "F-evaluation": "tutorials/50_evaluation",
    "H-transfer-foundation": "tutorials/70_transfer_foundation",
    "I-scaling-hpc": "how_to",
}

CATEGORY_LABEL: dict[str, str] = {
    "A-start-here": "A start-here",
    "B-core-workflow": "B core",
    "C-event-related": "C event",
    "D-resting-state": "D resting",
    "E-feature-engineering": "E features",
    "F-evaluation": "F evaluation",
    "H-transfer-foundation": "H transfer",
    "I-scaling-hpc": "I scaling",
}


# Allowed states (also drives the badge CSS class). The leading hyphen makes
# them safe HTML class names: ``evidence-badge state-<key>``.
KNOWN_STATES = (
    "proposed",
    "drafted",
    "audited",
    "reviewed",
    "merged",
    "deprecated",
)


# --------------------------------------------------------------------------- #
# YAML / JSON readers
# --------------------------------------------------------------------------- #


_YAML_LINE = re.compile(
    r"^(?P<key>id|title|category|difficulty|state|kind)\s*:\s*(?P<val>.*?)\s*$"
)


def _strip_yaml_value(raw: str) -> str:
    """Strip surrounding quotes from a scalar YAML value."""
    raw = raw.strip()
    if (raw.startswith("'") and raw.endswith("'")) or (
        raw.startswith('"') and raw.endswith('"')
    ):
        return raw[1:-1]
    return raw


def _read_spec(tutorial_id: str) -> dict[str, str]:
    """Read the top-level scalar fields of a tutorial spec YAML.

    A full YAML parser is deliberately not pulled in: the spec format keeps
    the values we care about (``id``, ``title``, ``category``,
    ``difficulty``, ``state``, ``kind``) on top-level scalar lines. Anything
    we cannot parse falls back to ``"unknown"`` so the table still renders.
    """
    spec_path = SPEC_ROOT / f"{tutorial_id}.yaml"
    out: dict[str, str] = {
        "id": tutorial_id,
        "title": tutorial_id,
        "category": "unknown",
        "difficulty": "0",
        "state": "unknown",
        "kind": "tutorial",
    }
    if not spec_path.exists():
        return out
    try:
        text = spec_path.read_text(encoding="utf-8")
    except OSError as exc:
        LOGGER.warning("eegdash-evidence: cannot read spec %s: %s", spec_path, exc)
        return out
    for raw_line in text.splitlines():
        # Stop at the first nested mapping so we don't consume the same key
        # name from a nested block (e.g. ``links:\n  api: …``).
        if raw_line.startswith(" ") or raw_line.startswith("\t"):
            continue
        match = _YAML_LINE.match(raw_line)
        if match is None:
            continue
        key = match.group("key")
        if key in out:
            out[key] = _strip_yaml_value(match.group("val"))
    return out


def _read_evidence(tutorial_id: str) -> dict | None:
    """Read ``docs/evidence/tutorials/<id>/evidence.json`` or return None."""
    evidence_path = EVIDENCE_ROOT / tutorial_id / "evidence.json"
    if not evidence_path.exists():
        return None
    try:
        return json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("eegdash-evidence: cannot parse %s: %s", evidence_path, exc)
        return None


def _read_reviewer(tutorial_id: str) -> dict | None:
    """Read the optional ``reviewer_score.json`` for a tutorial."""
    reviewer_path = EVIDENCE_ROOT / tutorial_id / "reviewer_score.json"
    if not reviewer_path.exists():
        return None
    try:
        return json.loads(reviewer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("eegdash-evidence: cannot parse %s: %s", reviewer_path, exc)
        return None


def _list_tutorial_ids() -> list[str]:
    """Return the sorted list of tutorial IDs that have an evidence dossier."""
    if not EVIDENCE_ROOT.exists():
        return []
    ids: list[str] = []
    for entry in sorted(EVIDENCE_ROOT.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        if not (entry / "evidence.json").exists():
            continue
        ids.append(entry.name)
    return ids


# --------------------------------------------------------------------------- #
# Row assembly
# --------------------------------------------------------------------------- #


def _difficulty_stars(value: str) -> str:
    """Render difficulty 1/2/3 as filled-versus-hollow stars."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return "n/a"
    n = max(0, min(3, n))
    return "★" * n + "☆" * (3 - n)


def _state_class(state: str) -> str:
    if state in KNOWN_STATES:
        return f"state-{state}"
    return "state-unknown"


def _gallery_link(tutorial_id: str, category: str, kind: str) -> str:
    """Compute the rendered gallery URL for the tutorial.

    Sphinx-Gallery produces ``generated/auto_examples/<gallery>/<id>.html``;
    those pages live under ``/generated/auto_examples/`` once published.
    For how-tos the file name carries no ``plot_`` prefix.
    """
    gallery = CATEGORY_TO_GALLERY.get(category)
    if gallery is None:
        return ""
    return f"/generated/auto_examples/{gallery}/{tutorial_id}.html"


def _make_paragraph(text: str) -> nodes.paragraph:
    p = nodes.paragraph()
    p += nodes.Text(text)
    return p


def _make_link(text: str, url: str) -> nodes.paragraph:
    p = nodes.paragraph()
    if url:
        ref = nodes.reference(refuri=url, text=text)
        p += ref
    else:
        p += nodes.Text(text)
    return p


def _make_badge(label: str, css_class: str) -> nodes.paragraph:
    """Return a paragraph carrying an inline-HTML badge.

    Inline HTML is needed because we want the colour-coded background that
    plain docutils text cannot express. ``raw`` nodes are the supported way
    to emit HTML directly into Sphinx output without breaking other writers
    (the ``format='html'`` argument restricts the snippet to HTML output).
    """
    p = nodes.paragraph()
    safe_label = label.replace("&", "&amp;").replace("<", "&lt;")
    p += nodes.raw(
        "",
        f'<span class="evidence-badge {css_class}">{safe_label}</span>',
        format="html",
    )
    return p


def _empty_state_table() -> nodes.table:
    table = nodes.table(classes=["evidence", "evidence-table"])
    tgroup = nodes.tgroup(cols=1)
    tgroup += nodes.colspec(colwidth=100)
    thead = nodes.thead()
    head_row = nodes.row()
    head_row += _wrap_entry(_make_paragraph("Tutorial evidence"))
    thead += head_row
    tgroup += thead
    tbody = nodes.tbody()
    body_row = nodes.row()
    body_row += _wrap_entry(
        _make_paragraph("no dossiers yet — run `make tutorial-audit` to generate them")
    )
    tbody += body_row
    tgroup += tbody
    table += tgroup
    return table


def _wrap_entry(child: nodes.Node) -> nodes.entry:
    entry = nodes.entry()
    entry += child
    return entry


# --------------------------------------------------------------------------- #
# Directive implementation
# --------------------------------------------------------------------------- #


COLUMNS: tuple[tuple[str, int], ...] = (
    ("Tutorial", 28),
    ("Category", 12),
    ("Difficulty", 8),
    ("State", 10),
    ("Errors", 6),
    ("Warns", 6),
    ("Infos", 6),
    ("Reviewer", 8),
    ("Dossier", 14),
)


class EegdashEvidenceDashboard(Directive):
    has_content = False
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec: dict[str, object] = {}

    def run(self) -> list[nodes.Node]:
        ids = _list_tutorial_ids()
        if not ids:
            return [_empty_state_table()]

        rows: list[dict[str, object]] = []
        for tutorial_id in ids:
            spec = _read_spec(tutorial_id)
            evidence = _read_evidence(tutorial_id) or {}
            reviewer = _read_reviewer(tutorial_id) or {}

            totals = evidence.get("totals", {}) if isinstance(evidence, dict) else {}
            rows.append(
                {
                    "id": tutorial_id,
                    "title": spec.get("title", tutorial_id),
                    "category": spec.get("category", "unknown"),
                    "difficulty": evidence.get("spec_difficulty")
                    or spec.get("difficulty", "0"),
                    "state": evidence.get("spec_state") or spec.get("state", "unknown"),
                    "errors": totals.get("errors", "n/a"),
                    "warns": totals.get("warns", "n/a"),
                    "infos": totals.get("infos", "n/a"),
                    "reviewer_min": reviewer.get("min_score", "n/a")
                    if reviewer
                    else "n/a",
                    "kind": spec.get("kind", "tutorial"),
                }
            )

        return [self._build_table(rows)]

    # -- table ----------------------------------------------------------------

    def _build_table(self, rows: Iterable[dict[str, object]]) -> nodes.table:
        table = nodes.table(classes=["evidence", "evidence-table"])
        tgroup = nodes.tgroup(cols=len(COLUMNS))
        for _, width in COLUMNS:
            tgroup += nodes.colspec(colwidth=width)

        thead = nodes.thead()
        head_row = nodes.row()
        for label, _ in COLUMNS:
            head_row += _wrap_entry(_make_paragraph(label))
        thead += head_row
        tgroup += thead

        tbody = nodes.tbody()
        for row in rows:
            tbody += self._build_row(row)
        tgroup += tbody
        table += tgroup
        return table

    # -- row ------------------------------------------------------------------

    def _build_row(self, row: dict[str, object]) -> nodes.row:
        body = nodes.row()
        category = str(row["category"])
        gallery_url = _gallery_link(str(row["id"]), category, str(row["kind"]))

        # Tutorial column: title (linked) plus the tutorial id underneath
        # in muted text — the id stays useful when scanning audit logs.
        tutorial_cell = nodes.entry()
        tutorial_cell += _make_link(str(row["title"]), gallery_url)
        sub = nodes.paragraph(classes=["evidence-id"])
        sub += nodes.literal(text=str(row["id"]))
        tutorial_cell += sub
        body += tutorial_cell

        body += _wrap_entry(_make_paragraph(CATEGORY_LABEL.get(category, category)))
        body += _wrap_entry(_make_paragraph(_difficulty_stars(str(row["difficulty"]))))
        body += _wrap_entry(
            _make_badge(str(row["state"]), _state_class(str(row["state"])))
        )
        body += _wrap_entry(_make_paragraph(str(row["errors"])))
        body += _wrap_entry(_make_paragraph(str(row["warns"])))
        body += _wrap_entry(_make_paragraph(str(row["infos"])))
        body += _wrap_entry(_make_paragraph(str(row["reviewer_min"])))

        # Dossier link — the JSON evidence files are checked in under
        # ``docs/evidence/tutorials/<id>/`` and surface as raw artefacts on
        # GitHub. We do not link to a Sphinx page because no per-dossier
        # RST stubs are generated at build time yet.
        dossier_url = (
            "https://github.com/eegdash/EEGDash/tree/develop/docs/evidence/"
            f"tutorials/{row['id']}"
        )
        body += _wrap_entry(_make_link("evidence", dossier_url))
        return body


# --------------------------------------------------------------------------- #
# Sphinx wiring
# --------------------------------------------------------------------------- #


def setup(app):
    app.add_directive("eegdash-evidence-dashboard", EegdashEvidenceDashboard)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

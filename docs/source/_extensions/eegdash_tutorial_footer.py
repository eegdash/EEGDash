"""Inject a 'Behind this lesson' footer block on every sphinx-gallery page.

The footer is informational only. It is rendered for HTML pages whose source
lives under one of the configured sphinx-gallery ``gallery_dirs`` (i.e.
auto-generated tutorial pages such as
``generated/auto_examples/tutorials/00_start_here/plot_00_first_search``)
*and* whose ``tutorial_id`` has a corresponding evidence dossier at
``docs/evidence/tutorials/<tutorial_id>/evidence.json``.

If no dossier is present, nothing is emitted (graceful no-op). The extension
must never crash on non-gallery pages such as ``index``, ``concepts/index``,
or autodoc API references.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

# Pagename prefix that flags a sphinx-gallery auto-generated tutorial page.
_GALLERY_PREFIX = "generated/auto_examples/"

# Pages we never want a footer on (gallery index, timing report, sub-galleries).
_SKIP_BASENAMES = frozenset({"index", "sg_execution_times", "sg_api_usage"})


def _repo_root(app: Sphinx) -> Path:
    """Return the repo root (two levels above docs/source/)."""
    return Path(app.srcdir).resolve().parents[1]


def _tutorial_id_from_pagename(pagename: str) -> str | None:
    """Map a Sphinx ``pagename`` to a tutorial id, or ``None`` if irrelevant.

    Examples:
        ``generated/auto_examples/tutorials/00_start_here/plot_00_first_search``
        -> ``plot_00_first_search``
        ``generated/auto_examples/how_to/how_to_download_a_dataset``
        -> ``how_to_download_a_dataset``
        ``index`` -> ``None``

    """
    if not pagename.startswith(_GALLERY_PREFIX):
        return None
    basename = pagename.rsplit("/", 1)[-1]
    if basename in _SKIP_BASENAMES:
        return None
    return basename


def _load_evidence(repo_root: Path, tutorial_id: str) -> dict[str, Any] | None:
    path = repo_root / "docs" / "evidence" / "tutorials" / tutorial_id / "evidence.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("[eegdash-tutorial-footer] failed to parse %s: %s", path, exc)
        return None


def _load_reviewer_min_score(
    repo_root: Path, tutorial_id: str
) -> tuple[float | None, bool | None]:
    """Return (min_score, overall_pass) from reviewer_score.json or (None, None)."""
    path = (
        repo_root
        / "docs"
        / "evidence"
        / "tutorials"
        / tutorial_id
        / "reviewer_score.json"
    )
    if not path.is_file():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    score = data.get("min_score")
    overall = data.get("overall_pass")
    try:
        score_val: float | None = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_val = None
    return score_val, overall if isinstance(overall, bool) else None


def _build_footer_html(
    tutorial_id: str,
    evidence: dict[str, Any],
    reviewer_min: float | None,
    reviewer_pass: bool | None,
) -> str:
    totals = evidence.get("totals") or {}
    errors = int(totals.get("errors") or 0)
    warns = int(totals.get("warns") or 0)
    infos = int(totals.get("infos") or 0)

    spec_path = evidence.get("spec_path") or (
        f"docs/tutorials/_spec/{tutorial_id}.yaml"
    )

    audit_line = f"Static audit: {errors} errors / {warns} warnings / {infos} infos"

    if reviewer_min is None:
        reviewer_line = "Reviewer score: not yet recorded"
    else:
        gate = "passes merge gate" if reviewer_pass else "below merge gate"
        # Render at one decimal so 4 -> "4.0/5".
        reviewer_line = f"Reviewer min score: {reviewer_min:.1f}/5 ({gate})"

    spec_line = f"Spec: <code>{spec_path}</code>"
    dossier_line = f"Dossier: <code>docs/evidence/tutorials/{tutorial_id}/</code>"

    return (
        '<div class="eegdash-tutorial-footer" role="note" '
        'aria-label="Behind this lesson">'
        "<strong>Behind this lesson</strong>"
        "<ul>"
        f"<li>{audit_line}</li>"
        f"<li>{reviewer_line}</li>"
        f"<li>{spec_line}</li>"
        f"<li>{dossier_line}</li>"
        "</ul>"
        "</div>"
    )


def _on_html_page_context(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: dict[str, Any],
    doctree: Any,
) -> None:
    """``html-page-context`` hook: append the footer HTML to ``body``."""
    if doctree is None:
        return  # pages without a doctree (e.g. genindex) — skip silently.

    tutorial_id = _tutorial_id_from_pagename(pagename)
    if tutorial_id is None:
        return

    repo_root = _repo_root(app)
    evidence = _load_evidence(repo_root, tutorial_id)
    if evidence is None:
        # No dossier yet -> no footer (graceful).
        return

    reviewer_min, reviewer_pass = _load_reviewer_min_score(repo_root, tutorial_id)

    footer_html = _build_footer_html(tutorial_id, evidence, reviewer_min, reviewer_pass)

    # ``body`` is the rendered HTML for the document. Append the footer just
    # before whatever the theme wraps after it; appending to ``body`` is the
    # supported public surface for sphinx html-page-context listeners.
    existing = context.get("body") or ""
    context["body"] = existing + footer_html


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("html-page-context", _on_html_page_context)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

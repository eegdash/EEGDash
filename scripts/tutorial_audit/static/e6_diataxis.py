"""Static E.6 Diataxis-purity checks.

The Diataxis framework (https://diataxis.fr/) splits documentation into four
quadrants -- tutorials, how-to guides, reference, explanation. EEGDash
tutorials live in the *tutorial* quadrant and must defer deeper conceptual
material to the *explanation* quadrant rather than inlining theory. This
module owns the static slice of that contract: scanning a tutorial for at
least one cross-reference into the explanation tree.

Two of the three E.6 rules are reviewer-only by design (E6.47 narrative
purity, E6.49 how-to extraction) -- they ask "should this paragraph have
been factored into a different document quadrant?" which a regex cannot
answer. They are exposed as no-op stubs that emit a single ``info`` finding
so the orchestrator records that a human/LLM reviewer must adjudicate.

Citations
---------
- ``cite_rubric``: the rubric line in ``compass_artifact.md`` (rule ID after
  the ``#``).
- ``cite_plan``: the prescriptive design template in
  ``tutorial_restructure_plan.md``. Lines 516-562 enumerate the
  Title/Opening/.../Links template; the Links section explicitly directs
  authors to a "Detailed explanation page" -- exactly the cross-reference
  E6.48 enforces.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..api import Finding

if TYPE_CHECKING:  # pragma: no cover
    from ..api import RunContext


# Sphinx ``:doc:`` cross-reference into the explanation tree. We accept both
# ``concepts/`` (current EEGDash convention, mirrored in the spec
# ``links.concept`` field) and ``explanation/`` (canonical Diataxis name).
# Examples accepted:
#   :doc:`/concepts/leakage_and_evaluation`
#   :doc:`leakage <../concepts/leakage_and_evaluation>`
#   :doc:`docs/source/explanation/why_montage`
DOC_REF_RE = re.compile(
    r":doc:`[^`]*?(?:concepts|explanation)/[^`]+`",
    re.I,
)
# Markdown / reST link whose URL points into ``concepts/`` or ``explanation/``.
# Catches both ``[label](docs/source/concepts/...)`` and bare URL references.
MD_LINK_RE = re.compile(
    r"(?:\]\(|<|\b)"
    r"(?:[\w./-]*?)(?:concepts|explanation)/[\w./-]+",
    re.I,
)
# Sphinx-gallery markdown block opener.
MD_BLOCK_RE = re.compile(r"^# %% \[markdown\]\s*$")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _spec_kind(spec: dict) -> str:
    """Return ``spec.kind`` lowercased; default ``"tutorial"``."""
    kind = spec.get("kind") or "tutorial"
    return str(kind).strip().lower()


# -- E6.48 ------------------------------------------------------------------


def check_concept_link_present(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    """E6.48 -- tutorial must link to at least one explanation/concept page.

    Acceptable forms:

    * ``:doc:`...concepts/<page>``` (Sphinx)
    * ``:doc:`...explanation/<page>``` (canonical Diataxis name)
    * Markdown / reST link whose URL contains ``concepts/`` or
      ``explanation/`` (e.g. ``[leakage](docs/source/concepts/leakage.rst)``).

    The check scans the entire tutorial source -- module docstring,
    sphinx-gallery markdown blocks, and even comments -- because the
    cross-reference can legitimately appear anywhere a reST/Markdown link
    is rendered.

    For how-tos (``kind == "how-to"``) the rule is downgraded to ``info``:
    the Diataxis convention encourages but does not require recipe pages to
    cross-link an explanation page.
    """
    src = _read(tutorial_path)
    matches: list[str] = []
    matches.extend(DOC_REF_RE.findall(src))
    matches.extend(MD_LINK_RE.findall(src))
    if matches:
        return []

    level = "info" if _spec_kind(spec) == "how-to" else "warn"
    return [
        Finding(
            rule_id="E6.48",
            level=level,
            message=(
                "Source does not link to any explanation/concept page; "
                "Diataxis purity asks tutorials to defer deeper theory to "
                "the explanation quadrant via a :doc: or markdown link"
            ),
            cite_rubric="compass_artifact.md#E6.48",
            cite_plan="tutorial_restructure_plan.md#L516-L562",
            evidence={"n_matches": 0, "kind": _spec_kind(spec)},
            tool="regex",
        )
    ]


# -- E6.47 (reviewer-only stub) --------------------------------------------


def check_diataxis_purity(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    """E6.47 -- 'one rail, one outcome' Diataxis purity is reviewer-only.

    Returned as a single ``info`` finding so the dossier records that this
    rubric item still requires a human or LLM reviewer.
    """
    return [
        Finding(
            rule_id="E6.47",
            level="info",
            message=(
                "E6.47 'Diataxis purity: stays a tutorial, doesn't drift into "
                "reference/how-to/explanation' is reviewer-only; see "
                "tutorial_implementation_strategy.md 'Reviewer-only rubric items'"
            ),
            cite_rubric="compass_artifact.md#E6.47",
            cite_plan="tutorial_restructure_plan.md#L516-L562",
            evidence={"reviewer_only": True},
            tool="reviewer-stub",
        )
    ]


# -- E6.49 (reviewer-only stub) --------------------------------------------


def check_how_to_extraction(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    """E6.49 -- 'split out a separate how-to where appropriate' is reviewer-only."""
    return [
        Finding(
            rule_id="E6.49",
            level="info",
            message=(
                "E6.49 'where a competent user wants a quick recipe, split out "
                "a separate how-to' is reviewer-only; see "
                "tutorial_implementation_strategy.md 'Reviewer-only rubric items'"
            ),
            cite_rubric="compass_artifact.md#E6.49",
            cite_plan="tutorial_restructure_plan.md#L516-L562",
            evidence={"reviewer_only": True},
            tool="reviewer-stub",
        )
    ]

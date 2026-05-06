"""Static E.4 engagement checks (DOI citation, Extensions section, motivation).

The engagement rubric (compass_artifact.md#E.4) blends machine-checkable
surface signals -- "is there a DOI string?", "is there an Extensions
section with at least three bullets?" -- with reviewer-only judgments
("does the opening name a real neuroscience question?", "is the tone
inclusive and explanatory?"). This module owns the static, machine-checkable
half. Reviewer-only rules (E4.31 narrative, E4.33 result-meaningful, E4.35
tone) are exposed as no-op stubs that emit a single ``info`` finding so
that the orchestrator records, per tutorial, that a human or LLM reviewer
must still adjudicate them.

Citations
---------
- ``cite_rubric``: the literature-anchored rubric line in
  ``compass_artifact.md`` (rule ID after the ``#``).
- ``cite_plan``: the prescriptive design template in
  ``tutorial_restructure_plan.md``. Lines 516-562 describe the
  Title/Opening/.../Wrap-up template; line 530 lists the "Links"
  block which holds related papers; lines 549-552 are the explicit
  prohibition on optional branches that the Extensions block softens by
  pushing them to the wrap-up.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..api import Finding

if TYPE_CHECKING:  # pragma: no cover
    from ..api import RunContext


# DOI regex per Crossref's published recommendation: ``10.<registrant>/<suffix>``.
# We anchor on the ``10.`` prefix and require at least four digits in the
# registrant code, which is the published Crossref minimum. The suffix is
# any run of non-whitespace characters.
DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+", re.I)
# Trailing punctuation we strip from the suffix when comparing duplicates --
# DOIs printed in prose pick up brackets, commas, parentheses, quotes.
DOI_TRAILING_PUNCT = ".,;:)]}\"'"

# Headers that satisfy E4.34's "Try it yourself / Extensions" wrap-up. We
# accept Markdown ATX, reST H1/H2 underline, and bold/italic emphasis. The
# matcher is case-insensitive.
EXTENSIONS_HEADER_RE = re.compile(
    r"^\s*(?:#+\s*|\*+\s*)?"
    r"(?:try\s+it\s+yourself|extensions?|further\s+exploration|"
    r"go\s+further|next\s+steps?|challenges?|exercises?)\b",
    re.I,
)
# A reST-style underline is ``=`` or ``-`` repeated at least three times.
REST_UNDERLINE_RE = re.compile(r"^[=\-]{3,}\s*$")
# Markdown bullet at the start of a line, allowing the ``# `` prefix that
# sphinx-gallery uses for prose lines inside markdown blocks.
BULLET_RE = re.compile(r"^\s*(?:#\s*)?[-*+]\s+\S")

# Sphinx-gallery markdown block opener.
MD_BLOCK_RE = re.compile(r"^# %% \[markdown\]\s*$")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_md_prefix(line: str) -> str:
    """Strip a leading ``# `` (sphinx-gallery prose prefix) and return the rest."""
    s = line
    # Drop a single leading ``# `` or ``#`` so prose lines parse as markdown.
    if s.startswith("# "):
        return s[2:]
    if s.startswith("#"):
        return s[1:]
    return s


def _markdown_blocks(src: str) -> list[str]:
    """Return text of each ``# %% [markdown]`` block, with ``# `` prefixes stripped.

    Mirrors the helper in ``e2_pedagogical.py`` so this module stays
    self-contained -- importing across static modules would couple their
    refactor cycles unnecessarily.
    """
    blocks: list[str] = []
    cur: list[str] | None = None
    for line in src.splitlines():
        if MD_BLOCK_RE.match(line):
            if cur is not None:
                blocks.append("\n".join(cur))
            cur = []
            continue
        if line.startswith("# %%"):
            if cur is not None:
                blocks.append("\n".join(cur))
                cur = None
            continue
        if cur is not None:
            cur.append(_strip_md_prefix(line))
    if cur is not None:
        blocks.append("\n".join(cur))
    return blocks


def _docstring_text(src: str) -> str:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""
    return ast.get_docstring(tree) or ""


def _normalise_doi(token: str) -> str:
    """Lowercase and strip trailing punctuation so duplicate detection is robust."""
    return token.rstrip(DOI_TRAILING_PUNCT).lower()


def _spec_lists_related_papers(spec: dict) -> bool:
    links = spec.get("links")
    if not isinstance(links, dict):
        return False
    related = links.get("related_papers")
    if isinstance(related, (list, tuple)):
        return any(str(item).strip() for item in related)
    if isinstance(related, str):
        return bool(related.strip())
    return False


# -- E4.32 ------------------------------------------------------------------


def check_dataset_citation(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    """E4.32 -- tutorial source must surface at least one DOI.

    Severity is ``error`` when the spec advertises ``links.related_papers``
    (so the citation is contractually required); otherwise ``warn`` so we
    still nudge authors. When the only DOIs detected are duplicates of each
    other we additionally emit a ``warn`` -- the boilerplate-DOI pattern is
    the most common shape of an unedited template.
    """
    src = _read(tutorial_path)
    raw_matches = DOI_RE.findall(src)
    normalised = [_normalise_doi(m) for m in raw_matches]
    unique_dois = sorted({d for d in normalised if d})

    findings: list[Finding] = []
    spec_demands_citation = _spec_lists_related_papers(spec)

    if not unique_dois:
        level = "error" if spec_demands_citation else "warn"
        findings.append(
            Finding(
                rule_id="E4.32",
                level=level,
                message=(
                    "No DOI ('10.<registrant>/<suffix>') found in tutorial source; "
                    "cite the dataset paper, eegdash entry, or a canonical reference"
                ),
                cite_rubric="compass_artifact.md#E4.32",
                cite_plan="tutorial_restructure_plan.md#L516-L562",
                evidence={
                    "spec_lists_related_papers": spec_demands_citation,
                    "n_doi_matches": 0,
                },
                tool="regex",
            )
        )
        return findings

    # Detect boilerplate: every DOI string is the same token. ``raw_matches``
    # captures repeats, ``unique_dois`` collapses them.
    if len(raw_matches) >= 2 and len(unique_dois) == 1:
        findings.append(
            Finding(
                rule_id="E4.32",
                level="warn",
                message=(
                    f"All {len(raw_matches)} DOI mentions resolve to the same "
                    "identifier; this often indicates an unedited boilerplate "
                    "citation block"
                ),
                cite_rubric="compass_artifact.md#E4.32",
                cite_plan="tutorial_restructure_plan.md#L516-L562",
                evidence={
                    "n_doi_matches": len(raw_matches),
                    "unique_dois": unique_dois,
                },
                tool="regex",
            )
        )
    return findings


# -- E4.34 ------------------------------------------------------------------


def _bullet_lines_after_header(block: str) -> int:
    """Return the count of bullet lines that immediately follow an Extensions header.

    Walks the block line by line: once the header is seen, we collect
    contiguous bullets (allowing blank lines between them). Tolerates a reST
    underline directly under the header. Stops when a non-bullet, non-blank,
    non-underline line is encountered.
    """
    lines = block.splitlines()
    in_block = False
    after_header_idx = -1
    for idx, line in enumerate(lines):
        if not in_block:
            if EXTENSIONS_HEADER_RE.match(line.strip()):
                in_block = True
                after_header_idx = idx + 1
            continue
        break

    if not in_block:
        return 0

    n_bullets = 0
    seen_bullet = False
    for line in lines[after_header_idx:]:
        stripped = line.strip()
        if not stripped:
            # Blank lines tolerated before/between bullets.
            continue
        if REST_UNDERLINE_RE.match(stripped):
            # Underline beneath the header itself; skip.
            continue
        if BULLET_RE.match(line):
            n_bullets += 1
            seen_bullet = True
            continue
        # First non-bullet, non-blank line after bullets started ends the block.
        if seen_bullet:
            break
        # Non-bullet prose between header and first bullet -- continue
        # scanning; some authors write a 1-line intro before the list.
        continue
    return n_bullets


def check_extensions_section(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    """E4.34 -- tutorial wrap-up must contain an Extensions section with >= 3 bullets.

    The header may live in the module docstring or in any
    ``# %% [markdown]`` block. We accept several common phrasings (Try it
    yourself, Extensions, Further exploration, Next steps, Challenges,
    Exercises). The bullet count is *contiguous* bullets after the header so
    a stray bulleted list elsewhere in the tutorial does not satisfy E4.34.
    """
    src = _read(tutorial_path)
    blocks: list[str] = []
    docstring = _docstring_text(src)
    if docstring:
        blocks.append(docstring)
    blocks.extend(_markdown_blocks(src))

    best: int = 0
    found_header = False
    for block in blocks:
        if not EXTENSIONS_HEADER_RE.search(block):
            continue
        found_header = True
        count = _bullet_lines_after_header(block)
        if count > best:
            best = count

    if not found_header:
        return [
            Finding(
                rule_id="E4.34",
                level="warn",
                message=(
                    "Tutorial wrap-up missing an 'Extensions' / 'Try it yourself' "
                    "section; sphinx-gallery tutorials should close with at least "
                    "three graded modifications"
                ),
                cite_rubric="compass_artifact.md#E4.34",
                cite_plan="tutorial_restructure_plan.md#L516-L562",
                evidence={"bullet_count": 0},
                tool="regex",
            )
        ]
    if best < 3:
        return [
            Finding(
                rule_id="E4.34",
                level="warn",
                message=(
                    f"Extensions section found but only {best} graded modification "
                    "bullets; rubric requires at least 3"
                ),
                cite_rubric="compass_artifact.md#E4.34",
                cite_plan="tutorial_restructure_plan.md#L516-L562",
                evidence={"bullet_count": best},
                tool="regex",
            )
        ]
    return []


# -- E4.31 (machine-checkable subset) --------------------------------------


def check_motivating_question_present(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    """E4.31 (subset) -- the first markdown block must end on a question mark.

    The full E4.31 rubric ("first 5 lines name a real neuroscience question,
    not a generic 'this example shows class X'") requires reviewer judgment.
    This static slice catches the most common mechanical failure: tutorials
    that close their opening block with a flat assertion. We always emit a
    single ``info`` finding noting the reviewer-only nature of the wider rule
    so the dossier records that the human/LLM reviewer is still on the hook.
    """
    src = _read(tutorial_path)
    findings: list[Finding] = []
    blocks: list[str] = []
    docstring = _docstring_text(src)
    if docstring:
        blocks.append(docstring)
    blocks.extend(_markdown_blocks(src))

    if not blocks:
        # No prose at all -- E1.2 covers that; we say nothing structural here
        # and only emit the reviewer-only reminder below.
        first_block_ends_with_question = False
    else:
        first = blocks[0].strip()
        first_block_ends_with_question = first.endswith("?")

    if blocks and not first_block_ends_with_question:
        findings.append(
            Finding(
                rule_id="E4.31",
                level="warn",
                message=(
                    "Opening prose block does not end with a question; E4.31 "
                    "asks the first lines to name a real neuroscience question"
                ),
                cite_rubric="compass_artifact.md#E4.31",
                cite_plan="tutorial_restructure_plan.md#L516-L562",
                evidence={
                    "ends_with_question_mark": False,
                    "tail": blocks[0].strip()[-160:],
                },
                tool="regex",
            )
        )

    findings.append(
        Finding(
            rule_id="E4.31",
            level="info",
            message=(
                "E4.31 'first lines name a real neuroscience question' requires "
                "reviewer judgment; see "
                "tutorial_implementation_strategy.md 'Reviewer-only rubric items'"
            ),
            cite_rubric="compass_artifact.md#E4.31",
            cite_plan="tutorial_restructure_plan.md#L516-L562",
            evidence={"reviewer_only": True},
            tool="reviewer-stub",
        )
    )
    return findings


# -- E4.33 (reviewer-only stub) --------------------------------------------


def check_result_meaningful(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    """E4.33 -- 'result has scientific meaning' is reviewer-only.

    Returned as a single ``info`` finding so the dossier records that this
    rubric item still requires a human or LLM reviewer.
    """
    return [
        Finding(
            rule_id="E4.33",
            level="info",
            message=(
                "E4.33 'result has scientific meaning' is reviewer-only; "
                "see tutorial_implementation_strategy.md 'Reviewer-only "
                "rubric items'"
            ),
            cite_rubric="compass_artifact.md#E4.33",
            cite_plan="tutorial_restructure_plan.md#L516-L562",
            evidence={"reviewer_only": True},
            tool="reviewer-stub",
        )
    ]


# -- E4.35 (reviewer-only stub) --------------------------------------------


def check_tone(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    """E4.35 -- inclusive 'we' tone, present tense, explanatory voice; reviewer-only."""
    return [
        Finding(
            rule_id="E4.35",
            level="info",
            message=(
                "E4.35 'tone: we-inclusive, present tense, explains why' is "
                "reviewer-only; see tutorial_implementation_strategy.md "
                "'Reviewer-only rubric items'"
            ),
            cite_rubric="compass_artifact.md#E4.35",
            cite_plan="tutorial_restructure_plan.md#L516-L562",
            evidence={"reviewer_only": True},
            tool="reviewer-stub",
        )
    ]

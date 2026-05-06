"""Static E.1 structural / sphinx-gallery conformity checks for tutorial files."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..api import Finding

if TYPE_CHECKING:  # pragma: no cover
    from ..api import RunContext


# A reST H1 underline is a line of '=' at least 3 long. Sphinx-gallery's own
# converter accepts shorter, but we follow the rubric language ("====" / H1).
REST_TITLE_RE = re.compile(r"^={3,}\s*$")
# Block delimiters per sphinx-gallery: '# %%' or '# %% [markdown]'.
BLOCK_DELIM_RE = re.compile(r"^# %%(\s*\[markdown\])?\s*$")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_filename(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    """E1.1 -- filename must match ``plot_*.py`` for sphinx-gallery to execute it."""
    name = tutorial_path.name
    if not (name.startswith("plot_") and name.endswith(".py")):
        return [
            Finding(
                rule_id="E1.1",
                level="error",
                message=(
                    "Tutorial filename must match plot_*.py for sphinx-gallery; "
                    f"got {name!r}"
                ),
                cite_rubric="compass_artifact.md#E1.1",
                cite_plan="tutorial_restructure_plan.md#L1194",
                evidence={"name": name},
                tool="filename",
            )
        ]
    return []


def check_docstring_header(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    """E1.2 -- module docstring with reST H1 title and a >=2-paragraph motivator."""
    src = _read(tutorial_path)
    findings: list[Finding] = []
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [
            Finding(
                rule_id="E1.2",
                level="error",
                message=f"Cannot parse tutorial source: {exc.msg} (line {exc.lineno})",
                cite_rubric="compass_artifact.md#E1.2",
                cite_plan="tutorial_restructure_plan.md#L516-L518",
                evidence={"syntax_error": str(exc)},
                tool="ast",
            )
        ]
    docstring = ast.get_docstring(tree)
    if not docstring:
        return [
            Finding(
                rule_id="E1.2",
                level="error",
                message="Module-level docstring missing",
                cite_rubric="compass_artifact.md#E1.2",
                cite_plan="tutorial_restructure_plan.md#L516-L517",
                tool="ast",
            )
        ]
    lines = docstring.strip().splitlines()
    underline = lines[1] if len(lines) > 1 else ""
    if not REST_TITLE_RE.match(underline):
        findings.append(
            Finding(
                rule_id="E1.2",
                level="error",
                message=(
                    "Docstring must open with a reST H1 title underlined by '=' "
                    "(at least three '=' on the second line)"
                ),
                cite_rubric="compass_artifact.md#E1.2",
                cite_plan="tutorial_restructure_plan.md#L516",
                evidence={"second_line": underline},
                tool="regex",
            )
        )
    paragraphs = [p.strip() for p in docstring.split("\n\n") if p.strip()]
    if len(paragraphs) < 2:
        findings.append(
            Finding(
                rule_id="E1.2",
                level="error",
                message=(
                    "Docstring must contain a title paragraph and at least one "
                    "motivating paragraph (got "
                    f"{len(paragraphs)} non-empty paragraph(s))"
                ),
                cite_rubric="compass_artifact.md#E1.2",
                cite_plan="tutorial_restructure_plan.md#L516-L518",
                evidence={"n_paragraphs": len(paragraphs)},
                tool="regex",
            )
        )
    elif len(paragraphs[1].split()) < 30:
        findings.append(
            Finding(
                rule_id="E1.2",
                level="warn",
                message=(
                    "Motivating paragraph after title should be 2-4 sentences "
                    "naming dataset and scientific question (saw "
                    f"{len(paragraphs[1].split())} words)"
                ),
                cite_rubric="compass_artifact.md#E1.2",
                cite_plan="tutorial_restructure_plan.md#L517-L518",
                evidence={"second_para_words": len(paragraphs[1].split())},
                tool="regex",
            )
        )
    return findings


def check_block_delimiters(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    """E1.4 -- at least one ``# %%`` or ``# %% [markdown]`` cell delimiter."""
    src = _read(tutorial_path)
    if any(BLOCK_DELIM_RE.match(line) for line in src.splitlines()):
        return []
    return [
        Finding(
            rule_id="E1.4",
            level="error",
            message=(
                "Tutorial must use '# %%' or '# %% [markdown]' block delimiters "
                "to separate code from prose"
            ),
            cite_rubric="compass_artifact.md#E1.4",
            cite_plan="tutorial_restructure_plan.md#L530",
            tool="regex",
        )
    ]


def check_loc_budget(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    """E1.8 -- LOC must be within ``spec.budgets.max_loc`` (default 220)."""
    src = _read(tutorial_path)
    n_loc = sum(1 for _ in src.splitlines())
    cap = int(spec.get("budgets", {}).get("max_loc", 220))
    if n_loc > cap:
        return [
            Finding(
                rule_id="E1.8",
                level="error",
                message=f"LOC {n_loc} exceeds budget {cap}",
                cite_rubric="compass_artifact.md#E1.8",
                cite_plan="tutorial_restructure_plan.md#L1174-L1175",
                evidence={"loc": n_loc, "budget": cap},
                tool="wc",
            )
        ]
    return []

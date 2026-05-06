"""Static E.5 domain-correctness checks (filter disclosure for now)."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..api import Finding

if TYPE_CHECKING:  # pragma: no cover
    from ..api import RunContext


# Tokens that count as "disclosure" of a filter choice (Cisotto & Chicco 2024,
# tips 4-5). We accept either generic English ("pass-band", "stop-band",
# "filter type") or a numeric Hz cutoff that names the filter -- "0.5 Hz",
# "30 Hz", "1-40 Hz".
DISCLOSURE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"pass[-\s]?band", re.I),
    re.compile(r"stop[-\s]?band", re.I),
    re.compile(r"\bfilter\s+type\b", re.I),
    re.compile(r"\bcausal\b", re.I),
    re.compile(r"\bnon[-\s]?causal\b", re.I),
    re.compile(r"\bfir\b", re.I),
    re.compile(r"\biir\b", re.I),
    re.compile(r"\bbutter(worth)?\b", re.I),
    re.compile(r"\bfirwin\b", re.I),
    re.compile(r"\bnotch\b", re.I),
    re.compile(r"\bhigh[-\s]?pass\b", re.I),
    re.compile(r"\blow[-\s]?pass\b", re.I),
    re.compile(r"\bband[-\s]?pass\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*Hz\b", re.I),
]

# Default "context window": how many source lines around the .filter() call
# count as accompanying disclosure. The rubric says 50 lines.
DEFAULT_CONTEXT_LINES = 50


class _FilterCallFinder(ast.NodeVisitor):
    """Collect line numbers of method calls named ``filter``.

    We only care about *method* invocations -- ``raw.filter(...)``,
    ``epochs.filter(...)`` etc. Plain function calls named ``filter`` are
    ambiguous (Python builtin, pandas, etc.), so they are skipped.
    """

    def __init__(self) -> None:
        self.lines: list[int] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Attribute) and node.func.attr == "filter":
            self.lines.append(node.lineno)
        self.generic_visit(node)


def check_filter_disclosed(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    """E5.37 -- when ``.filter(...)`` is called, disclose pass-band / stop-band /
    filter type within the surrounding 50 lines of source (docstring or
    markdown prose).
    """
    src = tutorial_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [
            Finding(
                rule_id="E5.37",
                level="error",
                message=f"Cannot parse tutorial source: {exc.msg} (line {exc.lineno})",
                cite_rubric="compass_artifact.md#E5.37",
                cite_plan="tutorial_restructure_plan.md#L1158-L1178",
                evidence={"syntax_error": str(exc)},
                tool="ast",
            )
        ]

    finder = _FilterCallFinder()
    finder.visit(tree)
    if not finder.lines:
        return []

    docstring = ast.get_docstring(tree) or ""
    docstring_disclosed = _has_disclosure(docstring)
    src_lines = src.splitlines()

    findings: list[Finding] = []
    for lineno in finder.lines:
        start = max(0, lineno - 1 - DEFAULT_CONTEXT_LINES)
        end = min(len(src_lines), lineno - 1 + DEFAULT_CONTEXT_LINES + 1)
        context = "\n".join(src_lines[start:end])
        if docstring_disclosed or _has_disclosure(context):
            continue
        findings.append(
            Finding(
                rule_id="E5.37",
                level="error",
                message=(
                    f"Filter call at line {lineno} without nearby disclosure "
                    "of pass-band, stop-band, or filter type within 50 lines"
                ),
                cite_rubric="compass_artifact.md#E5.37",
                cite_plan="tutorial_restructure_plan.md#L1158-L1178",
                evidence={
                    "filter_line": lineno,
                    "context_lines": [start + 1, end],
                },
                tool="ast+regex",
            )
        )
    return findings


def _has_disclosure(text: str) -> bool:
    return any(pat.search(text) for pat in DISCLOSURE_PATTERNS)

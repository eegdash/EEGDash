"""Static E.3 technical / reproducibility checks (seeds, paths, inline pip).

The AST-based validators (seed detection, hard-coded paths) skip Markdown
how-tos, since those are template + commentary, not Python source. The
inline-pip regex still runs on Markdown sources because shell-bang patterns
are equally bad in a SLURM script as in a notebook.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..api import Finding

if TYPE_CHECKING:  # pragma: no cover
    from ..api import RunContext


# Hard-coded host paths we never accept inside a tutorial source. Cache
# directories must come from a variable, ``Path("./...")``, or env lookup --
# these are the substring patterns that flag a literal absolute path.
HARDCODED_PATH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^/Users/"),
    re.compile(r"^/home/"),
    re.compile(r"^[A-Za-z]:\\\\"),
    re.compile(r"^[A-Za-z]:\\"),
    re.compile(r"^~/Downloads/"),
    re.compile(r"^~/Desktop/"),
]

# Inline package-installation patterns. These should never appear in a
# rendered tutorial -- environments are pinned at the top of the gallery.
INLINE_PIP_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("shell-bang-pip", re.compile(r"^\s*!\s*pip\s+install\b")),
    ("magic-pip", re.compile(r"^\s*%pip\s+install\b")),
    ("magic-conda", re.compile(r"^\s*%conda\s+install\b")),
    ("shell-bang-conda", re.compile(r"^\s*!\s*conda\s+install\b")),
    (
        "subprocess-pip",
        re.compile(r"subprocess\.[A-Za-z_]+\([^)]*['\"]pip['\"]"),
    ),
]


def _spec_output_kind(spec: dict) -> str:
    """Return ``spec.output_kind`` lowercased; default ``"python"``."""
    kind = spec.get("output_kind") or "python"
    return str(kind).strip().lower()


def _is_markdown_source(tutorial_path: Path, spec: dict) -> bool:
    """True when the tutorial source is a Markdown file (no AST parsing)."""
    if tutorial_path.suffix.lower() == ".md":
        return True
    return _spec_output_kind(spec) == "markdown"


def _seed_keyword_args() -> set[str]:
    return {"random_state", "seed"}


class _SeedVisitor(ast.NodeVisitor):
    """Collect evidence of any RNG seeding pattern in a tutorial."""

    def __init__(self) -> None:
        self.has_np_seed = False
        self.has_torch_seed = False
        self.has_random_seed = False  # stdlib random.seed
        self.has_random_state_kw = False
        self._evidence: list[str] = []

    @property
    def evidence(self) -> list[str]:
        return self._evidence

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        # Detect attribute calls like np.random.seed(...) and torch.manual_seed(...).
        func = node.func
        attr_chain = self._dotted_name(func)
        if attr_chain:
            if attr_chain.endswith("np.random.seed") or attr_chain.endswith(
                "numpy.random.seed"
            ):
                self.has_np_seed = True
                self._evidence.append(f"line {node.lineno}: {attr_chain}(...)")
            elif (
                attr_chain.endswith("torch.manual_seed")
                or attr_chain.endswith("torch.cuda.manual_seed")
                or attr_chain.endswith("torch.cuda.manual_seed_all")
            ):
                self.has_torch_seed = True
                self._evidence.append(f"line {node.lineno}: {attr_chain}(...)")
            elif attr_chain.endswith("random.seed"):
                self.has_random_seed = True
                self._evidence.append(f"line {node.lineno}: {attr_chain}(...)")
        # Any keyword argument named random_state or seed counts.
        for kw in node.keywords:
            if kw.arg in _seed_keyword_args():
                self.has_random_state_kw = True
                self._evidence.append(f"line {node.lineno}: {kw.arg}=...")
        self.generic_visit(node)

    @staticmethod
    def _dotted_name(node: ast.AST) -> str:
        parts: list[str] = []
        cur: ast.AST | None = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        return ""


def check_seeds(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    """E3.21 -- at least one RNG-seeding pattern must appear in the tutorial.

    Markdown how-tos are skipped: they are template + commentary documents
    that do not embed runnable Python code paths and thus cannot be expected
    to seed an RNG.
    """
    if _is_markdown_source(tutorial_path, spec):
        return []
    overrides = spec.get("rule_overrides") or {}
    if str(overrides.get("E3.21", "")).strip().lower() == "exempt":
        return []
    src = tutorial_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [
            Finding(
                rule_id="E3.21",
                level="error",
                message=f"Cannot parse tutorial source: {exc.msg} (line {exc.lineno})",
                cite_rubric="compass_artifact.md#E3.21",
                cite_plan="tutorial_restructure_plan.md#L1164",
                evidence={"syntax_error": str(exc)},
                tool="ast",
            )
        ]

    visitor = _SeedVisitor()
    visitor.visit(tree)
    seeded = (
        visitor.has_np_seed
        or visitor.has_torch_seed
        or visitor.has_random_seed
        or visitor.has_random_state_kw
    )
    if seeded:
        return []
    return [
        Finding(
            rule_id="E3.21",
            level="error",
            message=(
                "No RNG seed detected; expected one of np.random.seed(...), "
                "torch.manual_seed(...), random.seed(...), or a random_state= "
                "keyword argument"
            ),
            cite_rubric="compass_artifact.md#E3.21",
            cite_plan="tutorial_restructure_plan.md#L1164",
            evidence={
                "has_np_seed": visitor.has_np_seed,
                "has_torch_seed": visitor.has_torch_seed,
                "has_random_seed": visitor.has_random_seed,
                "has_random_state_kw": visitor.has_random_state_kw,
            },
            tool="ast",
        )
    ]


class _StringLiteralVisitor(ast.NodeVisitor):
    """Collect every string literal with its source line number."""

    def __init__(self) -> None:
        self.strings: list[tuple[int, str]] = []

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if isinstance(node.value, str):
            self.strings.append((node.lineno, node.value))
        self.generic_visit(node)


def check_no_hardcoded_paths(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    r"""E3.24 -- no string literals that look like absolute host paths.

    The eegdash cache directory should come from a variable, ``Path("./...")``,
    or an environment variable lookup -- never a hard-coded ``/Users/...``,
    ``/home/...``, ``C:\...`` or ``~/Downloads/...`` path. For Markdown
    how-tos we fall back to the line-level regex scan since there is no
    Python AST to walk.
    """
    src = tutorial_path.read_text(encoding="utf-8")
    if _is_markdown_source(tutorial_path, spec):
        return _hardcoded_paths_line_scan(src)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # Fall back to line-level scan when AST parsing fails so we still
        # surface obvious leaks.
        return _hardcoded_paths_line_scan(src)

    visitor = _StringLiteralVisitor()
    visitor.visit(tree)
    findings: list[Finding] = []
    for lineno, literal in visitor.strings:
        for pattern in HARDCODED_PATH_PATTERNS:
            if pattern.search(literal):
                findings.append(
                    Finding(
                        rule_id="E3.24",
                        level="error",
                        message=(
                            f"Hard-coded host path literal {literal!r} at line "
                            f"{lineno}; cache directories must be parametrised"
                        ),
                        cite_rubric="compass_artifact.md#E3.24",
                        cite_plan="tutorial_restructure_plan.md#L1166",
                        evidence={"line": lineno, "literal": literal},
                        tool="ast",
                    )
                )
                break
    return findings


def _hardcoded_paths_line_scan(src: str) -> list[Finding]:
    """Regex fallback when the file is unparseable."""
    findings: list[Finding] = []
    line_re = re.compile(
        r"(['\"])(/Users/|/home/|[A-Za-z]:\\\\|[A-Za-z]:\\|~/Downloads/|~/Desktop/)"
    )
    for i, line in enumerate(src.splitlines(), start=1):
        if line_re.search(line):
            findings.append(
                Finding(
                    rule_id="E3.24",
                    level="error",
                    message=(
                        f"Hard-coded host path detected at line {i}; "
                        "cache directories must be parametrised"
                    ),
                    cite_rubric="compass_artifact.md#E3.24",
                    cite_plan="tutorial_restructure_plan.md#L1166",
                    evidence={"line": i, "snippet": line.strip()[:200]},
                    tool="regex",
                )
            )
    return findings


def check_no_inline_pip(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    """E3.30 -- no ``!pip install``, ``%pip install``, or subprocess pip call."""
    src = tutorial_path.read_text(encoding="utf-8")
    findings: list[Finding] = []
    for i, line in enumerate(src.splitlines(), start=1):
        for kind, pattern in INLINE_PIP_PATTERNS:
            if pattern.search(line):
                findings.append(
                    Finding(
                        rule_id="E3.30",
                        level="error",
                        message=(
                            f"Inline package installation detected ({kind}) "
                            f"at line {i}; tutorials must not pip/conda install"
                            " mid-execution"
                        ),
                        cite_rubric="compass_artifact.md#E3.30",
                        cite_plan="tutorial_restructure_plan.md#L1166",
                        evidence={
                            "line": i,
                            "kind": kind,
                            "snippet": line.strip()[:200],
                        },
                        tool="regex",
                    )
                )
                break
    return findings

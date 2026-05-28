"""AST-based audit of function-scoped imports in the PR diff.

Reports every ``import``/``from … import …`` statement that lives inside a
function body (i.e. would trip ruff's preview rule ``PLC0415``,
``import-outside-top-level``). Used as a one-shot inventory while the
codebase converges on PEP 8 module-level imports.

Usage::

    python scripts/ingestions/dev/audit_local_imports.py [path ...]

Default scope: ``eegdash/`` + ``scripts/ingestions/`` (skipping tests/,
data/, and *.bak*).
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOTS = ("eegdash", "scripts/ingestions")
SKIP_DIRS = {"__pycache__", ".cache", "tests", "mutants", "data", "1_fetch_sources"}


def _iter_py(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix == ".py":
            yield root
            continue
        for p in root.rglob("*.py"):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            yield p


def _local_imports(path: Path) -> list[tuple[int, str]]:
    """Return ``[(lineno, statement_text), ...]`` for nested imports in ``path``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        print(f"# skip {path}: {exc}", file=sys.stderr)
        return []
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for sub in ast.walk(node):
            if sub is node:
                continue
            if isinstance(sub, ast.Import | ast.ImportFrom):
                hits.append((sub.lineno, ast.unparse(sub)))
    return hits


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    roots = [REPO / a for a in argv] if argv else [REPO / r for r in DEFAULT_ROOTS]
    total = 0
    for py in sorted(_iter_py(roots)):
        hits = _local_imports(py)
        if not hits:
            continue
        rel = py.relative_to(REPO)
        for lineno, text in hits:
            print(f"{rel}:{lineno}: {text}")
            total += 1
    print(f"\n# {total} function-scoped imports found")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

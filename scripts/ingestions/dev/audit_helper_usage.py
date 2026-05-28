"""AST-based audit of how often each private helper is used.

A *private helper* is any module-level ``def _foo(…)`` (single-underscore
prefix, not dunder). For each one we count the references *within the same
file* (private helpers shouldn't be imported across files, but we still
inspect siblings as a sanity check).

The output highlights candidates for inlining:

* **0 callers** → unused, delete.
* **1 caller**  → consider inlining (small functions only).
* **2+ callers** → keep as a helper.

Run::

    python scripts/ingestions/dev/audit_helper_usage.py [path ...]

Default scope mirrors ``audit_local_imports.py``.
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


def _private_helpers(tree: ast.AST) -> list[tuple[str, int, int]]:
    """Return ``[(name, lineno, body_len_lines), ...]`` for top-level ``_foo``."""
    out: list[tuple[str, int, int]] = []
    if not isinstance(tree, ast.Module):
        return out
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not node.name.startswith("_") or node.name.startswith("__"):
            continue
        body_len = (node.end_lineno or node.lineno) - node.lineno
        out.append((node.name, node.lineno, body_len))
    return out


def _count_uses(tree: ast.AST, name: str) -> int:
    """Count Name references to ``name`` *outside* its own definition body."""
    own_def: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    if isinstance(tree, ast.Module):
        for node in tree.body:
            if (
                isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                and node.name == name
            ):
                own_def = node
                break
    own_ids = set(map(id, ast.walk(own_def))) if own_def is not None else set()

    n = 0
    for node in ast.walk(tree):
        if id(node) in own_ids:
            continue
        if isinstance(node, ast.Name) and node.id == name:
            n += 1
        elif isinstance(node, ast.Attribute) and node.attr == name:
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    roots = [REPO / a for a in argv] if argv else [REPO / r for r in DEFAULT_ROOTS]
    rows: list[tuple[str, str, int, int, int]] = []
    for py in sorted(_iter_py(roots)):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for name, lineno, body_len in _private_helpers(tree):
            uses = _count_uses(tree, name)
            rows.append((str(py.relative_to(REPO)), name, lineno, body_len, uses))

    rows.sort(key=lambda r: (r[4], r[3]))
    print(f"{'file':<60} {'helper':<35} {'L':>5} {'body':>5} {'uses':>5}")
    print("-" * 115)
    n_unused = n_single = 0
    for rel, name, lineno, body_len, uses in rows:
        marker = " *" if uses == 0 else ("  ~" if uses == 1 else "")
        print(f"{rel:<60} {name:<35} {lineno:>5} {body_len:>5} {uses:>5}{marker}")
        if uses == 0:
            n_unused += 1
        elif uses == 1:
            n_single += 1
    print()
    print(
        f"# {len(rows)} private helpers; {n_unused} unused (*), {n_single} single-use (~)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

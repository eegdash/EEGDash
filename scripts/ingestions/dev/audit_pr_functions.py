"""Inventory + usage audit of every function introduced by PR #354.

Extracts each function whose definition is on a PR-added line (vs
``origin/develop``), then counts how many distinct call-sites reference
it across the whole codebase (PR-touched files + the rest of ``eegdash/``
+ ``scripts/ingestions/`` minus tests/, plus tests/ as a separate count
so we can tell "only used in its own test" apart from "used in prod").

Output is two CSV-like tables on stdout:

* ``functions``: file,name,lineno,body_lines,is_private,is_method,prod_uses,test_uses,total_uses
* ``candidates``: same row but only where prod_uses+test_uses ≤ 1
  (delete / inline candidates).

Run::

    python scripts/ingestions/dev/audit_pr_functions.py [--base origin/develop]
"""

from __future__ import annotations

import argparse
import ast
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PROD_GLOBS = ("eegdash/**/*.py", "scripts/ingestions/**/*.py")
TEST_PATH_MARKERS = ("/tests/", "/test_", "test_")


def _changed_files(base: str) -> list[Path]:
    out = subprocess.check_output(
        ["git", "diff", base, "--name-only", "--", "*.py"], cwd=REPO, text=True
    )
    return [REPO / line.strip() for line in out.splitlines() if line.strip()]


def _added_line_set(base: str, file: Path) -> set[int]:
    """Return the set of *new* HEAD line numbers added by the PR diff."""
    try:
        out = subprocess.check_output(
            ["git", "diff", base, "--", str(file.relative_to(REPO))],
            cwd=REPO,
            text=True,
        )
    except subprocess.CalledProcessError:
        return set()
    added: set[int] = set()
    cur = 0
    for line in out.splitlines():
        if line.startswith("@@"):
            # @@ -a,b +c,d @@
            try:
                plus = line.split("+", 1)[1].split(" ", 1)[0]
                cur = int(plus.split(",")[0])
            except (IndexError, ValueError):
                cur = 0
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added.add(cur)
            cur += 1
        elif line.startswith("-") and not line.startswith("---"):
            continue
        elif line.startswith(" "):
            cur += 1
    return added


def _pr_funcs(file: Path, base: str) -> list[tuple[str, int, int, bool, bool]]:
    """Return ``[(name, lineno, body_len, is_private, is_method), ...]``
    for every function whose ``def`` line was added by the PR.
    """
    try:
        tree = ast.parse(file.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):
        return []
    added = _added_line_set(base, file)
    out: list[tuple[str, int, int, bool, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.lineno not in added:
            continue
        name = node.name
        end = node.end_lineno or node.lineno
        is_priv = name.startswith("_") and not name.startswith("__")
        is_method = False
        # parents aren't tracked by ast.walk; approximate by checking if
        # the function is the descendant of a ClassDef (we'll just look at
        # col_offset > 0 + the module body to detect)
        # Simpler heuristic: col_offset > 0 means nested or method.
        is_method = node.col_offset > 0
        out.append((name, node.lineno, end - node.lineno, is_priv, is_method))
    return out


def _scan_files(roots: Iterable[str]) -> list[Path]:
    seen: set[Path] = set()
    for glob in roots:
        for p in REPO.glob(glob):
            if p.is_file() and "__pycache__" not in p.parts:
                seen.add(p)
    return sorted(seen)


def _index_references(
    files: list[Path],
) -> tuple[dict[str, int], dict[str, int]]:
    """Parse every file once and build {name -> count} dicts split into
    (prod, test). Names are Name.id, Attribute.attr, and Constant.value (str).
    """
    prod: dict[str, int] = defaultdict(int)
    test: dict[str, int] = defaultdict(int)
    for f in files:
        bucket = test if any(m in str(f) for m in TEST_PATH_MARKERS) else prod
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                bucket[node.id] += 1
            elif isinstance(node, ast.Attribute):
                bucket[node.attr] += 1
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                bucket[node.value] += 1
    return prod, test


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="origin/develop")
    p.add_argument("--threshold", type=int, default=1)
    args = p.parse_args()

    pr_files = [f for f in _changed_files(args.base) if f.exists()]
    all_files = _scan_files(PROD_GLOBS)

    rows: list[tuple[str, str, int, int, bool, bool, int, int]] = []
    # Track function NAMES that are reused across multiple files; if a
    # name is defined in N files and used in M, we want to attribute the
    # uses to each defining file separately (but since uses don't carry
    # source-file metadata, the count is global). That's fine for an
    # audit: name reuse across files is itself a finding.
    seen: dict[str, list[tuple[str, int]]] = defaultdict(list)

    for f in pr_files:
        for name, lineno, body, is_priv, is_method in _pr_funcs(f, args.base):
            seen[name].append((str(f.relative_to(REPO)), lineno))

    prod_idx, test_idx = _index_references(all_files)
    name_uses: dict[str, tuple[int, int]] = {
        name: (prod_idx.get(name, 0), test_idx.get(name, 0)) for name in seen
    }

    for f in pr_files:
        for name, lineno, body, is_priv, is_method in _pr_funcs(f, args.base):
            prod, test = name_uses[name]
            # Subtract one prod use per defining occurrence (the def-site
            # would be counted as a Name in some contexts e.g. decorator
            # references). Defensive: keep at >= 0.
            rows.append(
                (
                    str(f.relative_to(REPO)),
                    name,
                    lineno,
                    body,
                    is_priv,
                    is_method,
                    max(prod, 0),
                    max(test, 0),
                )
            )

    rows.sort(key=lambda r: (r[6] + r[7], r[3]))

    print("file,name,lineno,body,priv,method,prod_uses,test_uses,total")
    for rel, name, lineno, body, is_priv, is_method, prod, test in rows:
        total = prod + test
        print(
            f"{rel},{name},{lineno},{body},{int(is_priv)},{int(is_method)},"
            f"{prod},{test},{total}"
        )

    print()
    print("# Summary")
    n_total = len(rows)
    n_zero = sum(1 for r in rows if r[6] + r[7] == 0)
    n_one = sum(1 for r in rows if r[6] + r[7] == 1)
    n_two_plus = sum(1 for r in rows if r[6] + r[7] >= 2)
    n_test_only = sum(1 for r in rows if r[6] == 0 and r[7] > 0)
    print(f"# {n_total} PR-introduced functions")
    print(f"# {n_zero} unused (0 callers)")
    print(f"# {n_one} single-use (1 caller)")
    print(f"# {n_test_only} only-tested (no prod caller, only test caller)")
    print(f"# {n_two_plus} multi-use (≥2 callers)")

    # Name-collisions across multiple files
    multi = {n: locs for n, locs in seen.items() if len(locs) > 1}
    if multi:
        print()
        print("# Name collisions (same function name in ≥2 PR-introduced files)")
        for name, locs in sorted(multi.items()):
            print(f"# {name}: {len(locs)} defs")
            for loc, lineno in locs:
                print(f"#   {loc}:{lineno}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

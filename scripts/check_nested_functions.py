#!/usr/bin/env python3
"""Check for nested function definitions in Python files.

This script detects function definitions nested inside other functions,
which violates the project's style guide. Factory functions that return
callables are allowed (detected by checking if the inner function is returned).
"""

import ast
import sys
from pathlib import Path


class NestedFunctionChecker(ast.NodeVisitor):
    """AST visitor that detects nested function definitions."""

    def __init__(self, filename: str):
        self.filename = filename
        self.errors: list[tuple[int, str, str]] = []
        self._function_stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._check_nested(node)
        self._function_stack.append(node)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._check_nested(node)
        self._function_stack.append(node)
        self.generic_visit(node)
        self._function_stack.pop()

    def _check_nested(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        """Check if this function is nested and report if problematic."""
        if not self._function_stack:
            return  # Top-level function, OK

        outer_func = self._function_stack[-1]

        # Allow factory pattern: inner function is returned
        if self._is_returned(outer_func, node.name):
            return

        # Allow closures used in comprehensions or as callbacks
        # (these are harder to detect, so we check if it's used in a return)

        self.errors.append(
            (
                node.lineno,
                node.name,
                outer_func.name,
            )
        )

    def _is_returned(
        self, outer: ast.FunctionDef | ast.AsyncFunctionDef, inner_name: str
    ) -> bool:
        """Check if inner function is returned by outer function."""
        for stmt in ast.walk(outer):
            if isinstance(stmt, ast.Return) and stmt.value is not None:
                if isinstance(stmt.value, ast.Name) and stmt.value.id == inner_name:
                    return True
                # Also check for return of call to inner function
                if isinstance(stmt.value, ast.Call):
                    if (
                        isinstance(stmt.value.func, ast.Name)
                        and stmt.value.func.id == inner_name
                    ):
                        return True
        return False


def check_file(filepath: Path) -> list[str]:
    """Check a single file for nested functions."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        return [f"{filepath}:{e.lineno}: SyntaxError: {e.msg}"]

    checker = NestedFunctionChecker(str(filepath))
    checker.visit(tree)

    errors = []
    for lineno, inner_name, outer_name in checker.errors:
        errors.append(
            f"{filepath}:{lineno}: "
            f"nested function '{inner_name}' inside '{outer_name}' "
            f"(move to module level or use a class)"
        )
    return errors


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: check_nested_functions.py <file1.py> [file2.py ...]")
        return 1

    all_errors = []
    for filepath in sys.argv[1:]:
        path = Path(filepath)
        if path.suffix == ".py" and path.exists():
            errors = check_file(path)
            all_errors.extend(errors)

    if all_errors:
        for error in all_errors:
            print(error)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

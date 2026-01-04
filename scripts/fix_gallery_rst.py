#!/usr/bin/env python3
"""Fix Sphinx-Gallery RST spacing and heading issues."""

from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path
from typing import Iterable

HASH_RE = re.compile(r"^#{10,}$")
DIRECTIVE_RE = re.compile(r"^\\.\\.\\s+\\S")
UNDERLINE_CHARS = set("=~-^\"'`_+*:")


def _strip_single_leading_space(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        if line.startswith(" ") and not line.startswith("  "):
            cleaned.append(line[1:])
        else:
            cleaned.append(line)
    return cleaned


def _ensure_transition_spacing(lines: list[str]) -> list[str]:
    out: list[str] = []
    for idx, line in enumerate(lines):
        if HASH_RE.match(line):
            if out and out[-1].strip():
                out.append("")
            out.append(line)
            next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
            if next_line.strip():
                out.append("")
        else:
            out.append(line)
    return out


def _ensure_blank_after_directives(lines: list[str]) -> list[str]:
    out: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        out.append(line)
        if DIRECTIVE_RE.match(line):
            idx += 1
            while idx < len(lines):
                next_line = lines[idx]
                if not next_line.strip() or next_line.startswith(" "):
                    out.append(next_line)
                    idx += 1
                    continue
                break
            if idx < len(lines) and out and out[-1].strip():
                out.append("")
            continue
        idx += 1
    return out


def _ensure_blank_after_indented_blocks(lines: list[str]) -> list[str]:
    out: list[str] = []
    for idx, line in enumerate(lines):
        out.append(line)
        if line.startswith(" ") and line.strip():
            next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
            if next_line and next_line.strip() and not next_line.startswith(" "):
                out.append("")
    return out


def _is_underline(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped[0] == "#":
        return False
    return len(set(stripped)) == 1 and stripped[0] in UNDERLINE_CHARS


def _fix_heading_underlines(lines: list[str]) -> list[str]:
    out = list(lines)
    for idx in range(len(out) - 1):
        title = out[idx]
        underline = out[idx + 1]
        if not title.strip() or title.lstrip().startswith(".. "):
            continue
        if not _is_underline(underline):
            continue
        title_indent = len(title) - len(title.lstrip())
        underline_char = underline.strip()[0]
        title_text = title.strip()
        underline_text = underline_char * max(len(title_text), len(underline.strip()))
        out[idx + 1] = (" " * title_indent) + underline_text
    return out


def fix_rst_text(text: str) -> str:
    has_trailing_newline = text.endswith("\n")
    lines = text.splitlines()
    lines = _strip_single_leading_space(lines)
    lines = _ensure_transition_spacing(lines)
    lines = _ensure_blank_after_directives(lines)
    lines = _ensure_blank_after_indented_blocks(lines)
    lines = _fix_heading_underlines(lines)
    fixed = "\n".join(lines)
    if has_trailing_newline:
        fixed += "\n"
    return fixed


def _iter_paths(items: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.rst")))
        else:
            paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fix spacing issues in Sphinx-Gallery RST output."
    )
    parser.add_argument("paths", nargs="+", help="RST files or directories.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Show diffs without writing files; exits non-zero if changes needed.",
    )
    args = parser.parse_args()

    changed = False
    for path in _iter_paths(args.paths):
        if not path.exists() or path.suffix != ".rst":
            continue
        original = path.read_text(encoding="utf-8")
        fixed = fix_rst_text(original)
        if fixed != original:
            changed = True
            if args.check:
                diff = difflib.unified_diff(
                    original.splitlines(),
                    fixed.splitlines(),
                    fromfile=str(path),
                    tofile=str(path),
                    lineterm="",
                )
                print("\n".join(diff))
            else:
                path.write_text(fixed, encoding="utf-8")

    if args.check and changed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

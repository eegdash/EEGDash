#!/usr/bin/env python3
"""Bump __version__ to a PEP 440 dev release.

Rules
-----
- Read current __version__ from the given init file (default: eegdash/__init__.py)
- Strip any pre/dev/rc/post/local parts to get the base release (e.g., 1.2.3 from 1.2.3.dev5)
- If --pr N is given  -> write __version__ = "<base>.dev<N>"
- Else if --suffix S  -> deterministically map S to an integer N and write __version__ = "<base>.dev<N>"
- With --dry-run, only print the computed version (no write).

Examples
--------
  python scripts/release_script.py --pr 123
  python scripts/release_script.py --suffix shaabcd12
  python scripts/release_script.py --init-file path/to/__init__.py --pr 42

"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

try:
    from packaging.version import Version
except Exception:
    print(
        "ERROR: 'packaging' is required. Install with: pip install packaging",
        file=sys.stderr,
    )
    sys.exit(2)


def _numeric_from_suffix(s: str) -> int:
    """Turn an arbitrary string into a stable positive integer.

    Strategy (PEP 440 requires an integer for .devN/.postN):
    1) If s is purely digits -> int(s)
    2) If s contains a hex run of >=5 chars (e.g., a git SHA) -> take the first 7 and base-16 decode
    3) Otherwise -> take sha1(s) and decode first 8 hex chars to int
    """
    s = s.strip()
    if s.isdigit():
        return int(s)

    m = re.search(r"([0-9a-fA-F]{5,})", s)
    if m:
        hex_run = m.group(1)[:7]  # keep it compact
        return int(hex_run, 16)

    # Fully generic and stable fallback
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--init-file",
        default="eegdash/__init__.py",
        help="Path to the module __init__.py containing __version__",
    )
    ap.add_argument(
        "--pr",
        type=int,
        required=False,
        help="Pull Request number to use in the .dev<PR> suffix",
    )
    ap.add_argument(
        "--suffix",
        type=str,
        required=False,
        help="Arbitrary string used to derive a numeric .devN when no --pr is provided (e.g. 'shaabcd12')",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print new version without writing",
    )
    args = ap.parse_args()

    if args.pr is None and not args.suffix:
        ap.error("Either --pr or --suffix must be provided")  # exits with code 2

    init_path = Path(args.init_file)
    if not init_path.exists():
        print(
            f"ERROR: Cannot find {init_path} (needed to set __version__)",
            file=sys.stderr,
        )
        return 1

    text = init_path.read_text(encoding="utf-8")
    m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not m:
        print(f"ERROR: __version__ not found in {init_path}", file=sys.stderr)
        return 1

    current = m.group(1)
    try:
        base_release = ".".join(map(str, Version(current).release))
    except Exception as e:
        print(
            f"ERROR: Could not parse current version '{current}': {e}", file=sys.stderr
        )
        return 1

    if args.pr is not None:
        n = int(args.pr)
    else:
        n = _numeric_from_suffix(args.suffix)

    new_version = f"{base_release}.dev{n}"

    # Replace only the captured version value
    new_text = text[: m.start(1)] + new_version + text[m.end(1) :]

    if args.dry_run:
        print(new_version)
        return 0

    init_path.write_text(new_text, encoding="utf-8")
    print(new_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

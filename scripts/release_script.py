#!/usr/bin/env python3
"""Bump __version__ to a PEP 440 dev release: <base_release>.dev<PR_NUMBER>

- Reads current __version__ from the given init file (default: eegdash/__init__.py)
- Strips any pre/post/dev/rc parts to get the base release
- Writes back __version__ = "<base>.dev<PR_NUMBER>"

Usage:
  python scripts/bump_pr_dev_version.py --pr 123
  python scripts/bump_pr_dev_version.py --init-file path/to/__init__.py --pr 123
"""

import argparse
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
        required=True,
        help="Pull Request number to use in the .dev<PR> suffix",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print new version without writing",
    )
    args = ap.parse_args()

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

    new_version = f"{base_release}.dev{args.pr}"

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

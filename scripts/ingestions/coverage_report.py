#!/usr/bin/env python3
"""Emit a per-field technical-metadata coverage report for a digestion output dir.

Answers "what is missing across all datasets, and where did each value come
from" — reading only the ``*_records.json`` (no signal access).

Usage::

    python coverage_report.py digestion_output
    python coverage_report.py digestion_output --json coverage.json

To measure at NEMAR + OpenNeuro scale, first run the digest stage over all
datasets (stages 1-3 on a shallow clone; see ``2_clone.py`` / ``3_digest.py``),
then point this at the resulting output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running both as ``python coverage_report.py`` and ``python -m``.
sys.path.insert(0, str(Path(__file__).parent))

from _coverage import aggregate_output_dir, format_coverage_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Digestion output directory (contains <id>/<id>_records.json).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write the full coverage report as JSON.",
    )
    args = parser.parse_args(argv)

    if not args.output_dir.is_dir():
        parser.error(f"not a directory: {args.output_dir}")

    report = aggregate_output_dir(args.output_dir)
    print(format_coverage_summary(report))

    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nFull report written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

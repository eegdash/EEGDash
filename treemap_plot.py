#!/usr/bin/env python3

"""Build a Plotly treemap for the EEG/MEG dataset summary CSV."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def _find_dataset_csv(base_path: Path) -> Path:
    """Return the dataset summary path, preferring a CSV next to the script."""
    local_csv = base_path / "dataset_summary.csv"
    if local_csv.exists():
        return local_csv

    bundled_csv = base_path / "eegdash" / "dataset" / "dataset_summary.csv"
    if bundled_csv.exists():
        return bundled_csv

    msg = "dataset_summary.csv not found next to the script or under eegdash/dataset."
    raise FileNotFoundError(msg)


def main() -> None:
    base_path = Path(__file__).resolve().parent
    docs_dir = base_path / "docs"
    if docs_dir.exists():
        sys.path.insert(0, str(docs_dir))

    try:
        from plot_dataset.treemap import generate_dataset_treemap
    except ImportError as exc:  # pragma: no cover - guard for CLI usage
        raise SystemExit(f"Unable to import treemap generator: {exc}") from exc

    dataset_csv = _find_dataset_csv(base_path)
    df = pd.read_csv(dataset_csv)

    output = base_path / "treemap.html"
    generate_dataset_treemap(df, output)
    print(f"Treemap saved to {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - guard for CLI usage
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

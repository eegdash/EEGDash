"""Datasets with no signal files, or deleted upstream, are never injected.

The 2026-10-17 catalogue diagnostic found 9 dataset docs that nobody can load:
4 deposits with no recording at all and 5 OpenNeuro datasets deleted upstream.
They were removed from the DB; inject must not bring any of them back from an
old digest in eegdash-dataset-listings.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from _inject_plan import find_digested_datasets

_GONE = (
    "ds002799",
    "ds005083",
    "ds006253",
    "nm000150",
    "ds004929",
    "ds005407",
    "ds005930",
    "ds006940",
    "ds007630",
)
_CSV = Path(__file__).resolve().parents[4] / "eegdash/dataset/dataset_summary.csv"


@pytest.mark.parametrize("dataset_id", _GONE)
def test_inject_skips_dataset(tmp_path, dataset_id):
    for ds in (dataset_id, "nm000108"):
        (tmp_path / ds).mkdir()
        (tmp_path / ds / f"{ds}_dataset.json").write_text(json.dumps({"id": ds}))
    found = [d.name for d in find_digested_datasets(tmp_path)]
    assert found == ["nm000108"]


def test_summary_csv_drops_them():
    with _CSV.open(newline="") as f:
        listed = {row["dataset"] for row in csv.DictReader(f)}
    assert listed.isdisjoint(_GONE)

"""The digested dataset document keeps the paper DOI found at stage 1.

Stage 1 puts ``associated_paper_doi`` on the source document (NEMAR: DataCite
related identifiers with a paper relation; OpenNeuro: ``associatedPaperDOI``)
and stage 2 copies it into manifest.json, but ``extract_dataset_metadata``
never passed it on, so every injected document carried
``associated_paper_doi: null`` and dataset pages had no "Read paper" link
unless the README held a paper badge.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import _dataset_metadata as dm


class _NoFilesBids:
    """Minimal stand-in for EEGBIDSDataset: a BIDS root without recordings."""

    def __init__(self, root: Path):
        self.bidsdir = str(root)

    def get_files(self):
        return []

    def get_bids_file_attribute(self, _attribute, _bids_file):
        return None


def _digest(root: Path, manifest: dict | None, dataset_doi: str | None = None):
    description = {"Name": "fNIRS dataset", "BIDSVersion": "1.10.0"}
    if dataset_doi:
        description["DatasetDOI"] = dataset_doi
    (root / "dataset_description.json").write_text(json.dumps(description))
    return dm.extract_dataset_metadata(
        _NoFilesBids(root), "nm000175", "nemar", "2026-11-13T00:00:00Z", manifest
    )


@pytest.mark.parametrize(
    ("stage1_value", "expected"),
    [
        ("10.1016/j.neuroimage.2021.118000", "10.1016/j.neuroimage.2021.118000"),
        ("https://doi.org/10.1093/cercor/bhab123", "10.1093/cercor/bhab123"),
        ("https://dx.doi.org/10.1093/cercor/bhab123", "10.1093/cercor/bhab123"),
        ("doi:10.1038/s41597-022-01234-5.", "10.1038/s41597-022-01234-5"),
        ("DOI: 10.1038/s41597-022-01234-5", "10.1038/s41597-022-01234-5"),
        ("10.1016/0006-8993(85)90003-3", "10.1016/0006-8993(85)90003-3"),
        ("10.3389/fnhum.2020.00123)", "10.3389/fnhum.2020.00123"),
        # A list of DOIs: the first, as NEMAR's related identifiers give it.
        ("10.1016/j.cell.2020.1, 10.1038/nn.1234", "10.1016/j.cell.2020.1"),
        ("10.1016/j.cell.2020.1;doi:10.1038/nn.1234", "10.1016/j.cell.2020.1"),
    ],
)
def test_stage1_paper_doi_reaches_the_document(tmp_path: Path, stage1_value, expected):
    doc = _digest(tmp_path, {"associated_paper_doi": stage1_value})
    assert doc["associated_paper_doi"] == expected


def test_paper_doi_equal_to_dataset_doi_is_dropped(tmp_path: Path):
    doc = _digest(
        tmp_path,
        {"associated_paper_doi": "https://doi.org/10.82901/nemar.nm000175"},
        dataset_doi="doi:10.82901/nemar.nm000175",
    )
    assert doc["associated_paper_doi"] is None


@pytest.mark.parametrize(
    "stage1_value",
    [
        "N/A",
        "https://example.org/article/1",
        42,
        # Free text or a publisher URL that embeds a DOI: no DOI is made up.
        "(see 10.3389/fnhum.2020.00123)",
        "[10.1016/j.cell.2020.1]",
        "https://www.biorxiv.org/content/10.1101/2020.01.01.123456v1.full",
        "10.1016/j.cell.2020.1 (preprint)",
    ],
)
def test_values_without_a_doi_give_none(tmp_path: Path, stage1_value):
    doc = _digest(tmp_path, {"associated_paper_doi": stage1_value})
    assert doc["associated_paper_doi"] is None


# --- unchanged behaviour (controls) -----------------------------------------


def test_no_manifest_gives_no_paper_doi(tmp_path: Path):
    assert _digest(tmp_path, None)["associated_paper_doi"] is None


def test_manifest_without_paper_doi_gives_none(tmp_path: Path):
    doc = _digest(tmp_path, {"senior_author": "Doe, J."})
    assert doc["associated_paper_doi"] is None
    assert doc["senior_author"] == "Doe, J."

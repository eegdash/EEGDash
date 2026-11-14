"""The digested dataset document carries the paper that describes the dataset.

Stage 1 puts ``associated_paper_doi`` on the source document (OpenNeuro:
``associatedPaperDOI``; NEMAR: a DataCite related identifier with a describing
relation) and stage 2 copies it into manifest.json, but
``extract_dataset_metadata`` never passed it on, so every injected document
carried ``associated_paper_doi: null`` and dataset pages had no "Read paper"
link unless the README held a paper badge.

The paper is now, in order: the README "Paper DOI" badge (the paper line the
deposit declares); else the deposit's paper-match record, which decides when
present (only a confirmed paper counts); else, for OpenNeuro, the paper its
authors declared. NEMAR's catalogue paper (DataCite relations) is no
confirmation, so a NEMAR dataset with neither a badge nor a confirmed record
has no paper: no link rather than a wrong, methods or toolbox paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import _dataset_metadata as dm

_BADGE = "[![Paper DOI](https://img.shields.io/badge/Paper-DOI-blue)]({url})"
_DATASET_IDS = {"openneuro": "ds000001", "nemar": "nm000001"}


class _NoFilesBids:
    """Minimal stand-in for EEGBIDSDataset: a BIDS root without recordings."""

    def __init__(self, root: Path):
        self.bidsdir = str(root)

    def get_files(self):
        return []

    def get_bids_file_attribute(self, _attribute, _bids_file):
        return None


def _match(verdict: str | None, doi: str | None) -> dict:
    return {"association": {"verdict": verdict}, "document": {"doi": doi}}


_RECORD_SOURCE = (
    "https://api.datacite.org/dois/10.5555%2Fdata.record [repository record "
    "(DataCite / repository API); paper doi:10.5555/data.record; record > description]"
)
_PAPER_SOURCE = "https://doi.org/10.1000/paper [Europe PMC full text PMC1]"


def _confirmed_record(
    *,
    doc_doi: str | None,
    doc_kind: str,
    confirmed_paper: dict | None = None,
    sources: tuple[str, ...] = (),
) -> dict:
    """A confirmed paper-match record whose document is a paper or a data record."""
    record = {
        "association": {
            "verdict": "confirmed",
            "evidence": [{"what": "N", "source": source} for source in sources],
        },
        "document": {"doi": doc_doi, "kind": doc_kind},
    }
    if confirmed_paper is not None:
        record["_paperenrich"] = {"confirmed_paper": confirmed_paper}
    return record


def _digest(
    root: Path,
    manifest: dict | None,
    dataset_doi: str | None = None,
    *,
    source: str = "openneuro",
    readme: str | None = None,
    match: dict | str | None = None,
):
    description = {"Name": "A dataset", "BIDSVersion": "1.10.0"}
    if dataset_doi:
        description["DatasetDOI"] = dataset_doi
    (root / "dataset_description.json").write_text(json.dumps(description))
    if readme is not None:
        (root / "README").write_text(f"# A dataset\n\n{readme}\n\nText.\n")
    if match is not None:
        path = root / "code" / "paper_match" / "match.json"
        path.parent.mkdir(parents=True)
        path.write_text(match if isinstance(match, str) else json.dumps(match))
    return dm.extract_dataset_metadata(
        _NoFilesBids(root),
        _DATASET_IDS[source],
        source,
        "2026-11-13T00:00:00Z",
        manifest,
    )


# --- the stage-1 value (an OpenNeuro dataset) -------------------------------


@pytest.mark.parametrize(
    ("stage1_value", "expected"),
    [
        ("10.1016/j.neuroimage.2021.118000", "10.1016/j.neuroimage.2021.118000"),
        ("https://doi.org/10.1093/cercor/bhab123", "10.1093/cercor/bhab123"),
        ("https://dx.doi.org/10.1093/cercor/bhab123", "10.1093/cercor/bhab123"),
        ("https://doi.org/10.1093/cercor/bhab123?utm=1", "10.1093/cercor/bhab123"),
        ("https://doi.org/10.1093/cercor/bhab123#refs", "10.1093/cercor/bhab123"),
        ("doi:10.1038/s41597-022-01234-5.", "10.1038/s41597-022-01234-5"),
        ("DOI: 10.1038/s41597-022-01234-5", "10.1038/s41597-022-01234-5"),
        ("10.1016/0006-8993(85)90003-3", "10.1016/0006-8993(85)90003-3"),
        ("10.3389/fnhum.2020.00123)", "10.3389/fnhum.2020.00123"),
        # A list of DOIs: the first.
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
        {"associated_paper_doi": "https://doi.org/10.1000/ds000001.v1.0.0"},
        dataset_doi="doi:10.1000/ds000001.v1.0.0",
    )
    assert doc["associated_paper_doi"] is None


def test_another_version_of_the_dataset_doi_is_dropped(tmp_path: Path):
    doc = _digest(
        tmp_path,
        {"associated_paper_doi": "10.1000/ds000001.v1.0.1"},
        dataset_doi="10.1000/ds000001.v1.0.0",
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


# --- which paper describes the dataset ---------------------------------------


@pytest.mark.parametrize("source", ["nemar", "openneuro"])
def test_readme_paper_badge_wins_over_the_catalogue_doi(tmp_path: Path, source):
    doc = _digest(
        tmp_path,
        {"associated_paper_doi": "10.1000/catalogue"},
        source=source,
        readme=_BADGE.format(url="https://doi.org/10.1000/readme"),
    )
    assert doc["associated_paper_doi"] == "10.1000/readme"


def test_readme_paper_badge_wins_over_the_paper_match(tmp_path: Path):
    doc = _digest(
        tmp_path,
        {"associated_paper_doi": "10.1000/catalogue"},
        source="nemar",
        readme=_BADGE.format(url="https://doi.org/10.1000/readme"),
        match=_match("confirmed", "10.1000/confirmed"),
    )
    assert doc["associated_paper_doi"] == "10.1000/readme"


@pytest.mark.parametrize("source", ["nemar", "openneuro"])
def test_confirmed_paper_match_gives_its_doi(tmp_path: Path, source):
    doc = _digest(
        tmp_path,
        {"associated_paper_doi": "10.1000/catalogue"},
        source=source,
        match=_match("confirmed", "https://doi.org/10.1000/confirmed"),
    )
    assert doc["associated_paper_doi"] == "10.1000/confirmed"


@pytest.mark.parametrize("verdict", ["probable", "wrong-paper", "no-paper", None])
def test_unconfirmed_paper_match_gives_no_paper(tmp_path: Path, verdict):
    # The record says no paper is confirmed: the catalogue's paper is not
    # shown either (it may be the wrong paper, or a methods paper).
    doc = _digest(
        tmp_path,
        {"associated_paper_doi": "10.1000/catalogue"},
        source="nemar",
        match=_match(verdict, "10.1000/other"),
    )
    assert doc["associated_paper_doi"] is None


@pytest.mark.parametrize(
    "match",
    [
        _match("confirmed", None),
        _match("confirmed", "not a doi"),
        "{not json",
        "[]",
        {"association": "confirmed"},
    ],
)
def test_paper_match_without_a_confirmed_doi_gives_no_paper(tmp_path: Path, match):
    doc = _digest(
        tmp_path,
        {"associated_paper_doi": "10.1000/catalogue"},
        source="nemar",
        match=match,
    )
    assert doc["associated_paper_doi"] is None


# A confirmed record whose document is the data record the match read: the
# paper comes from _paperenrich.confirmed_paper or from the evidence, and the
# data record is never shown as the paper, in any version.


def test_data_record_document_takes_the_enriched_paper(tmp_path: Path):
    record = _confirmed_record(
        doc_doi="10.5555/data.record",
        doc_kind="repository-record",
        confirmed_paper={"doi": "https://doi.org/10.1000/paper"},
        sources=(_RECORD_SOURCE,),
    )
    doc = _digest(tmp_path, {}, source="nemar", match=record)
    assert doc["associated_paper_doi"] == "10.1000/paper"


def test_data_record_document_takes_a_paper_the_evidence_quotes(tmp_path: Path):
    record = _confirmed_record(
        doc_doi="10.5555/data.record",
        doc_kind="repository-record",
        sources=(_RECORD_SOURCE, _PAPER_SOURCE),
    )
    doc = _digest(tmp_path, {}, source="nemar", match=record)
    assert doc["associated_paper_doi"] == "10.1000/paper"


@pytest.mark.parametrize(
    ("confirmed_paper", "sources"),
    [
        # Only the data record was read.
        (None, (_RECORD_SOURCE,)),
        # A paper entry without a DOI (e.g. proceedings with a URL only).
        ({"doi": None, "url": "https://example.org/paper.pdf"}, (_RECORD_SOURCE,)),
        # Another version of the data record, as the enriched paper ...
        ({"doi": "10.5555/data.record.v2"}, (_RECORD_SOURCE,)),
        # ... or as a source outside a repository record.
        (None, ("https://doi.org/10.5555/data.concept [landing page]",)),
    ],
)
def test_data_record_document_without_a_paper_gives_none(
    tmp_path: Path, confirmed_paper, sources
):
    record = _confirmed_record(
        doc_doi="10.5555/data.record",
        doc_kind="repository-record",
        confirmed_paper=confirmed_paper,
        sources=sources,
    )
    doc = _digest(
        tmp_path,
        {"associated_paper_doi": "10.1000/catalogue"},
        source="nemar",
        match=record,
    )
    assert doc["associated_paper_doi"] is None


def test_data_record_named_by_its_source_kind_only(tmp_path: Path):
    # A record without "kind" whose source_kind says it read a repository record.
    record = _confirmed_record(
        doc_doi="10.5555/data.record",
        doc_kind="repository-record",
        sources=(_RECORD_SOURCE,),
    )
    del record["document"]["kind"]
    record["document"]["source_kind"] = "repository record (DataCite / repository API)"
    doc = _digest(tmp_path, {}, source="nemar", match=record)
    assert doc["associated_paper_doi"] is None


def test_enriched_paper_preferred_over_a_paper_document(tmp_path: Path):
    record = _confirmed_record(
        doc_doi="10.1000/preprint",
        doc_kind="paper",
        confirmed_paper={"doi": "10.1000/published"},
    )
    doc = _digest(tmp_path, {}, source="nemar", match=record)
    assert doc["associated_paper_doi"] == "10.1000/published"


def test_paper_document_keeps_its_doi(tmp_path: Path):
    # The evidence is only searched when the document is the data record, so
    # a confirmed paper is not swapped for another text the quotes came from.
    record = _confirmed_record(
        doc_doi="10.1000/journal-article",
        doc_kind="paper",
        confirmed_paper={"title": "no doi here"},
        sources=("https://example.org/x [PMC; paper doi:10.1000/preprint; p. 2]",),
    )
    doc = _digest(tmp_path, {}, source="nemar", match=record)
    assert doc["associated_paper_doi"] == "10.1000/journal-article"


def test_nemar_catalogue_paper_alone_gives_no_paper(tmp_path: Path):
    # No README badge and no paper-match record: NEMAR's DataCite
    # IsDescribedBy DOI is unconfirmed, so no paper is shown.
    doc = _digest(
        tmp_path,
        {"associated_paper_doi": "10.1000/described-by"},
        source="nemar",
        readme="A README without a paper line.",
    )
    assert doc["associated_paper_doi"] is None


@pytest.mark.parametrize("source", ["nemar", "openneuro"])
def test_readme_badge_without_a_doi_is_skipped(tmp_path: Path, source):
    doc = _digest(
        tmp_path,
        {"associated_paper_doi": "10.1000/catalogue"},
        source=source,
        readme=_BADGE.format(url="https://example.org/article"),
    )
    expected = None if source == "nemar" else "10.1000/catalogue"
    assert doc["associated_paper_doi"] == expected


def test_readme_badge_with_the_dataset_doi_is_skipped(tmp_path: Path):
    doc = _digest(
        tmp_path,
        {"associated_paper_doi": "10.1000/catalogue"},
        dataset_doi="10.1000/ds000001.v1.0.0",
        readme=_BADGE.format(url="https://doi.org/10.1000/ds000001.v1.0.0"),
    )
    assert doc["associated_paper_doi"] == "10.1000/catalogue"


# --- unchanged behaviour (controls) -----------------------------------------


def test_no_manifest_gives_no_paper_doi(tmp_path: Path):
    assert _digest(tmp_path, None)["associated_paper_doi"] is None


def test_manifest_without_paper_doi_gives_none(tmp_path: Path):
    doc = _digest(tmp_path, {"senior_author": "Doe, J."})
    assert doc["associated_paper_doi"] is None
    assert doc["senior_author"] == "Doe, J."


def test_readme_without_badge_keeps_the_openneuro_doi(tmp_path: Path):
    doc = _digest(
        tmp_path,
        {"associated_paper_doi": "10.1000/catalogue"},
        readme="See https://doi.org/10.1000/mentioned for the methods.",
    )
    assert doc["associated_paper_doi"] == "10.1000/catalogue"

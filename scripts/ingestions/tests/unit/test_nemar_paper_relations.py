"""NEMAR stage 1: only a describing relation names the dataset's paper.

``_paper_doi`` took the first DataCite related identifier whose relation was
IsSupplementTo, IsDescribedBy or References, in list order. A dataset also
"References" methods, toolbox and preprint papers, which pages would then show
as "Read paper". Now IsDescribedBy (preferred) and IsSupplementTo name the
paper, wherever they sit in the list, and anything else gives no paper.
"""

from __future__ import annotations

import pytest
from _helpers import load_module


@pytest.fixture(scope="module")
def nemar():
    return load_module("1_fetch_sources/nemar.py")


def _rel(identifier, relation, kind="DOI"):
    return {
        "identifier": identifier,
        "identifier_type": kind,
        "relation_type": relation,
    }


def test_describing_paper_wins_over_a_reference_listed_first(nemar):
    related = [
        _rel("10.1000/toolbox", "References"),
        _rel("10.1000/paper", "IsDescribedBy"),
    ]
    assert nemar._paper_doi(related) == "10.1000/paper"


def test_is_described_by_preferred_over_is_supplement_to(nemar):
    related = [
        _rel("10.1000/supplement", "IsSupplementTo"),
        _rel("10.1000/described", "IsDescribedBy"),
    ]
    assert nemar._paper_doi(related) == "10.1000/described"


def test_is_supplement_to_used_without_is_described_by(nemar):
    related = [
        _rel("10.1000/toolbox", "References"),
        _rel("https://doi.org/10.1000/supplement", "IsSupplementTo"),
    ]
    assert nemar._paper_doi(related) == "10.1000/supplement"


@pytest.mark.parametrize(
    "relation", ["References", "IsDerivedFrom", "Cites", "IsReferencedBy", "", None]
)
def test_other_relations_give_no_paper(nemar, relation):
    assert nemar._paper_doi([_rel("10.1000/toolbox", relation)]) is None


def test_url_identifiers_are_not_papers(nemar):
    related = [
        _rel("https://github.com/nemarDatasets/nm000001", "IsDescribedBy", kind="URL"),
    ]
    assert nemar._paper_doi(related) is None


def test_deposit_layout_with_a_preprint_listed_before_the_paper(nemar):
    # How NEMAR lists a re-hosted dataset: the source record it derives from,
    # a preprint it references, the paper, and the repository URLs.
    related = [
        _rel("10.1000/source-record", "IsDerivedFrom"),
        _rel("10.1000/preprint", "References"),
        _rel("10.1000/paper", "IsDescribedBy"),
        _rel("https://github.com/nemarDatasets/nm000001", "IsDescribedBy", kind="URL"),
    ]
    assert nemar._paper_doi(related) == "10.1000/paper"


def test_deposit_layout_without_a_describing_doi(nemar):
    # Only the source record, the repository URLs and a referenced toolbox
    # paper: no paper.
    related = [
        _rel("10.1000/source-record", "IsDerivedFrom"),
        _rel("https://github.com/nemarDatasets/nm000001", "IsDescribedBy", kind="URL"),
        _rel("10.1000/toolbox", "References"),
    ]
    assert nemar._paper_doi(related) is None


def test_no_related_identifiers(nemar):
    assert nemar._paper_doi(None) is None
    assert nemar._paper_doi([]) is None


def test_dataset_document_carries_the_describing_paper(nemar):
    metadata = {
        "dataset_id": "nm000001",
        "related_identifiers": [
            _rel("10.1000/toolbox", "References"),
            _rel("10.1000/paper", "IsDescribedBy"),
        ],
    }
    doc = nemar.build_dataset_document(metadata, catalogue_entry={"id": "nm000001"})
    assert doc["associated_paper_doi"] == "10.1000/paper"


def test_dataset_document_has_no_paper_from_references_alone(nemar):
    metadata = {
        "dataset_id": "nm000001",
        "related_identifiers": [_rel("10.1000/toolbox", "References")],
    }
    doc = nemar.build_dataset_document(metadata, catalogue_entry={"id": "nm000001"})
    assert doc["associated_paper_doi"] is None

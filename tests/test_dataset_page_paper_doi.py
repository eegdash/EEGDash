"""A dataset's paper DOI gives its page a "Paper" link, not only a README badge.

The field card already turns ``associated_paper_doi`` into "Read paper" and a
"Paper DOI" row. The provenance strip's "Paper" mirror link read only
``external_links.paper_url`` or a README ``[![Paper DOI](...)](...)`` badge,
so NEMAR pages whose paper DOI comes from the catalogue had no such link. The
API call is stubbed, so the tests stay offline.
"""

from __future__ import annotations

import io
import json

import pytest
from tests_support.dataset_page import load_dataset_page_module


@pytest.fixture
def data_loaders(monkeypatch):
    module = load_dataset_page_module("data_loaders")
    monkeypatch.delenv("EEGDASH_NO_API", raising=False)
    return module


@pytest.fixture
def sections():
    return load_dataset_page_module("sections")


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _stub_summary(monkeypatch, module, doc):
    payload = json.dumps({"success": True, "data": doc}).encode()
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda _url, timeout=10: _Response(payload)
    )


def test_paper_doi_becomes_the_paper_link(data_loaders, monkeypatch):
    _stub_summary(
        monkeypatch,
        data_loaders,
        {
            "dataset_id": "nm000175",
            "associated_paper_doi": "10.1016/j.neuroimage.2021.118000",
            "external_links": {"paper_url": None},
            "readme": "No badge here.",
        },
    )
    details = data_loaders._fetch_dataset_details_from_api("nm000175")
    assert details["paper_url"] == "https://doi.org/10.1016/j.neuroimage.2021.118000"


def test_structured_paper_url_still_wins(data_loaders, monkeypatch):
    _stub_summary(
        monkeypatch,
        data_loaders,
        {
            "dataset_id": "nm000175",
            "associated_paper_doi": "10.1016/j.neuroimage.2021.118000",
            "external_links": {"paper_url": "https://example.org/paper"},
        },
    )
    details = data_loaders._fetch_dataset_details_from_api("nm000175")
    assert details["paper_url"] == "https://example.org/paper"


def test_paper_doi_wins_over_readme_badge(data_loaders, monkeypatch):
    badge = (
        "[![Paper DOI](https://img.shields.io/badge/Paper-DOI-blue)]"
        "(https://doi.org/10.1093/cercor/bhab123)"
    )
    _stub_summary(
        monkeypatch,
        data_loaders,
        {
            "dataset_id": "nm000175",
            "associated_paper_doi": "10.1016/j.neuroimage.2021.118000",
            "external_links": {},
            "readme": badge,
        },
    )
    details = data_loaders._fetch_dataset_details_from_api("nm000175")
    assert details["paper_url"] == "https://doi.org/10.1016/j.neuroimage.2021.118000"


def test_url_form_paper_doi_is_not_prefixed_twice(data_loaders, monkeypatch):
    _stub_summary(
        monkeypatch,
        data_loaders,
        {
            "dataset_id": "nm000175",
            "associated_paper_doi": "https://doi.org/10.1093/cercor/bhab123",
            "external_links": {},
        },
    )
    details = data_loaders._fetch_dataset_details_from_api("nm000175")
    assert details["paper_url"] == "https://doi.org/10.1093/cercor/bhab123"


def test_readme_badge_used_without_paper_doi(data_loaders, monkeypatch):
    badge = (
        "[![Paper DOI](https://img.shields.io/badge/Paper-DOI-blue)]"
        "(https://doi.org/10.1093/cercor/bhab123)"
    )
    _stub_summary(
        monkeypatch,
        data_loaders,
        {"dataset_id": "nm000175", "external_links": {}, "readme": badge},
    )
    details = data_loaders._fetch_dataset_details_from_api("nm000175")
    assert details["paper_url"] == "https://doi.org/10.1093/cercor/bhab123"


def test_no_paper_link_without_doi_or_badge(data_loaders, monkeypatch):
    _stub_summary(
        monkeypatch,
        data_loaders,
        {"dataset_id": "nm000175", "external_links": {}, "readme": "Plain."},
    )
    assert data_loaders._fetch_dataset_details_from_api("nm000175")["paper_url"] == ""


def test_field_card_and_provenance_link_the_doi(sections):
    # Rendering control: given the context the loader now builds, both the
    # field card and the provenance strip link the paper.
    context = {
        "class_name": "NM000175",
        "dataset_id": "nm000175",
        "associated_paper_doi": "10.1016/j.neuroimage.2021.118000",
        "paper_url": "https://doi.org/10.1016/j.neuroimage.2021.118000",
    }
    href = 'href="https://doi.org/10.1016/j.neuroimage.2021.118000"'
    card = sections._format_editorial_fieldcard_section(context)
    assert f"<a {href}>Read paper</a>" in card
    assert "<dt>Paper DOI</dt>" in card
    strip = sections._format_editorial_provenance_section(context)
    assert f"<a {href}>Paper</a>" in strip

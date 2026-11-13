r"""The dataset-page year never comes out as a two-digit century fragment.

When a README has no citation-style "(YYYY)", the page fell back to
``re.findall(r"\b(19|20)\d{2}\b", readme)``. ``findall`` returns the capture
group, so a README mentioning "2019" gave the year "20" in the citation,
byline, info table and JSON-LD. NEMAR documents carry no
``dataset_created_at``, so every NEMAR page without "(YYYY)" in its README
went through this fallback.
"""

from __future__ import annotations

import pytest
from tests_support.dataset_page import load_dataset_page_module


@pytest.fixture
def data_loaders():
    return load_dataset_page_module("data_loaders")


@pytest.mark.parametrize(
    "readme,expected",
    [
        ("Recorded in 2019 at the lab.", "2019"),
        ("Data from 1998 and 2004.", "1998"),
        ("Smith et al. (2021) describe the task; recorded 2019.", "2021"),
        ("First release (2022), see also (2023).", "2022"),
        ("No year here.", ""),
        ("Participants 20 to 40 years old, 19 sessions.", ""),
        ("", ""),
        ("Recorded 2018-2019.", "2018"),
        ("Sessions from 2019-05-03 on.", "2019"),
        # Years inside identifiers are not years of the dataset.
        ("Paper: https://doi.org/10.1016/j.neuroimage.2021.118000; 2019.", "2019"),
        ("See https://example.org/2017/05/data, recorded 2019.", "2019"),
        ("Folder sub-2019 holds the pilot.", ""),
        ("Built with toolbox 1.2019.3.", ""),
        ("(1234) trials, recorded 2019.", "2019"),
    ],
)
def test_year_from_readme(data_loaders, readme, expected):
    assert data_loaders._year_from_readme(readme) == expected


def test_year_from_readme_never_returns_two_digits(data_loaders):
    for readme in ("in 1999", "in 2000", "since 2019.", "2026-11-13 release"):
        year = data_loaders._year_from_readme(readme)
        assert len(year) == 4 and year.isdigit(), (readme, year)


def test_context_year_from_bare_readme_year(data_loaders, monkeypatch):
    monkeypatch.setattr(
        data_loaders,
        "_load_dataset_details",
        lambda _dataset_id: {"readme": "fNIRS finger tapping, recorded in 2019."},
    )
    context = data_loaders._build_dataset_context("NM000175", None)
    assert context["year"] == "2019"


def test_context_year_prefers_structured_value(data_loaders, monkeypatch):
    monkeypatch.setattr(
        data_loaders,
        "_load_dataset_details",
        lambda _dataset_id: {"year": "2017", "readme": "Recorded (2019)."},
    )
    context = data_loaders._build_dataset_context("NM000175", None)
    assert context["year"] == "2017"


def test_context_year_empty_without_any_year(data_loaders, monkeypatch):
    monkeypatch.setattr(
        data_loaders,
        "_load_dataset_details",
        lambda _dataset_id: {"readme": "A dataset without dates."},
    )
    context = data_loaders._build_dataset_context("NM000175", None)
    assert context["year"] == ""

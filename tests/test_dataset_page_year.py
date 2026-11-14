r"""The dataset-page year never comes out as a two-digit century fragment.

When a README has no citation-style "(YYYY)", the page fell back to
``re.findall(r"\b(19|20)\d{2}\b", readme)``. ``findall`` returns the capture
group, so a README mentioning "2019" gave the year "20" in the citation,
byline, info table and JSON-LD. NEMAR documents carry no
``dataset_created_at``, so every NEMAR page without "(YYYY)" in its README
went through this fallback.

The fallback must not turn a measurement into a year either: a number next to
a unit ("2048 Hz") or inside a number range or dimension ("2000-4000 Hz",
"2018-2019", "1920 x 1080") is never taken, nor is a year after the current
one. No year is better than a wrong one.
"""

from __future__ import annotations

import pytest
from tests_support.dataset_page import load_dataset_page_module


@pytest.fixture
def data_loaders():
    return load_dataset_page_module("data_loaders")


@pytest.fixture
def no_future_bound(data_loaders, monkeypatch):
    """Lift the future-year guard (tested on its own below).

    The cases then test the unit, range and identifier rules alone, e.g.
    "2048 Hz" must give no year whatever the clock says. ``raising=False``
    keeps the cases runnable against code that predates the guard.
    """
    monkeypatch.setattr(data_loaders, "_current_year", lambda: 2099, raising=False)


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
        ("Sessions from 2019-05-03 on.", "2019"),
        ("Recorded 2019-10 at 2000 Hz.", "2019"),
        # Years inside identifiers are not years of the dataset.
        ("Paper: https://doi.org/10.1016/j.neuroimage.2021.118000; 2019.", "2019"),
        ("See https://example.org/2017/05/data, recorded 2019.", "2019"),
        ("Folder sub-2019 holds the pilot.", ""),
        ("Built with toolbox 1.2019.3.", ""),
        ("(1234) trials, recorded 2019.", "2019"),
        # A number next to a unit is a measurement.
        ("EEG sampled at 2048 Hz; recorded 2019.", "2019"),
        ("Sampled at 2048 Hz.", ""),
        ("Sampling rate 2000 Hz.", ""),
        ("Sampled at 2048 kHz.", ""),
        ("A 2048-Hz amplifier, 2016.", "2016"),
        ("Sampled at 2000\u00a0Hz.", ""),
        ("Epochs of 2000 ms.", ""),
        ("Impedances below 2000 \u03a9; recorded 2014.", "2014"),
        ("Impedances below 2000 k\u03a9.", ""),
        ("Impedances below 2000 \u2126.", ""),
        ("Light at 2000 nm.", ""),
        ("Optodes 2000 mm apart.", ""),
        ("2048 samples per epoch, recorded 2013.", "2013"),
        ("1999 trials in total.", ""),
        ("1998 events over all runs; recorded 2012.", "2012"),
        ("Each trial lasted 2000 milliseconds.", ""),
        ("An ISI of 2000 msec; recorded 2015.", "2015"),
        ("Pulses of 2000 microseconds.", ""),
        ("Sampled at 2000 Hertz.", ""),
        ("Sampled at 2000 hertz.", ""),
        ("Light at 1999 nanometers.", ""),
        # A number inside a number range or dimension is not a year.
        ("Band-pass 2000-4000 Hz; recorded 2014.", "2014"),
        ("Band-pass 20\u20132000 Hz; recorded 2014.", "2014"),
        ("Recorded 2018-2019.", ""),
        ("Recorded 2018\u20132019, released 2020.", "2020"),
        ("Collected from 2015 to 2017; released 2018.", "2018"),
        ("Screen 1920 x 1080; recorded 2019.", "2019"),
        ("Screen 1920\u00d71080.", ""),
        ("ISI jittered between 1900 and 2100 ms; recorded 2016.", "2016"),
        ("Collected between 2015 and 2017.", ""),
        ("Between 2015 and 2017 we recorded; released 2018.", "2018"),
        ("Durations of 2000 \u00b1 500 ms; recorded 2016.", "2016"),
        ("Durations of 2000 +/- 500.", ""),
        ("Collected from 2015 until 2017; released 2018.", "2018"),
        ("Recorded 2015 through 2017.", ""),
        # A version or an item number before a dash is no range.
        ("Version 1.0 - 2019 release.", "2019"),
        ("Dataset 3 - 2019 cohort.", "2019"),
        # Words after a year are no units.
        ("In 2019 S. Smith ran the study.", "2019"),
        ("In 2019 subjects came twice.", "2019"),
    ],
)
def test_year_from_readme(data_loaders, no_future_bound, readme, expected):
    assert data_loaders._year_from_readme(readme) == expected


def test_year_after_the_current_year_is_skipped(data_loaders, monkeypatch):
    monkeypatch.setattr(data_loaders, "_current_year", lambda: 2026, raising=False)
    year_from_readme = data_loaders._year_from_readme
    assert year_from_readme("Released 2027.") == ""
    assert year_from_readme("Released 2026.") == "2026"
    assert year_from_readme("Smith (2031) and Doe (2019).") == "2019"
    assert year_from_readme("Sampled at 2048, recorded 2018.") == "2018"


def test_future_year_is_skipped_with_the_real_clock(data_loaders):
    assert data_loaders._year_from_readme("Planned for 2099.") == ""


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


def test_context_year_skips_sampling_rate(data_loaders, no_future_bound, monkeypatch):
    monkeypatch.setattr(
        data_loaders,
        "_load_dataset_details",
        lambda _dataset_id: {"readme": "BioSemi at 2048 Hz, recorded in 2017."},
    )
    context = data_loaders._build_dataset_context("NM000175", None)
    assert context["year"] == "2017"


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

"""Age charts on dataset pages: infant cohorts and declared age units.

Two things broke the age charts of infant (mostly fNIRS) datasets:

* The per-subject chart reads ``participant_tsv['age']`` from the records, in
  the unit of the participants.tsv column. Ingestion now converts the
  dataset-level ages to years and records the column's declared unit in
  ``demographics.participants_age_units``; the chart has to apply it too, or a
  cohort aged 100-140 days is drawn as 100-140 years.
* Ages under two years all fell into one "0" bar with a whole-year caption
  such as "range 0-1 yr". Such cohorts are now drawn in months.

Adult cohorts render exactly as before. The section renders only when the
dataset document has some statistic (here its ``ages``), as on real pages.
Offline: only formatting helpers run.
"""

from __future__ import annotations

import pytest
from tests_support.dataset_page import load_dataset_page_module


@pytest.fixture
def sections():
    return load_dataset_page_module("sections")


def _rows(*pairs):
    return [
        {"subject": f"{i:02d}", "age": age, "sex": sex}
        for i, (age, sex) in enumerate(pairs, start=1)
    ]


def test_infant_cohort_in_years_is_drawn_in_months(sections):
    rst = sections._format_recording_stats_section(
        {
            "demographics": {"ages": [0.25, 0.5, 0.75], "age_mean": 0.5},
            "participants_rows": _rows((0.25, "F"), (0.5, "M"), (0.75, "F")),
        }
    )
    assert "Age distribution by gender" in rst
    assert "(n=3, range 3.0\u20139.0 mo, mean 6.0 mo)" in rst
    assert 'title="3 mo: female n=1"' in rst
    assert 'title="6 mo: male n=1"' in rst
    assert 'title="9 mo: female n=1"' in rst
    assert "0-4" not in rst


def test_declared_days_are_converted_before_plotting(sections):
    # participants.tsv ages in days, declared in participants.json; the
    # dataset document carries the unit and its ages in years.
    rst = sections._format_recording_stats_section(
        {
            "demographics": {
                "ages": [0.2738, 0.3833],
                "participants_age_units": "days",
                "age_mean": 120 / 365.25,
            },
            "participants_rows": _rows((100.0, "M"), (140.0, "F")),
        }
    )
    assert "(n=2, range 3.3\u20134.6 mo, mean 3.9 mo)" in rst
    assert 'title="3 mo: male n=1"' in rst
    assert 'title="4 mo: female n=1"' in rst
    assert " yr" not in rst


def test_declared_months_for_older_children_stay_in_years(sections):
    rst = sections._format_recording_stats_section(
        {
            "demographics": {"ages": [5, 10], "participants_age_units": "months"},
            "participants_rows": _rows((60.0, "F"), (120.0, "M")),
        }
    )
    assert "(n=2, range 5\u201310 yr)" in rst
    assert 'title="5-9: female n=1"' in rst
    assert 'title="10-14: male n=1"' in rst


def test_dataset_level_infant_ages_are_drawn_in_months(sections):
    rst = sections._format_recording_stats_section(
        {"demographics": {"ages": [0.5, 0.75], "age_mean": 0.625}}
    )
    assert (
        "(n=2, range 6.0\u20139.0 mo, mean 7.5 mo \u00b7 sex per subject not reported)"
        in rst
    )
    assert 'title="6 mo: 1"' in rst
    assert 'title="9 mo: 1"' in rst


def test_unknown_unit_reads_as_years(sections):
    rst = sections._format_recording_stats_section(
        {
            "demographics": {"ages": [20, 30], "participants_age_units": "fortnights"},
            "participants_rows": _rows((20.0, "F"), (30.0, "M")),
        }
    )
    assert "(n=2, range 20\u201330 yr)" in rst


@pytest.mark.parametrize(
    ("ages", "expected"),
    [
        ([0.3, 1.99], (12.0, 1, "mo")),
        ([0.3, 2.0], (1.0, 5, "yr")),
        ([25, 40], (1.0, 5, "yr")),
        ([], (1.0, 5, "yr")),
    ],
)
def test_age_axis(sections, ages, expected):
    assert sections._age_axis(ages) == expected


# --- unchanged output for adult cohorts (controls) ---------------------------


def test_adult_per_subject_chart_unchanged(sections):
    rst = sections._format_recording_stats_section(
        {
            "demographics": {"ages": [20, 22, 34, 53], "age_mean": 32.25},
            "participants_rows": _rows(
                (20.0, "F"), (22.0, "M"), (34.0, "F"), (53.0, "M")
            ),
        }
    )
    assert "(n=4, range 20\u201353 yr, mean 32.2 yr)" in rst
    assert 'title="20-24: female n=1"' in rst
    assert 'title="20-24: male n=1"' in rst
    assert 'title="30-34: female n=1"' in rst
    assert 'title="50-54: male n=1"' in rst
    assert 'title="20-24: n=2"' in rst


def test_adult_dataset_level_histogram_unchanged(sections):
    rst = sections._format_recording_stats_section(
        {"demographics": {"ages": [20, 25, 30], "age_mean": 25.0}}
    )
    assert (
        "(n=3, range 20\u201330 yr, mean 25.0 yr \u00b7 sex per subject not reported)"
        in rst
    )
    assert 'title="20-24: 1"' in rst
    assert 'title="25-29: 1"' in rst
    assert 'title="30-34: 1"' in rst


def test_rows_without_usable_ages_are_skipped(sections):
    rst = sections._format_recording_stats_section(
        {
            "demographics": {"ages": [30]},
            "participants_rows": [
                {"subject": "01", "age": "n/a", "sex": "F"},
                {"subject": "02", "age": None, "sex": "M"},
                {"subject": "03", "age": 0, "sex": "M"},
                {"subject": "04", "age": 30.0, "sex": "M"},
            ],
        }
    )
    assert "(n=1, range 30\u201330 yr)" in rst

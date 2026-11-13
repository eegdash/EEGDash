"""participants.tsv ages: fractions of a year are kept, declared units converted.

Ages were read as ``int(float(value))`` and kept when 0 < age < 120, so every
age under one year became 0 and was dropped: infant fNIRS cohorts lost all
their ages. Ages recorded in months or days were read as years. BIDS defines
``age`` in years; another unit is applied only when the age column's entry in
participants.json declares it.
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


def _write_participants(
    root: Path,
    ages: list[str],
    *,
    column: str = "age",
    units: str | None = None,
    sidecar_key: str | None = None,
) -> None:
    rows = "".join(f"sub-{i:02d}\t{age}\n" for i, age in enumerate(ages, start=1))
    (root / "participants.tsv").write_text(f"participant_id\t{column}\n{rows}")
    if units is not None:
        entry = {"Description": "age of the participant", "Units": units}
        (root / "participants.json").write_text(
            json.dumps({sidecar_key or column: entry})
        )


def _ages(root: Path) -> list[float]:
    return dm._read_participants_demographics(root)[1]


# --- behaviour fixed here ---------------------------------------------------


def test_ages_under_one_year_are_kept(tmp_path: Path):
    _write_participants(tmp_path, ["0.5", "0.75", "1.25", "30"])
    assert _ages(tmp_path) == [0.5, 0.75, 1.25, 30]


def test_months_declared_in_participants_json(tmp_path: Path):
    _write_participants(tmp_path, ["6", "18", "240"], units="months")
    assert _ages(tmp_path) == [0.5, 1.5, 20]


def test_days_declared_in_participants_json(tmp_path: Path):
    # An infant cohort recorded in days, declared as "Units": "day".
    _write_participants(tmp_path, ["100", "140"], units="day")
    assert _ages(tmp_path) == [
        pytest.approx(100 / 365.25, abs=1e-4),
        pytest.approx(140 / 365.25, abs=1e-4),
    ]


def test_weeks_declared_in_participants_json(tmp_path: Path):
    _write_participants(tmp_path, ["26"], units="weeks")
    assert _ages(tmp_path) == [pytest.approx(26 * 7 / 365.25, abs=1e-4)]


def test_years_declared_keeps_fractions(tmp_path: Path):
    _write_participants(tmp_path, ["6", "0.5"], units="years")
    assert _ages(tmp_path) == [6, 0.5]


def test_infinite_age_is_dropped_instead_of_raising(tmp_path: Path):
    # int(float("inf")) raised OverflowError out of
    # _read_participants_demographics, which caught ValueError/TypeError only.
    _write_participants(tmp_path, ["inf", "30"])
    assert _ages(tmp_path) == [30]


def test_sidecar_key_matched_case_insensitively(tmp_path: Path):
    _write_participants(
        tmp_path, ["6"], column="Age", units="months", sidecar_key="age"
    )
    assert _ages(tmp_path) == [0.5]


@pytest.mark.parametrize(
    ("units", "expected"),
    [
        ("months", "months"),
        ("day", "days"),
        ("weeks", "weeks"),
        ("years", None),
        ("decimal score", None),
    ],
)
def test_read_participants_age_units(tmp_path: Path, units, expected):
    _write_participants(tmp_path, ["6"], units=units)
    assert dm._read_participants_age_units(tmp_path) == expected


def test_read_participants_age_units_none_without_sidecar(tmp_path: Path):
    _write_participants(tmp_path, ["6"])
    assert dm._read_participants_age_units(tmp_path) is None


def test_read_participants_age_units_ignores_other_columns(tmp_path: Path):
    _write_participants(tmp_path, ["6"])
    (tmp_path / "participants.json").write_text(
        json.dumps({"gestational_age": {"Units": "weeks"}})
    )
    assert dm._read_participants_age_units(tmp_path) is None


def test_read_participants_age_units_tolerates_bad_json(tmp_path: Path):
    _write_participants(tmp_path, ["6"])
    (tmp_path / "participants.json").write_text("{not json")
    assert dm._read_participants_age_units(tmp_path) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("years", "years"),
        ("Year(s)", "years"),
        ("y", "years"),
        ("months", "months"),
        ("mo", "months"),
        ("age in days", "days"),
        ("day", "days"),
        ("weeks", "weeks"),
        ("hours", "hours"),
        ("decimal years", "years"),
        ("years (decimal)", "years"),
        # Free text is never parsed for a unit; it reads as years.
        ("years and months", None),
        ("age at the day of testing", None),
        ("weeks of gestation", None),
        ("Y/M", None),
        ("yo", None),
        ("", None),
        ("score", None),
        (12, None),
        (None, None),
    ],
)
def test_canonical_age_unit(raw, expected):
    assert dm._canonical_age_unit(raw) == expected


def test_dataset_doc_records_the_declared_unit(tmp_path: Path):
    _write_participants(tmp_path, ["100", "140"], units="day")
    doc = dm.extract_dataset_metadata(
        _NoFilesBids(tmp_path), "nm000001", "nemar", "2026-11-13T00:00:00Z"
    )
    demographics = doc["demographics"]
    assert demographics["participants_age_units"] == "days"
    assert demographics["age_min"] == pytest.approx(100 / 365.25, abs=1e-4)
    assert demographics["age_max"] == pytest.approx(140 / 365.25, abs=1e-4)
    assert demographics["age_mean"] == pytest.approx(120 / 365.25, abs=1e-4)


def test_dataset_doc_keeps_infant_ages_in_years(tmp_path: Path):
    _write_participants(tmp_path, ["30", "0.5"], units="years")
    doc = dm.extract_dataset_metadata(
        _NoFilesBids(tmp_path), "nm000175", "nemar", "2026-11-13T00:00:00Z"
    )
    demographics = doc["demographics"]
    assert "participants_age_units" not in demographics
    assert demographics["ages"] == [30, 0.5]
    assert demographics["age_min"] == 0.5


# --- unchanged behaviour (controls) -----------------------------------------


def test_whole_year_ages_stay_integers(tmp_path: Path):
    _write_participants(tmp_path, ["30", "25.0", "41"])
    ages = _ages(tmp_path)
    assert ages == [30, 25, 41]
    assert all(type(age) is int for age in ages)


def test_out_of_range_and_non_numeric_ages_still_dropped(tmp_path: Path):
    _write_participants(tmp_path, ["n/a", "0", "-1", "150", "120", "nan", ""])
    assert _ages(tmp_path) == []


def test_unit_is_never_guessed_without_participants_json(tmp_path: Path):
    # Values that look like months stay years unless participants.json says so.
    _write_participants(tmp_path, ["6", "9"])
    assert _ages(tmp_path) == [6, 9]


def test_unrecognised_unit_reads_as_years(tmp_path: Path):
    _write_participants(tmp_path, ["6"], units="decimal score")
    assert _ages(tmp_path) == [6]


def test_units_of_another_column_are_ignored(tmp_path: Path):
    _write_participants(tmp_path, ["6"])
    (tmp_path / "participants.json").write_text(
        json.dumps({"gestational_age": {"Units": "weeks"}})
    )
    assert _ages(tmp_path) == [6]


def test_malformed_participants_json_reads_as_years(tmp_path: Path):
    _write_participants(tmp_path, ["6"])
    (tmp_path / "participants.json").write_text("{not json")
    assert _ages(tmp_path) == [6]


def test_dataset_doc_without_participants_json_has_no_unit_key(tmp_path: Path):
    _write_participants(tmp_path, ["30", "41"])
    doc = dm.extract_dataset_metadata(
        _NoFilesBids(tmp_path), "ds000001", "openneuro", "2026-11-13T00:00:00Z"
    )
    assert "participants_age_units" not in doc["demographics"]
    assert doc["demographics"]["ages"] == [30, 41]

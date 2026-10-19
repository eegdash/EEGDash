"""Fixtures for the EEGDash Corpus renderer tests: a tiny atlas covering every status but "to convert" (as the corpus
stands since v1 was converted; tests that need one add it), four buckets (one spelling ``CZ``, three ``Cz``), the
channel set the v1 store adds, a position bank, the per-dataset recording tables (both with tokens) and the
demographics (one dataset with ages and sex, one with sex only), all built in memory or under ``tmp_path``. No
network.

``docs/`` is not a package, so it is put on ``sys.path`` for ``plot_corpus`` imports (as in ``test_plot_dataset``).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _REPO_ROOT / "docs"
if str(_DOCS) not in sys.path:
    sys.path.insert(0, str(_DOCS))

from plot_corpus.data import (  # noqa: E402
    BLOCKED,
    CONVERTED,
    CONVERTED_V1,
    CSV_COLUMNS,
    DEMOGRAPHIC_COLUMNS,
    EXCLUDED,
    GATED,
    PAPER_ONLY,
    RECORDING_COLUMNS,
    TO_CONVERT,
    TO_INGEST,
)
from plot_corpus.electrodes import aggregate_positions  # noqa: E402

REPO_ROOT = _REPO_ROOT
DOCS = _DOCS


def make_row(i: str, status: str, **kw) -> dict[str, str]:
    row = dict.fromkeys(CSV_COLUMNS, "")
    row.update(
        dataset_id=i,
        name=f"Dataset {i}",
        short_name=i.upper(),
        status=status,
        paradigm="resting state",
        licence="CC0-1.0",
        in_reve_paper="yes",
        provenance="hours:zarr; subjects:zarr",
    )
    row["in_v0"] = "yes" if status == CONVERTED else "no"
    row["in_v1"] = (
        "yes" if status in (CONVERTED, CONVERTED_V1, TO_CONVERT, TO_INGEST) else "no"
    )
    row["in_release_card"] = row["in_v0"]
    row.update(kw)
    return row


@pytest.fixture
def rows() -> list[dict[str, str]]:
    """Eight rows: two converted (v0), one converted into the v1 store, then one of every other status but to
    convert.
    """
    conv, conv_ = CONVERTED, CONVERTED_V1
    return [
        make_row("ds000001", conv, hours="120.5", n_subjects="40", n_sessions="80", mean_channels="64",
                 native_sfreq="500", paradigm="motor imagery", licence="CC-BY-4.0"),
        make_row("ds000002", conv, hours="30.0", n_subjects="10", n_sessions="10", mean_channels="19.7",
                 native_sfreq="200"),
        make_row("nm000003", conv_, hours="12.0", n_subjects="5", n_sessions="9", mean_channels="32",
                 native_sfreq="250", provenance="hours:zarr"),
        make_row("x-ingest", TO_INGEST),
        make_row("x-blocked", BLOCKED, licence="CC-BY-NC-ND-4.0"),
        make_row("x-gated", GATED, hours="26847", provenance="hours:stated"),
        make_row("x-excluded", EXCLUDED),
        make_row("x-paper", PAPER_ONLY),
    ]  # fmt: skip


@pytest.fixture
def to_convert() -> dict[str, str]:
    """A dataset ingested by EEGDash and not converted yet, its numbers from EEGDash's catalogue."""
    return make_row("nm000004", TO_CONVERT, hours="6.0", n_subjects="3", n_sessions="3", mean_channels="16",
                    native_sfreq="256", provenance="hours:eegdash")  # fmt: skip


@pytest.fixture
def bank() -> dict[str, list[float]]:
    return {"Cz": [0.0, 0.0, 0.1], "CZ": [0.0, 0.0, 0.1], "C3": [-0.07, 0.0, 0.06], "C4": [0.07, 0.0, 0.06],
            "Fz": [0.0, 0.07, 0.07], "Oz": [0.0, -0.1, 0.01], "Pz": [0.0, -0.07, 0.07]}  # fmt: skip


@pytest.fixture
def buckets() -> list[dict[str, str]]:
    return [
        {"bucket": "0", "dataset": "ds000001", "n_chans": "4", "n_sessions": "3", "hours": "100.0",
         "channels": "C3|C4|Cz|Fz"},
        {"bucket": "1", "dataset": "ds000001", "n_chans": "3", "n_sessions": "2", "hours": "20.5",
         "channels": "C3|C4|Cz"},
        {"bucket": "2", "dataset": "ds000002", "n_chans": "3", "n_sessions": "4", "hours": "25.0",
         "channels": "CZ|Fz|Oz"},
        {"bucket": "3", "dataset": "ds000002", "n_chans": "2", "n_sessions": "1", "hours": "5.0",
         "channels": "Cz|Oz"},
    ]  # fmt: skip


@pytest.fixture
def added() -> list[dict[str, str]]:
    """One channel set the v1 store adds (nm000003, converted in v1): Pz is new to the map, Cz shared with v0."""
    return [
        {
            "bucket": "4",
            "dataset": "nm000003",
            "n_chans": "2",
            "n_sessions": "9",
            "hours": "12.0",
            "channels": "Cz|Pz",
        }
    ]


def make_rec(i: str, **kw) -> dict[str, str]:
    row = dict.fromkeys(RECORDING_COLUMNS, "")
    row.update(dataset_id=i, **kw)
    return row


@pytest.fixture
def recordings() -> list[dict[str, str]]:
    """The v0 recording table: ds000001 long recordings, ds000002 short ones; tokens for both."""
    return [
        make_rec("ds000001", n_recordings="80", n_subjects="40", minutes_p10="30", minutes_p25="45",
                 minutes_p50="90", minutes_p75="120", minutes_p90="150", minutes_max="600",
                 rec_per_subject_p50="2", rec_per_subject_max="4", hours_per_subject_p50="3.0",
                 hours_per_subject_max="10", tokens="6000000000"),
        make_rec("ds000002", n_recordings="10", n_subjects="10", minutes_p10="5", minutes_p25="8",
                 minutes_p50="12", minutes_p75="20", minutes_p90="40", minutes_max="60",
                 rec_per_subject_p50="1", rec_per_subject_max="1", hours_per_subject_p50="0.5",
                 hours_per_subject_max="1", tokens="2000000000"),
    ]  # fmt: skip


@pytest.fixture
def recordings_v1() -> list[dict[str, str]]:
    """The v1 recording table: the added dataset only, measured in the v1 store (tokens included)."""
    return [
        make_rec("nm000003", n_recordings="9", n_subjects="5", minutes_p10="60", minutes_p25="70",
                 minutes_p50="80", minutes_p75="90", minutes_p90="100", minutes_max="110",
                 rec_per_subject_p50="2", rec_per_subject_max="3", hours_per_subject_p50="2.4",
                 hours_per_subject_max="4", tokens="1000000000"),
    ]  # fmt: skip


@pytest.fixture
def demographics() -> list[dict[str, str]]:
    """ds000001 with ages and sex, ds000002 with sex only, nm000003 absent (nothing known)."""
    full = dict.fromkeys(DEMOGRAPHIC_COLUMNS, "")
    full.update(dataset_id="ds000001", year="2020", n_ages="40", age_min="18", age_p25="22", age_p50="25",
                age_p75="31", age_max="60", n_male="25", n_female="15", tasks="rest|motor")  # fmt: skip
    sex_only = dict.fromkeys(DEMOGRAPHIC_COLUMNS, "")
    sex_only.update(dataset_id="ds000002", n_male="4", n_female="6")
    return [full, sex_only]


@pytest.fixture
def electrodes(buckets, bank, added) -> list[dict]:
    return aggregate_positions(buckets, bank, added)


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def data_dir(
    tmp_path, rows, buckets, electrodes, added, recordings, recordings_v1, demographics
) -> Path:
    """A data folder as sync_data writes it (atlas, buckets, recordings, demographics, electrodes, manifest)."""
    d = tmp_path / "data"
    d.mkdir()
    write_csv(d / "atlas.csv", rows, CSV_COLUMNS)
    write_csv(d / "eegdash-corpus-v0-recordings.csv", recordings, RECORDING_COLUMNS)
    write_csv(d / "eegdash-corpus-v1-recordings.csv", recordings_v1, RECORDING_COLUMNS)
    write_csv(d / "eegdash-corpus-demographics.csv", demographics, DEMOGRAPHIC_COLUMNS)
    write_csv(d / "eegdash-corpus-v0-channels.csv", buckets, list(buckets[0]))
    write_csv(d / "eegdash-corpus-v1-channels.csv", added, list(added[0]))
    (d / "electrodes.json").write_text(
        json.dumps({"about": "test", "electrodes": electrodes})
    )
    (d / "MANIFEST.json").write_text(
        json.dumps({"synced": "2026-01-02", "source_commit": "abc", "files": {}})
    )
    return d

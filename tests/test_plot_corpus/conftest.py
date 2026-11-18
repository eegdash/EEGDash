"""Fixtures for the EEGDash Corpus renderer tests: a tiny atlas with every status (two converted datasets, two blocked,
one by a No Derivatives licence and one with none, one gated, one excluded), four buckets (one spelling ``CZ``, three
``Cz``), a position bank, the per-dataset recording table (with tokens) and the demographics (one dataset with ages and
sex, one with sex only) and the corpus table (the two converted datasets), all built in memory or under ``tmp_path``.
No network.

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
    CSV_COLUMNS,
    DEMOGRAPHIC_COLUMNS,
    EXCLUDED,
    GATED,
    RECORDING_COLUMNS,
    TABLE,
    TABLE_COLUMNS,
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
        paradigm="resting & consciousness",
        licence="CC0-1.0",
        in_reve_paper="yes",
        provenance="hours:zarr; subjects:zarr",
    )
    row["in_v0"] = "yes" if status == CONVERTED else "no"
    row["in_release_card"] = row["in_v0"]
    row.update(kw)
    return row


@pytest.fixture
def rows() -> list[dict[str, str]]:
    """Six rows: two converted (v0), then two blocked (No Derivatives, no licence), one gated, one excluded."""
    conv = CONVERTED
    return [
        make_row("ds000001", conv, hours="120.5", n_subjects="40", n_sessions="80", mean_channels="64",
                 native_sfreq="500", paradigm="BCI (MI & P300)", licence="CC-BY-4.0"),
        make_row("ds000002", conv, hours="30.0", n_subjects="10", n_sessions="10", mean_channels="19.7",
                 native_sfreq="200"),
        make_row("x-blocked", BLOCKED, licence="CC-BY-NC-ND-4.0", reason="No Derivatives licence (CC-BY-NC-ND-4.0)"),
        make_row("x-nolicence", BLOCKED, licence="", reason="no licence stated"),
        make_row("x-gated", GATED, hours="26847", provenance="hours:stated",
                 reason="gated (registration): redistribution forbidden"),
        make_row("x-excluded", EXCLUDED, short_name="Slow2016",
                 reason="recorded at 100 Hz, below REVE's 200 Hz input"),
    ]  # fmt: skip


# The internal notes of the source table (``note``, never published) from which the sync derives each row's reason.
NOTES = {
    "x-nolicence": "no licence stated by the original host; checked 2026-10-16",
    "x-gated": "gated (registration): not converted; licence check 2026-10-30: no redistribution",
    "x-excluded": "excluded from the corpus: recorded at 100 Hz, below REVE's 200 Hz input; found 2026-10-09",
}


def source_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """The rows as the source repository holds them: an internal ``note`` and ``in_v1``, no ``reason``."""
    out = []
    for r in rows:
        src = {k: v for k, v in r.items() if k != "reason"}
        src.update(
            note=NOTES.get(r["dataset_id"], "found on NEMAR 2026-10-09"),
            in_v1=r["in_v0"],
        )
        out.append(src)
    return out


def table_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """The corpus table of the fixture: one row per converted dataset, as the source holds it (with a dated ``note``)."""
    out = []
    for r in rows:
        if r["status"] == CONVERTED:
            row = {c: r.get(c, "") for c in TABLE_COLUMNS}
            row.update(eegdash_id=r["dataset_id"].replace("ds", "on"), distribution="public", publishable="yes",
                       note="licence checked 2026-10-29")  # fmt: skip
            out.append(row)
    return out


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
def demographics() -> list[dict[str, str]]:
    """ds000001 with ages and sex, ds000002 with sex only."""
    full = dict.fromkeys(DEMOGRAPHIC_COLUMNS, "")
    full.update(dataset_id="ds000001", year="2020", n_ages="40", age_min="18", age_p25="22", age_p50="25",
                age_p75="31", age_max="60", n_male="25", n_female="15", tasks="rest|motor")  # fmt: skip
    sex_only = dict.fromkeys(DEMOGRAPHIC_COLUMNS, "")
    sex_only.update(dataset_id="ds000002", n_male="4", n_female="6")
    return [full, sex_only]


@pytest.fixture
def electrodes(buckets, bank) -> list[dict]:
    return aggregate_positions(buckets, bank)


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def data_dir(tmp_path, rows, buckets, electrodes, recordings, demographics) -> Path:
    """A data folder as sync_data writes it (atlas, buckets, recordings, demographics, corpus table, electrodes,
    manifest).
    """
    d = tmp_path / "data"
    d.mkdir()
    write_csv(d / "atlas.csv", rows, CSV_COLUMNS)
    write_csv(d / "eegdash-corpus-v0-recordings.csv", recordings, RECORDING_COLUMNS)
    write_csv(d / "eegdash-corpus-demographics.csv", demographics, DEMOGRAPHIC_COLUMNS)
    write_csv(d / "eegdash-corpus-v0-channels.csv", buckets, list(buckets[0]))
    table = [{k: v for k, v in r.items() if k != "note"} for r in table_rows(rows)]
    write_csv(d / TABLE, table, list(table[0]))
    (d / "electrodes.json").write_text(
        json.dumps({"about": "test", "electrodes": electrodes})
    )
    (d / "MANIFEST.json").write_text(
        json.dumps({"synced": "2026-01-02", "source_commit": "abc", "files": {}})
    )
    return d

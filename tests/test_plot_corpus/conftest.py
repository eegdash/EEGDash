"""Fixtures for the EEGDash Corpus renderer tests: a tiny atlas covering every status, four buckets (one spelling
``CZ``, three ``Cz``), a position bank, all built in memory or under ``tmp_path``. No network.

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

from plot_corpus.data import ATLAS_COLUMNS, STATUSES  # noqa: E402
from plot_corpus.electrodes import aggregate_positions  # noqa: E402

REPO_ROOT = _REPO_ROOT
DOCS = _DOCS


def make_row(i: str, status: str, **kw) -> dict[str, str]:
    row = dict.fromkeys(ATLAS_COLUMNS, "")
    row.update(dataset_id=i, name=f"Dataset {i}", short_name=i.upper(), status=status, paradigm="resting state",
               licence="CC0-1.0", in_reve_paper="yes", provenance="hours:zarr; subjects:zarr")
    row["in_v0"] = "yes" if status == STATUSES[0] else "no"
    row["in_v1"] = "yes" if status in STATUSES[:3] else "no"
    row["in_release_card"] = row["in_v0"]
    row.update(kw)
    return row


@pytest.fixture
def rows() -> list[dict[str, str]]:
    """Eight rows: two converted (v0), then one of every other status."""
    conv, conv_ = STATUSES[0], STATUSES[1]
    return [
        make_row("ds000001", conv, hours="120.5", n_subjects="40", n_sessions="80", mean_channels="64",
                 native_sfreq="500", paradigm="motor imagery", licence="CC-BY-4.0"),
        make_row("ds000002", conv, hours="30.0", n_subjects="10", n_sessions="10", mean_channels="19.7",
                 native_sfreq="200"),
        make_row("nm000003", conv_, hours="12.0", n_subjects="5", n_sessions="9", mean_channels="32",
                 native_sfreq="250", provenance="hours:eegdash"),
        make_row("x-ingest", STATUSES[2]),
        make_row("x-blocked", STATUSES[3], licence="CC-BY-NC-ND-4.0"),
        make_row("x-gated", STATUSES[4], hours="26847", provenance="hours:stated"),
        make_row("x-excluded", STATUSES[5]),
        make_row("x-paper", STATUSES[6]),
    ]  # fmt: skip


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
    """One channel set of the dataset v1 adds, from EEGDash's records: Pz is new to the map, Cz shared with v0."""
    return [{"bucket": "nm000003-0", "dataset": "nm000003", "n_chans": "2", "n_sessions": "9", "hours": "12.0",
             "channels": "Cz|Pz", "queried_on": "2026-01-01"}]


@pytest.fixture
def electrodes(buckets, bank, added) -> list[dict]:
    return aggregate_positions(buckets, bank, added)


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def data_dir(tmp_path, rows, buckets, electrodes, added) -> Path:
    """A data folder as sync_data writes it (atlas, buckets, electrodes, manifest)."""
    d = tmp_path / "data"
    d.mkdir()
    write_csv(d / "atlas.csv", rows, ATLAS_COLUMNS)
    write_csv(d / "reve-public-nc-channels.csv", buckets, list(buckets[0]))
    write_csv(d / "eegdash-channel-sets.csv", added, list(added[0]))
    (d / "electrodes.json").write_text(json.dumps({"about": "test", "electrodes": electrodes}))
    (d / "MANIFEST.json").write_text(json.dumps({"synced": "2026-01-02", "source_commit": "abc", "files": {}}))
    return d

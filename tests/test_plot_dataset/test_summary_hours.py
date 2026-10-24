"""Catalog hours: NEMAR-measured where complete, EEGDash value otherwise."""

import json

import pandas as pd
import prepare_summary_tables as pst
import pytest


def _row(dataset_id, seconds, unavailable=0):
    catalog = {
        "total_recording_duration": seconds,
        "recordings_unavailable": unavailable,
    }
    return {"dataset_id": dataset_id, "nemar_catalog": catalog}


ROWS = [
    _row("on002718", 7200),  # ds002718's twin, fully measured: 2 h
    _row("nm000276", 3600, unavailable=28),  # partly measured: keep EEGDash
    _row("nm000104", None),
]
CATALOG = pd.DataFrame(
    {
        "dataset": ["ds002718", "nm000276", "nm000104", "EEG2025r1", "x1"],
        "duration_hours_total": [10.0, 6672.0, 300.0, 120.0, None],
        "n_subjects": 1,
        "n_records": 1,
    }
)


def _hours(tmp_path, monkeypatch, find):
    monkeypatch.setattr(pst.NemarMetadataClient, "find_datasets", find)
    monkeypatch.setattr(pst, "STATIC_DATASET_DIR", tmp_path)
    monkeypatch.setattr(pst, "BUILD_STATIC_DIR", tmp_path / "absent")
    pst.save_summary_stats(CATALOG, aggregations={})
    return json.loads((tmp_path / "summary_stats.json").read_text())["duration_hours"]


def test_nemar_hours_only_where_every_recording_is_measured(tmp_path, monkeypatch):
    hours = _hours(tmp_path, monkeypatch, lambda self, limit: ROWS)
    assert hours == 2 + 6672 + 300 + 120


def test_nemar_unavailable_falls_back_to_eegdash(tmp_path, monkeypatch):
    def boom(self, limit):
        raise OSError("offline")

    assert _hours(tmp_path, monkeypatch, boom) == pytest.approx(10 + 6672 + 300 + 120)

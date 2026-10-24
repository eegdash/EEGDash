"""Catalog hours counter: every EEGDash dataset, NEMAR-measured where possible."""

from __future__ import annotations

import json
import math

import nemar_hours as nh
import pandas as pd
import prepare_summary_tables as pst
import pytest


def _page(rows, total, offset=0):
    return {"datasets": rows, "total_count": total, "offset": offset}


ROWS = [
    {"dataset_id": "on002718", "total_recording_duration": 53439},  # 14.84 h
    {"dataset_id": "nm000104", "total_recording_duration": 1240633},  # 344.62 h
    {"dataset_id": "on005083", "total_recording_duration": None},  # not measured
]


def test_parse_catalog_pages_seconds_to_hours():
    hours = nh.parse_catalog_pages([_page(ROWS[:2], 3), _page(ROWS[2:], 3, 2)])
    assert hours["on002718"] == pytest.approx(53439 / 3600)
    assert hours["nm000104"] == pytest.approx(1240633 / 3600)
    assert hours["on005083"] == 0.0


@pytest.mark.parametrize(
    "pages",
    [
        [_page(ROWS[:2], 3)],  # a page is missing
        [_page(ROWS, 3), _page(ROWS[:1], 3)],  # duplicate id
        [_page(ROWS[:2], 3), _page(ROWS[2:], 4)],  # total changed between pages
        [_page([{"dataset_id": "on1", "total_recording_duration": -1}], 1)],
        [_page([{"dataset_id": "on1", "total_recording_duration": "n/a"}], 1)],
        [{"error": "boom"}],
    ],
)
def test_inconsistent_catalog_is_rejected_whole(pages):
    assert nh.parse_catalog_pages(pages) is None


def test_fetch_paginates(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        offset = int(url.split("offset=")[1])
        return _page(ROWS[offset : offset + 2], 3, offset)

    monkeypatch.setattr(nh, "_PAGE", 2)
    monkeypatch.setattr(nh, "_get", fake_get)
    hours = nh.fetch_catalog_hours()
    assert len(calls) == 2
    assert set(hours) == {"on002718", "nm000104", "on005083"}


def test_fetch_failure_and_toggle_fall_back(monkeypatch):
    def boom(url, timeout):
        raise OSError("offline")

    monkeypatch.setattr(nh, "_get", boom)
    assert nh.fetch_catalog_hours() is None
    monkeypatch.setenv(nh.ENV_TOGGLE, "0")
    monkeypatch.setattr(nh, "_get", lambda *a: pytest.fail("fetched"))
    assert nh.fetch_catalog_hours() is None


CATALOG = pd.DataFrame(
    {
        "dataset": ["ds002718", "nm000104", "EEG2025r1", "ds008730", "on005083", "x1"],
        "duration_hours_total": [10.0, 300.0, 120.0, 50.0, 7.0, None],
    }
)
NEMAR = {"on002718": 53439 / 3600, "nm000104": 1240633 / 3600, "on005083": 0.0}


def test_combined_hours_prefers_nemar_and_covers_every_dataset():
    out = nh.combined_hours(CATALOG, NEMAR)
    # ds002718 -> on002718 and nm000104 from NEMAR; EEG2025r1, ds008730 (no
    # mirror) and on005083 (NEMAR has not measured it) from EEGDash; x1 none.
    assert out["hours"] == pytest.approx(53439 / 3600 + 1240633 / 3600 + 120 + 50 + 7)
    assert out["datasets_from_nemar"] == 2
    assert out["datasets_from_eegdash"] == 3
    assert out["datasets_without_hours"] == 1


def test_combined_hours_without_nemar_is_the_eegdash_sum():
    out = nh.combined_hours(CATALOG, None)
    assert out["hours"] == pytest.approx(10 + 300 + 120 + 50 + 7)
    assert out["datasets_from_nemar"] == 0


def test_combined_hours_accepts_seconds_column():
    df = pd.DataFrame({"dataset": ["ds008730"], "total_duration_s": [7200]})
    assert nh.combined_hours(df, {})["hours"] == pytest.approx(2.0)


def _stats(tmp_path, monkeypatch, nemar):
    monkeypatch.setattr(pst, "STATIC_DATASET_DIR", tmp_path)
    monkeypatch.setattr(pst, "BUILD_STATIC_DIR", tmp_path / "absent")
    df = CATALOG.assign(n_subjects=1, n_records=2)
    pst.save_summary_stats(df, aggregations={}, nemar_hours=nemar)
    return json.loads((tmp_path / "summary_stats.json").read_text())


def test_summary_stats_use_combined_hours(tmp_path, monkeypatch):
    stats = _stats(tmp_path, monkeypatch, NEMAR)
    expected = 53439 / 3600 + 1240633 / 3600 + 120 + 50 + 7
    assert stats["duration_hours"] == round(expected)
    assert stats["duration_hours_datasets_from_nemar"] == 2
    assert stats["duration_hours_datasets_from_eegdash"] == 3
    assert stats["duration_hours_datasets_without_hours"] == 1
    assert stats["datasets_total"] == 6  # other counters unchanged


def test_summary_stats_offline_fallback(tmp_path, monkeypatch):
    stats = _stats(tmp_path, monkeypatch, None)
    assert stats["duration_hours"] == 487
    assert stats["duration_hours_datasets_from_nemar"] == 0
    assert not math.isnan(stats["duration_hours"])

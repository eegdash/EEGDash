"""Catalog hours counter from the NEMAR observability dashboard.

The fixture is the ``recordings`` section of the live snapshot
(``https://dashboard.nemar.org/observability/api/snapshot``, generated
2026-10-23T22:47:30Z), trimmed to EEG and MEG. The expected numbers are the
ones the dashboard shows for ``#hours=eeg:16``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import nemar_observability as nob
import pandas as pd
import prepare_summary_tables as pst
import pytest

SNAPSHOT = json.loads(
    (Path(__file__).with_name("nemar_observability_snapshot.json")).read_text()
)


def test_eeg_at_least_16_channels_matches_dashboard():
    out = nob.hours_at_least(SNAPSHOT, "eeg", 16)
    assert out["hours"] == pytest.approx(55081.0453)
    assert out["recordings"] == 162239
    assert out["datasets"] == 574
    assert out["generated_at"] == "2026-10-23T22:47:30.867Z"


def test_min_one_channel_is_the_modality_total():
    eeg = next(
        m
        for m in SNAPSHOT["sections"][0]["channel_hours"]["modalities"]
        if m["modality"] == "EEG"
    )
    out = nob.hours_at_least(SNAPSHOT, "EEG", 1)
    assert out["hours"] == pytest.approx(eeg["hours"], rel=1e-9)
    assert out["recordings"] == eeg["recordings"]
    assert out["datasets"] == eeg["datasets"]


def test_counter_carries_label_and_view_link():
    out = nob.hours_counter(SNAPSHOT)
    assert out["label"] == "EEG hours (≥16 channels)"
    assert out["url"] == "https://dashboard.nemar.org/observability#hours=eeg:16"
    assert out["source"] == "nemar-observability"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(sections=[]),
        lambda s: s["sections"][0].pop("channel_hours"),
        lambda s: s["sections"][0]["channel_hours"]["modalities"].pop(0),  # no EEG
        lambda s: s["sections"][0]["channel_hours"]["modalities"][0]["bins"][5].update(
            hours="n/a"
        ),
        lambda s: s["sections"][0]["channel_hours"]["modalities"][0]["dataset_peaks"][
            0
        ].pop("datasets"),
    ],
)
def test_malformed_snapshot_falls_back(mutate):
    snapshot = copy.deepcopy(SNAPSHOT)
    mutate(snapshot)
    assert nob.hours_at_least(snapshot, "eeg", 16) is None


def test_env_toggle_skips_network(monkeypatch):
    monkeypatch.setenv(nob.ENV_TOGGLE, "0")
    monkeypatch.setattr(
        nob.urllib.request, "urlopen", lambda *a, **k: pytest.fail("fetched")
    )
    assert nob.fetch_snapshot() is None


def _stats(tmp_path, monkeypatch, nemar_hours):
    monkeypatch.setattr(pst, "STATIC_DATASET_DIR", tmp_path)
    monkeypatch.setattr(pst, "BUILD_STATIC_DIR", tmp_path / "absent")
    df = pd.DataFrame(
        {"n_subjects": [3, 4], "n_records": [5, 6], "total_duration_s": [7200, 3600]}
    )
    pst.save_summary_stats(df, aggregations={}, nemar_hours=nemar_hours)
    return json.loads((tmp_path / "summary_stats.json").read_text())


def test_summary_stats_use_nemar_hours(tmp_path, monkeypatch):
    stats = _stats(tmp_path, monkeypatch, nob.hours_counter(SNAPSHOT))
    assert stats["duration_hours"] == 55081
    assert stats["duration_hours_source"] == "nemar-observability"
    assert stats["duration_hours_label"] == "EEG hours (≥16 channels)"
    assert stats["duration_hours_url"].endswith("#hours=eeg:16")
    assert stats["duration_hours_recordings"] == 162239
    assert stats["subjects_total"] == 7  # other counters unchanged


def test_summary_stats_fall_back_to_catalog_sum(tmp_path, monkeypatch):
    stats = _stats(tmp_path, monkeypatch, None)
    assert stats["duration_hours"] == 3
    assert stats["duration_hours_source"] == "eegdash-catalog"
    assert "duration_hours_label" not in stats

"""Offline tests for the iso-hour landscape: cleaning rules and rendering."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from plot_dataset import isohour, isohour_static


def _doc(
    ds,
    name,
    n_subj,
    dur_s,
    *,
    mod=("eeg",),
    sfreq=((500.0, 10),),
    types=(),
    species=None,
):
    return {
        "dataset_id": ds,
        "name": name,
        "demographics": {"subjects_count": n_subj, "species": species},
        "total_duration_s": dur_s,
        "recording_modality": list(mod),
        "sfreq_counts": [{"val": v, "count": c} for v, c in sfreq],
        "tags": {"type": list(types), "pathology": []},
        "sessions": [],
        "source": "nemar",
    }


@pytest.fixture
def docs():
    return [
        _doc("on000001", "Sleep cohort", 100, 100 * 12 * 3600, types=["Sleep"]),
        _doc("on000002", "Visual task", 20, 20 * 3600, types=["Perception"]),
        # HBN BDF copy and mini subset of an original release.
        _doc("on005505", "HBN Release 1", 136, 423140.0),
        _doc("EEG2025r1", "HBN Release 1 (BDF Converted)", 136, 423811.0),
        _doc("EEG2025r1mini", "HBN Release 1 (BDF Converted)", 20, 73290.0),
        # Same name: the bigger version wins.
        _doc("on000010", "Auditory naming", 106, 5000.0, sfreq=((1000.0, 106),)),
        _doc("ds000010", "Auditory naming", 119, 460800.0, sfreq=((1000.0, 119),)),
        # Same recordings under two names.
        _doc("on000020", "Flicker A", 69, 190083.0, sfreq=((512.0, 70),)),
        _doc("on000021", "Flicker B", 69, 190083.4, sfreq=((512.0, 70),)),
        # Same subjects and duration but different sampling profile: kept.
        _doc("on000030", "Other study", 69, 190083.0, sfreq=((256.0, 69),)),
        # Excluded for cause.
        _doc("on006366", "Mouse sleep", 92, 7.8e6),
        _doc("on000040", "Rat study", 8, 3600.0, species="Rattus norvegicus"),
        _doc("on004849", "STRONG", 1, 1926.0),
        _doc("on000050", "No duration", 30, None),
        _doc("on000060", "Epoched EEGLAB", 40, 240.0),
        _doc("on000070", "Split modality", 10, 36000.0, mod=("e", "e", "g")),
        _doc("on000080", "Short but real", 10, 1800.0, sfreq=((250.0, 10),)),
    ]


def test_cleaning_rules(docs):
    t = isohour.build_isohour_table(docs).set_index("dataset_id")
    status = t["status"].to_dict()

    assert status["on000001"] == "kept"
    assert status["on000002"] == "kept"
    assert status["on005505"] == "kept"
    assert status["EEG2025r1"] == "duplicate"
    assert status["EEG2025r1mini"] == "duplicate"
    assert status["ds000010"] == "kept" and status["on000010"] == "duplicate"
    assert t.at["on000010", "duplicate_of"] == "ds000010"
    assert status["on000020"] == "kept" and status["on000021"] == "duplicate"
    assert t.at["on000021", "duplicate_of"] == "on000020"
    assert status["on000030"] == "kept"
    assert status["on006366"] == "non_human"
    assert status["on000040"] == "non_human"
    assert status["on004849"] == "not_a_recording"
    assert status["on000050"] == "no_duration"
    assert status["on000060"] == "duration_unreliable"
    assert t.at["on000070", "modality"] == "eeg"
    assert t.at["on000001", "hours_per_subject"] == pytest.approx(12.0)
    assert t.at["on000001", "hours_total"] == pytest.approx(1200.0)


def test_every_catalogue_row_is_accounted_for(docs):
    t = isohour.build_isohour_table(docs)
    assert len(t) == len(docs)
    allowed = {
        "kept",
        "duplicate",
        "non_human",
        "not_a_recording",
        "no_duration",
        "duration_unreliable",
    }
    assert set(t["status"]) <= allowed
    assert t.loc[t["status"] == "duplicate", "duplicate_of"].notna().all()


def test_record_correction_only_raises_short_durations(docs):
    t = isohour.build_isohour_table(docs)
    sums = {"on000060": 40 * 600.0, "on000002": 1.0, "on000080": 100.0}
    fixed = isohour.correct_durations_from_records(
        t, record_sum=lambda ds: sums.get(ds, 0.0)
    )
    fixed = fixed.set_index("dataset_id")
    assert fixed.at["on000060", "status"] == "kept"
    assert fixed.at["on000060", "duration_source"] == "records"
    assert fixed.at["on000060", "hours_per_subject"] == pytest.approx(600 / 3600)
    # A short dataset whose records sum to less keeps its aggregate.
    assert fixed.at["on000080", "duration_source"] == "dataset"
    assert fixed.at["on000080", "hours_total"] == pytest.approx(0.5)
    # Long datasets are never queried or lowered.
    assert fixed.at["on000002", "duration_source"] == "dataset"
    assert fixed.at["on000002", "hours_total"] == pytest.approx(20.0)


def test_summary_numbers(docs):
    t = isohour.build_isohour_table(docs)
    s = isohour.summarise_isohour(t)
    kept = t[t["status"] == "kept"]
    assert s["n_kept"] == len(kept)
    assert s["hours_total"] == pytest.approx(kept["hours_total"].sum())
    # Only datasets with >=100 subjects AND >=10 h per subject are "dense";
    # ds000010 (119 subjects x 1.1 h) is not.
    assert [d["dataset_id"] for d in s["dense"]] == ["on000001"]


def test_landscape_renders_offline(docs, tmp_path):
    t = isohour.build_isohour_table(docs)
    out = isohour.generate_isohour_landscape(None, tmp_path / "iso.html", table=t)
    text = out.read_text(encoding="utf-8")
    assert 'id="dataset-isohour"' in text
    assert "Sleep cohort" in text
    assert "Mouse sleep" not in text  # excluded rows are not plotted
    assert "10,000 h" in text


def test_fetch_catalogue_pages(monkeypatch):
    pages = {
        0: {"total": 3, "data": [{"dataset_id": "a"}, {"dataset_id": "b"}]},
        2: {"total": 3, "data": [{"dataset_id": "c"}]},
    }

    def fake_get(url, ttl_s=0):
        skip = int(url.split("skip=")[1])
        return pages[skip]

    with patch.object(isohour, "_cached_get_json", side_effect=fake_get):
        docs = isohour.fetch_catalogue("http://x/api", "db", page_size=2)
    assert [d["dataset_id"] for d in docs] == ["a", "b", "c"]


def test_static_figure_writes_files(docs, tmp_path):
    t = isohour.build_isohour_table(docs)
    stats = isohour_static.make_isohour_figure(t, tmp_path / "fig")
    for ext in ("pdf", "png", "svg"):
        assert (tmp_path / f"fig.{ext}").stat().st_size > 0
    assert stats["n_kept"] == int((t["status"] == "kept").sum())
    assert isinstance(pd.DataFrame(stats["dense"]), pd.DataFrame)

"""Per-field metadata coverage aggregation (Phase 4)."""

from __future__ import annotations

import json
from pathlib import Path

from _coverage import (
    INFERABLE_FIELDS,
    aggregate_output_dir,
    aggregate_records,
    format_coverage_summary,
    iter_records_from_output,
)


def _rec(ext, modality, sfreq, nchans, ntimes, dur, prov):
    return {
        "extension": ext,
        "recording_modality": [modality],
        "sampling_frequency": sfreq,
        "nchans": nchans,
        "ntimes": ntimes,
        "ch_names": ["a"] if nchans else None,
        "duration_seconds": dur,
        "_metadata_provenance": prov,
    }


def test_inferable_fields_set():
    assert "ntimes" in INFERABLE_FIELDS
    assert "duration_seconds" in INFERABLE_FIELDS
    assert "sampling_frequency" in INFERABLE_FIELDS


def test_aggregate_counts_resolved_and_missing():
    records = [
        _rec(
            ".vhdr",
            "eeg",
            250.0,
            32,
            10000,
            40.0,
            {"ntimes": "binary_parser", "duration_seconds": "derived"},
        ),
        _rec(
            ".fif",
            "meg",
            1100.0,
            394,
            None,
            None,
            {"ntimes": None, "duration_seconds": None},
        ),
    ]
    report = aggregate_records(records)
    assert report["total_records"] == 2
    assert report["fields"]["ntimes"]["resolved"] == 1
    assert report["fields"]["ntimes"]["missing"] == 1
    assert report["fields"]["ntimes"]["coverage"] == 0.5
    assert report["fields"]["sampling_frequency"]["resolved"] == 2


def test_aggregate_breaks_down_by_source_format_modality():
    records = [
        _rec(
            ".vhdr",
            "eeg",
            250.0,
            32,
            10000,
            40.0,
            {"ntimes": "binary_parser", "duration_seconds": "derived"},
        ),
        _rec(
            ".edf",
            "eeg",
            256.0,
            19,
            5120,
            20.0,
            {"ntimes": "sidecar_arithmetic", "duration_seconds": "sidecar_arithmetic"},
        ),
        _rec(
            ".fif",
            "meg",
            1100.0,
            394,
            None,
            None,
            {"ntimes": None, "duration_seconds": None},
        ),
    ]
    report = aggregate_records(records)
    nt = report["fields"]["ntimes"]
    assert nt["by_source"]["binary_parser"] == 1
    assert nt["by_source"]["sidecar_arithmetic"] == 1
    assert nt["by_source"][""] == 1  # missing -> empty-string bucket
    # by format: .fif is the worst for ntimes
    assert report["by_format"][".fif"]["ntimes"]["missing"] == 1
    assert report["by_format"][".vhdr"]["ntimes"]["resolved"] == 1
    # by modality
    assert report["by_modality"]["meg"]["ntimes"]["missing"] == 1
    assert report["by_modality"]["eeg"]["ntimes"]["resolved"] == 2


def test_summary_is_human_readable():
    records = [
        _rec(
            ".fif",
            "meg",
            1100.0,
            394,
            None,
            None,
            {"ntimes": None, "duration_seconds": None},
        ),
    ]
    text = format_coverage_summary(aggregate_records(records))
    assert "ntimes" in text
    assert "%" in text


def test_iter_and_aggregate_output_dir(tmp_path: Path):
    ds = tmp_path / "ds001"
    ds.mkdir()
    (ds / "ds001_records.json").write_text(
        json.dumps(
            {
                "records": [
                    _rec(
                        ".vhdr",
                        "eeg",
                        250.0,
                        32,
                        10000,
                        40.0,
                        {"ntimes": "binary_parser"},
                    ),
                    _rec(".fif", "meg", 1100.0, 394, None, None, {"ntimes": None}),
                ]
            }
        )
    )
    assert len(list(iter_records_from_output(tmp_path))) == 2
    report = aggregate_output_dir(tmp_path)
    assert report["total_records"] == 2
    assert report["fields"]["ntimes"]["resolved"] == 1

"""Digested records may be committed gzipped (GitHub's 100 MB file limit).

Covers the shared reader, the inject loader and the validator, plus the
institutional-figshare storage pattern that failed daily validation.
"""

from __future__ import annotations

import gzip
import importlib
import json
import sys
from pathlib import Path

import pytest

_INGESTIONS_DIR = str(Path(__file__).resolve().parents[2] / "scripts" / "ingestions")


def _mod(name):
    if _INGESTIONS_DIR not in sys.path:
        sys.path.insert(0, _INGESTIONS_DIR)
    return importlib.import_module(name)


_RECORDS = {"records": [{"dataset": "nm1", "bids_relpath": "sub-01/eeg/a.edf"}]}


def _write(tmp_path, gz):
    d = tmp_path / "nm1"
    d.mkdir()
    if gz:
        with gzip.open(d / "nm1_records.json.gz", "wt", encoding="utf-8") as f:
            json.dump(_RECORDS, f)
    else:
        (d / "nm1_records.json").write_text(json.dumps(_RECORDS))
    return d


@pytest.mark.parametrize("gz", [False, True])
def test_records_path_and_load(tmp_path, gz):
    rio = _mod("_records_io")
    d = _write(tmp_path, gz)
    path = rio.records_path(d, "nm1")
    assert path is not None and path.name.endswith(".gz") == gz
    assert rio.load_json(path) == _RECORDS


def test_plain_file_wins_over_stale_gz(tmp_path):
    rio = _mod("_records_io")
    d = _write(tmp_path, gz=True)
    (d / "nm1_records.json").write_text("{}")
    assert rio.records_path(d, "nm1").name == "nm1_records.json"


def test_missing_records_is_none(tmp_path):
    assert _mod("_records_io").records_path(tmp_path, "nm1") is None


def test_inject_loads_gzipped_records(tmp_path):
    pytest.importorskip("httpx")
    plan = _mod("_inject_plan")
    d = _write(tmp_path, gz=True)
    records = plan.load_records(d)
    assert [r["bids_relpath"] for r in records] == ["sub-01/eeg/a.edf"]
    assert records[0]["dataset"] == "nm1"


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("https://figshare.com/articles/dataset/x/123", True),
        ("https://ndownloader.figshare.com/files/1", True),
        (
            "https://rdr.ucl.ac.uk/articles/dataset/"
            "Random_and_Sequential_Order_Finger_Motor_Imagery/33332307",
            True,
        ),
        ("https://data.dtu.dk/articles/dataset/Dataset_for_publication/30868751", True),
        ("https://example.org/some/file.zip", False),
    ],
)
def test_figshare_storage_pattern(url, ok):
    pytest.importorskip("pydantic")
    validate = _mod("_validate")
    assert validate.validate_storage_url("figshare", url)[0] is ok

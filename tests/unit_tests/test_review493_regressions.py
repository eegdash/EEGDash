"""Regressions for authoritative download identity (no network)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from test_download_integrity import DATA, KEY, OPENNEURO_URI, URI, _client

import eegdash.downloader as dl
from eegdash.dataset import base


@pytest.mark.parametrize("pin", [None, "historical"])
def test_twin_key_is_authoritative(tmp_path, pin):
    dest = tmp_path / "raw.edf"
    with pytest.raises(OSError, match="expected 20 bytes, got 10") as caught:
        dl.download_s3_file(
            OPENNEURO_URI,
            dest,
            filesystem=_client(DATA[:10]),
            annex_key=KEY,
            version_id=pin,
        )
    assert str(dest) in str(caught.value)
    assert ".part" not in str(caught.value)
    if pin:
        assert pin in str(caught.value)
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part"))
    assert not dest.with_name(dest.name + ".s3-version.json").exists()


def test_uri_key_takes_precedence(tmp_path):
    dest = tmp_path / "raw.edf"
    other = "WORM-s10--other.edf"
    # Body matches the caller's key but not the URI's key: never publish it.
    with pytest.raises(OSError, match="expected 20 bytes, got 10"):
        dl.download_s3_file(URI, dest, filesystem=_client(DATA[:10]), annex_key=other)
    assert not dest.exists()


def test_prescan_is_offline(tmp_path, monkeypatch):
    dest = tmp_path / "raw.edf"
    dest.write_bytes(DATA[:10])

    def network(*a, **k):
        pytest.fail("cache prescan must not access S3")

    monkeypatch.setattr(dl, "get_s3_filesystem", network)
    assert not base._cached_complete(dest, KEY, OPENNEURO_URI)


def test_nemar_manifest_disagrees_refuse_before_transfer(tmp_path, monkeypatch):
    dest = tmp_path / "raw.edf"

    class Manifest:
        version = "1.0.1"

        def __contains__(self, path):
            return True

        def file(self, path):
            return SimpleNamespace(size=10)

    monkeypatch.setattr(base, "_fetch_nemar_manifest", lambda _: Manifest())
    calls = []

    def transfer(entry, staging):
        calls.append(entry)
        Path(staging).write_bytes(DATA[:10])
        return SimpleNamespace(name="OK")

    monkeypatch.setattr(base.nemar, "download_one", transfer)
    from eegdash.dataset.exceptions import DataIntegrityError

    for _ in range(2):
        with pytest.raises(DataIntegrityError, match="re-digest"):
            base._download_via_nemar("nm000289", "raw.edf", dest, KEY)
    assert not calls
    assert not dest.exists()


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_bulk_attempts_other_records_and_aggregates(tmp_path, monkeypatch, n_jobs):
    from test_download_integrity import _openneuro_record

    from eegdash.dataset.dataset import EEGDashDataset
    from eegdash.dataset.exceptions import DataIntegrityError

    records = [_openneuro_record(), _openneuro_record()]
    records[1]["bids_relpath"] = records[1]["bids_relpath"].replace("sub-01", "sub-02")
    records[1]["storage"]["raw_key"] = records[1]["bids_relpath"]
    records[1]["storage"]["annex_keys"] = {records[1]["bids_relpath"]: KEY}
    ds = EEGDashDataset(cache_dir=tmp_path, dataset="ds001234", records=records)
    attempted = []

    def acquire(raw):
        attempted.append(raw.record["bids_relpath"])
        if "sub-01" in raw.record["bids_relpath"]:
            raise DataIntegrityError("short stored object", record=raw.record)

    monkeypatch.setattr(base.EEGDashRaw, "_download_required_files", acquire)
    with pytest.raises(
        DataIntegrityError, match="1 of 2 recording downloads failed"
    ) as caught:
        ds.download_all(n_jobs=n_jobs)
    assert len(attempted) == 2
    assert len(caught.value.failures) == 1


def test_keyed_cache_repairs_are_skipped(tmp_path):
    from eegdash.dataset.io import _repair_channels_tsv, _repair_participants_tsv_ids

    channels = tmp_path / "sub-01_channels.tsv"
    channels.write_text("")
    participants = tmp_path / "participants.tsv"
    participants.write_text("participant_id\nsub-1\n")
    (tmp_path / "sub-01").mkdir()
    protected = {channels, participants}
    assert not _repair_channels_tsv(tmp_path, protected=protected)
    assert not _repair_participants_tsv_ids(tmp_path, protected=protected)
    assert channels.exists()
    assert participants.read_text() == "participant_id\nsub-1\n"


def test_cached_stat_race_is_incomplete(tmp_path, monkeypatch):
    dest = tmp_path / "raw.edf"
    calls = []

    def disappeared(path):
        calls.append(path)
        raise FileNotFoundError(path)

    monkeypatch.setattr(Path, "stat", disappeared)
    assert not base._cached_complete(dest, KEY)
    assert calls == [dest]


def test_bulk_prescan_checks_pin_marker_offline(tmp_path, monkeypatch):
    import json

    from test_download_integrity import RAW, _openneuro_record

    raw = base.EEGDashRaw(_openneuro_record(version_id="past"), cache_dir=tmp_path)
    raw.filecache.parent.mkdir(parents=True)
    raw.filecache.write_bytes(DATA)
    monkeypatch.setattr(
        dl, "get_s3_filesystem", lambda **kw: pytest.fail("network scan")
    )
    assert not raw._cached_files_complete()
    marker = raw.filecache.with_name(raw.filecache.name + ".s3-version.json")
    marker.write_text(
        json.dumps({"uri": f"s3://openneuro.org/ds001234/{RAW}", "version_id": "past"})
    )
    assert raw._cached_files_complete()

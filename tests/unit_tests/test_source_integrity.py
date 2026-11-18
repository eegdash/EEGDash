"""Source pins and raw-cache identity through the actual acquisition paths."""

import copy
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from botocore.response import StreamingBody
from test_download_integrity import DATA, DEP, KEY, RAW, _client, _openneuro_record

from eegdash import downloader
from eegdash.dataset import base
from eegdash.dataset._source_inference import correct_storage_inplace
from eegdash.dataset.dataset import EEGDashDataset
from eegdash.dataset.exceptions import DataIntegrityError


def _quiet_metadata(raw, monkeypatch):
    """Isolate acquisition from unrelated root/position discovery, not raw IO."""
    for name in (
        "_fetch_nemar_root_metadata",
        "_fetch_nemar_session_metadata",
        "_download_position_sidecars",
    ):
        monkeypatch.setattr(raw, name, Mock())


@pytest.mark.parametrize("via_dataset", [False, True])
@pytest.mark.parametrize("pin_kind", ["raw", "dependency"])
def test_approved_on_source_pin_survives_runtime_and_get(
    tmp_path, monkeypatch, via_dataset, pin_kind
):
    record = _openneuro_record(dep_keys=[DEP], annex_keys={})
    record["dataset"] = "on001234"
    if pin_kind == "raw":
        record["storage"]["version_id"] = "raw-past"
    else:
        record["storage"]["version_ids"] = {DEP: "sidecar-past"}
    storage = copy.deepcopy(record["storage"])
    if via_dataset:
        ds = EEGDashDataset(cache_dir=tmp_path, records=[record])
        raw = ds.datasets[0]
    else:
        raw = base.EEGDashRaw(record, cache_dir=tmp_path)
    assert raw.record["storage"] == storage
    assert raw._storage_backend == "s3"
    assert raw._raw_uri == f"s3://openneuro.org/ds001234/{RAW}"
    _quiet_metadata(raw, monkeypatch)
    client = downloader.S3Client.__new__(downloader.S3Client)
    client._client = Mock()
    client._client.head_object.return_value = {"ContentLength": len(DATA)}
    client._client.get_object.side_effect = lambda **kwargs: {
        "Body": StreamingBody(BytesIO(DATA), len(DATA)),
        "ContentLength": len(DATA),
    }
    monkeypatch.setattr(downloader, "get_s3_filesystem", lambda **kwargs: client)
    raw._download_required_files()
    gets = [call.kwargs for call in client._client.get_object.call_args_list]
    assert all(call["Bucket"] == "openneuro.org" for call in gets)
    raw_call = next(call for call in gets if call["Key"].endswith(RAW))
    dep_call = next(call for call in gets if call["Key"].endswith(DEP))
    assert raw_call.get("VersionId") == ("raw-past" if pin_kind == "raw" else None)
    assert dep_call.get("VersionId") == (
        "sidecar-past" if pin_kind == "dependency" else None
    )
    assert raw.record["storage"] == storage


@pytest.mark.parametrize("failure", [FileNotFoundError, PermissionError])
def test_approved_on_missing_pin_never_changes_source(tmp_path, monkeypatch, failure):
    record = _openneuro_record(version_id="missing")
    record["dataset"] = "on001234"
    raw = base.EEGDashRaw(record, cache_dir=tmp_path)
    download = Mock(side_effect=failure("missing pinned object"))
    monkeypatch.setattr(downloader, "download_s3_file", download)
    monkeypatch.setattr(downloader, "get_s3_filesystem", Mock())
    _quiet_metadata(raw, monkeypatch)
    with pytest.raises((DataIntegrityError, PermissionError)):
        raw._download_required_files()
    assert download.call_count == 1
    assert download.call_args.args[0] == f"s3://openneuro.org/ds001234/{RAW}"
    assert download.call_args.kwargs["version_id"] == "missing"
    assert raw._storage_backend == "s3"


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"base": "s3://openneuro.org/ds999999", "version_id": "past"},
        {"base": "s3://openneuro.org/ds001234/sub-01", "version_id": "past"},
        {"backend": "https", "version_id": "past"},
        {"version_id": "", "version_ids": {}},
    ],
)
def test_unapproved_on_source_remains_nemar_first(overrides):
    record = _openneuro_record(**overrides)
    record["dataset"] = "on001234"
    correct_storage_inplace(record)
    assert record["storage"]["base"] == "s3://nemar/on001234"
    assert record["storage"]["backend"] == "nemar"


@pytest.mark.parametrize("mode", ["lazy", "bulk"])
@pytest.mark.parametrize("cached", [None, DATA, b"x" * len(DATA)])
def test_nemar_raw_key_governs_fresh_and_same_size_warm_cache(
    tmp_path, monkeypatch, mode, cached
):
    record = _openneuro_record(
        backend="nemar", base="s3://nemar/on001234", dep_keys=[DEP]
    )
    record["dataset"] = "on001234"
    ds = EEGDashDataset(cache_dir=tmp_path, records=[record])
    raw = ds.datasets[0]
    _quiet_metadata(raw, monkeypatch)
    monkeypatch.setattr(ds, "_download_dataset_files", Mock())
    raw.filecache.parent.mkdir(parents=True, exist_ok=True)
    sidecar = raw.bids_root / DEP
    sidecar.write_bytes(b"unchanged sidecar")
    if cached is not None:
        raw.filecache.write_bytes(cached)
    client = _client(DATA)
    monkeypatch.setattr(downloader, "get_s3_filesystem", lambda **kwargs: client)
    assert raw._cached_files_complete() is (cached == DATA)
    if mode == "bulk":
        ds.download_all(n_jobs=1)
    else:
        raw._download_required_files()
    assert raw.filecache.read_bytes() == DATA
    assert sidecar.read_bytes() == b"unchanged sidecar"
    assert client._client.gets == (0 if cached == DATA else 1)
    downloader.verify_file(raw.filecache, KEY, verify_hash=True)
    assert not list(raw.filecache.parent.glob("*.part"))


def test_nemar_wrong_hash_transfer_is_not_published(tmp_path, monkeypatch):
    record = _openneuro_record(backend="nemar", base="s3://nemar/on001234")
    record["dataset"] = "on001234"
    raw = base.EEGDashRaw(record, cache_dir=tmp_path)
    _quiet_metadata(raw, monkeypatch)
    client = _client(b"x" * len(DATA))
    monkeypatch.setattr(downloader, "get_s3_filesystem", lambda **kwargs: client)
    with pytest.raises(DataIntegrityError, match="Checksum mismatch"):
        raw._download_required_files()
    assert not raw.filecache.exists()
    assert not list(raw.filecache.parent.glob("*.part"))


def test_unkeyed_nemar_cache_keeps_existing_default(tmp_path, monkeypatch):
    record = _openneuro_record(
        backend="nemar", base="s3://nemar/on001234", annex_keys={}
    )
    record["dataset"] = "on001234"
    raw = base.EEGDashRaw(record, cache_dir=tmp_path)
    _quiet_metadata(raw, monkeypatch)
    raw.filecache.parent.mkdir(parents=True, exist_ok=True)
    raw.filecache.write_bytes(DATA)
    monkeypatch.setattr(
        downloader, "get_s3_filesystem", Mock(side_effect=AssertionError("network"))
    )
    raw._download_required_files()
    assert raw.filecache.read_bytes() == DATA


@pytest.mark.parametrize("payload", [DATA, b"x" * len(DATA)])
def test_nemar_manifest_fallback_obeys_record_hash(tmp_path, monkeypatch, payload):
    record = _openneuro_record(backend="nemar", base="s3://nemar/on001234")
    record["dataset"] = "on001234"
    raw = base.EEGDashRaw(record, cache_dir=tmp_path)
    _quiet_metadata(raw, monkeypatch)
    monkeypatch.setattr(downloader, "get_s3_filesystem", Mock())
    monkeypatch.setattr(
        downloader, "download_s3_file", Mock(side_effect=FileNotFoundError("object"))
    )
    manifest = MagicMock()
    manifest.__contains__.return_value = True
    manifest.file.return_value = SimpleNamespace(size=len(DATA))
    monkeypatch.setattr(base, "_fetch_nemar_manifest", lambda _: manifest)

    def transfer(entry, dest):
        dest.write_bytes(payload)
        return SimpleNamespace(name="OK")

    monkeypatch.setattr(base.nemar, "download_one", transfer)
    if payload == DATA:
        raw._download_required_files()
        assert raw.filecache.read_bytes() == DATA
    else:
        with pytest.raises(DataIntegrityError, match="Checksum mismatch"):
            raw._download_required_files()
        assert not raw.filecache.exists()
    assert not list(raw.filecache.parent.glob("*.part"))
    assert not list(raw.filecache.parent.glob(raw.filecache.name + ".*"))

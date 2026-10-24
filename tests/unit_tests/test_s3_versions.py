"""Pinned raw objects must never silently read HEAD or a same-size cache."""

from io import BytesIO
from unittest.mock import Mock

import pytest
from botocore.response import StreamingBody

from eegdash import downloader
from eegdash.dataset.base import EEGDashRaw
from eegdash.schemas import create_record


def test_pinned_download_selects_version_and_replaces_same_size_cache(tmp_path):
    client = downloader.S3Client.__new__(downloader.S3Client)
    client._client = Mock()
    client._client.head_object.return_value = {"ContentLength": 4}
    client._client.get_object.return_value = {
        "ContentLength": 4,
        "Body": StreamingBody(BytesIO(b"past"), 4),
    }
    dest = tmp_path / "raw.edf"
    dest.write_bytes(b"head")
    downloader.download_s3_file(
        "s3://bucket/raw.edf", dest, filesystem=client, version_id="old+/=version"
    )
    assert dest.read_bytes() == b"past"
    client._client.head_object.assert_called_once_with(
        Bucket="bucket", Key="raw.edf", VersionId="old+/=version"
    )
    client._client.get_object.assert_called_once_with(
        Bucket="bucket", Key="raw.edf", VersionId="old+/=version"
    )
    # Verified cache reuse, but changing the version forces another download.
    downloader.download_s3_file(
        "s3://bucket/raw.edf", dest, filesystem=client, version_id="old+/=version"
    )
    assert client._client.get_object.call_count == 1
    client._client.get_object.return_value["Body"] = StreamingBody(BytesIO(b"next"), 4)
    downloader.download_s3_file(
        "s3://bucket/raw.edf", dest, filesystem=client, version_id="another"
    )
    assert dest.read_bytes() == b"next"
    client._client.get_object.return_value["Body"] = StreamingBody(BytesIO(b"head"), 4)
    downloader.download_s3_file("s3://bucket/raw.edf", dest, filesystem=client)
    assert dest.read_bytes() == b"head"
    client._client.get_object.assert_called_with(Bucket="bucket", Key="raw.edf")


def test_pinned_failed_head_never_trusts_existing_cache(tmp_path):
    client = downloader.S3Client.__new__(downloader.S3Client)
    client._client = Mock()
    client._client.head_object.side_effect = OSError("offline")
    client._client.get_object.side_effect = OSError("offline")
    dest = tmp_path / "raw.edf"
    dest.write_bytes(b"head")
    with pytest.raises(OSError, match="offline"):
        downloader.download_s3_file(
            "s3://bucket/raw.edf", dest, filesystem=client, version_id="past"
        )
    assert dest.read_bytes() == b"head"


def test_record_pin_only_applies_to_raw_and_never_reroutes(tmp_path, monkeypatch):
    rel = "sub-01/eeg/sub-01_task-rest_eeg.edf"
    record = create_record(
        dataset="on003574",
        bids_relpath=rel,
        storage_base="s3://openneuro.org/ds003574",
        dep_keys=["dataset_description.json"],
    )
    record["storage"]["version_id"] = "past"
    raw = EEGDashRaw(record=record, cache_dir=tmp_path)
    download = Mock()
    monkeypatch.setattr(downloader, "download_s3_file", download)
    monkeypatch.setattr(downloader, "get_s3_filesystem", Mock())
    monkeypatch.setattr(downloader, "download_files", Mock())
    raw._download_required_files()
    assert download.call_args_list[0].kwargs["version_id"] == "past"
    assert all("versionId" not in uri for uri in raw._dep_uris)
    assert raw._reroute_twin_to_nemar() is False


def test_dependency_pin_does_not_pin_header_or_other_sidecars(tmp_path, monkeypatch):
    rel = "sub-01/eeg/sub-01_task-rest_eeg.vhdr"
    eeg = rel.replace(".vhdr", ".eeg")
    record = create_record(
        dataset="ds003768",
        bids_relpath=rel,
        storage_base="s3://openneuro.org/ds003768",
        dep_keys=[eeg, rel.replace(".vhdr", ".vmrk")],
    )
    record["storage"]["version_ids"] = {eeg: "binary-version"}
    raw = EEGDashRaw(record=record, cache_dir=tmp_path)
    download = Mock()
    monkeypatch.setattr(downloader, "download_s3_file", download)
    monkeypatch.setattr(downloader, "get_s3_filesystem", Mock())
    others = Mock()
    monkeypatch.setattr(downloader, "download_files", others)
    monkeypatch.setattr(raw, "_download_companion_files", Mock())
    monkeypatch.setattr(raw, "_download_position_sidecars", Mock())
    raw._download_required_files()
    calls = download.call_args_list
    assert len(calls) == 2
    assert not calls[0].kwargs.get("version_id")
    assert calls[1].args[0].endswith(eeg)
    assert calls[1].kwargs["version_id"] == "binary-version"
    assert all(not uri.endswith(eeg) for uri, _ in others.call_args.args[0])


@pytest.mark.parametrize("failure", [FileNotFoundError, PermissionError])
def test_missing_pin_never_reads_head(tmp_path, monkeypatch, failure):
    rel = "sub-01/eeg/sub-01_task-rest_eeg.edf"
    record = create_record(
        dataset="on003574", bids_relpath=rel, storage_base="s3://openneuro.org/ds003574"
    )
    record["storage"]["version_id"] = "missing"
    raw = EEGDashRaw(record=record, cache_dir=tmp_path)
    download = Mock(side_effect=failure("missing pin"))
    monkeypatch.setattr(downloader, "download_s3_file", download)
    monkeypatch.setattr(downloader, "get_s3_filesystem", Mock())
    with pytest.raises(Exception):
        raw._download_required_files()
    assert download.call_count == 1
    assert raw._storage_backend == "s3"


@pytest.mark.parametrize("twin_has_sidecar", [True, False])
def test_sidecar_missing_on_twin_takes_record_from_nemar(
    tmp_path, monkeypatch, twin_has_sidecar
):
    """ds003810's OpenNeuro copy lacks events.tsv; NEMAR has it (#503 docs CI)."""
    rel = "sub-02/eeg/sub-02_task-MIvsRest_run-1_eeg.edf"
    events = rel.replace("_eeg.edf", "_events.tsv")
    record = create_record(
        dataset="on003810",
        bids_relpath=rel,
        storage_base="s3://openneuro.org/ds003810",
        dep_keys=[events],
    )
    raw = EEGDashRaw(record=record, cache_dir=tmp_path)

    def fetch_deps(pairs, **kwargs):
        if twin_has_sidecar:
            for _, dest in pairs:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text("onset\tduration\n")

    monkeypatch.setattr(downloader, "download_s3_file", Mock())
    monkeypatch.setattr(downloader, "get_s3_filesystem", Mock())
    monkeypatch.setattr(downloader, "download_files", fetch_deps)
    monkeypatch.setattr(raw, "_download_companion_files", Mock())
    monkeypatch.setattr(raw, "_download_position_sidecars", Mock())
    retry = Mock()
    monkeypatch.setattr(raw, "_download_required_files", retry)
    raw._download_required_files_impl()
    assert retry.called is not twin_has_sidecar
    assert raw._storage_backend == ("s3" if twin_has_sidecar else "nemar")

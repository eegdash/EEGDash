"""Size and hash checks of downloads against git-annex keys (no network).

A truncated NEMAR object (nm000289, 2026-10-16) used to be accepted: the
stored object itself can be short, even when HEAD and GET agree.
"""

import hashlib
import io
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

import eegdash.downloader as downloader
from eegdash.dataset import base

DATA = b"0123456789abcdefghij"
SHA256 = hashlib.sha256(DATA).hexdigest()
KEY = f"SHA256E-s{len(DATA)}--{SHA256}.edf"
URI = f"s3://nemar/nm000289/objects/{KEY}"


class _FakeS3:
    """boto3 client stand-in: HEAD denied (as on NEMAR), GET serves *data*."""

    def __init__(self, data, errors=()):
        self.data, self.errors, self.gets = data, list(errors), 0

    def head_object(self, **_):
        raise ClientError(
            {"Error": {"Code": "403"}, "ResponseMetadata": {"HTTPStatusCode": 403}},
            "HeadObject",
        )

    def get_object(self, **_):
        self.gets += 1
        if self.errors:
            raise self.errors.pop(0)
        data = self.data

        class _Body:
            def iter_chunks(self, chunk_size):
                yield data

        return {"Body": _Body(), "ContentLength": len(data)}


def _client(data, errors=()):
    client = downloader.S3Client.__new__(downloader.S3Client)
    client._client = _FakeS3(data, errors)
    return client


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (KEY, (20, "sha256", SHA256)),
        (f"SHA512E-s5--{'a' * 128}.set", (5, "sha512", "a" * 128)),
        (f"MD5E-s11657--{'7a' * 16}.vhdr", (11657, "md5", "7a" * 16)),
        (f"SHA256-s7--{'b' * 64}", (7, "sha256", "b" * 64)),
        (f"SHA256E-s7--{'b' * 64}", (7, "sha256", "b" * 64)),  # no extension
        (f"SHA256E-s7-m1600000000--{'b' * 64}.tar.gz", (7, "sha256", "b" * 64)),
        ("WORM-s42-m1600000000--sub-01_eeg.edf", (42, None, None)),
        ("URL--https&c%%example.org%file.edf", (None, None, None)),
        ("URL-s9--https&c%%example.org%file.edf", (9, None, None)),
        (f"SHA256E-s7-S4-C1--{'b' * 64}.edf", (None, None, None)),  # a chunk
        (f"SHA256E-s7--{'b' * 10}.edf", (7, None, None)),  # short digest
        ("sub-01_task-rest_eeg.edf", (None, None, None)),
        ("SHA256E-sX--abc.edf", (None, None, None)),
        ("", (None, None, None)),
    ],
)
def test_annex_key_info(key, expected):
    assert downloader.annex_key_info(key) == expected


def test_short_object_is_refused_and_removed(tmp_path):
    dest = tmp_path / "sub-01_eeg.edf"
    with pytest.raises(OSError, match="expected 20 bytes, got 10"):
        downloader.download_s3_file(URI, dest, filesystem=_client(DATA[:10]))
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part"))


def test_existing_file_of_key_size_is_skipped(tmp_path):
    dest = tmp_path / "sub-01_eeg.edf"
    dest.write_bytes(DATA)
    client = _client(b"never fetched")
    downloader.download_s3_file(URI, dest, filesystem=client)
    assert client._client.gets == 0


def test_existing_file_of_wrong_size_is_refetched(tmp_path):
    dest = tmp_path / "sub-01_eeg.edf"
    dest.write_bytes(DATA[:10])
    client = _client(DATA)
    downloader.download_s3_file(URI, dest, filesystem=client)
    assert client._client.gets == 1
    assert dest.read_bytes() == DATA


def test_download_files_checks_key_size(tmp_path):
    dest = tmp_path / "sub-01_eeg.edf"
    with pytest.raises(OSError, match="expected 20 bytes, got 10"):
        downloader.download_files([(URI, dest)], filesystem=_client(DATA[:10]))
    assert not dest.exists()


def test_hash_verification(tmp_path):
    dest = tmp_path / "sub-01_eeg.edf"
    downloader.download_s3_file(URI, dest, filesystem=_client(DATA), verify_hash=True)
    assert downloader.verify_file(dest, KEY, verify_hash=True) == dest

    corrupt = DATA[:-1] + b"!"
    dest.unlink()
    with pytest.raises(OSError, match="Checksum mismatch"):
        downloader.download_s3_file(
            URI, dest, filesystem=_client(corrupt), verify_hash=True
        )
    assert not dest.exists()


def test_cached_file_with_bad_hash_is_refetched(tmp_path):
    dest = tmp_path / "sub-01_eeg.edf"
    dest.write_bytes(DATA[:-1] + b"!")  # right size, wrong bytes
    client = _client(DATA)
    downloader.download_s3_file(URI, dest, filesystem=client, verify_hash=True)
    assert client._client.gets == 1
    assert dest.read_bytes() == DATA


def test_verify_file_size_only(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(DATA)
    downloader.verify_file(path, len(DATA))
    with pytest.raises(OSError, match="expected 21 bytes, got 20"):
        downloader.verify_file(path, len(DATA) + 1)


def test_get_retries_server_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)
    busy = ClientError(
        {"Error": {"Code": "429"}, "ResponseMetadata": {"HTTPStatusCode": 429}},
        "GetObject",
    )
    client = _client(DATA, errors=[busy])
    dest = tmp_path / "sub-01_eeg.edf"
    downloader.download_s3_file(URI, dest, filesystem=client)
    assert client._client.gets == 2
    assert dest.read_bytes() == DATA


def test_cached_nemar_file_needs_key_size(tmp_path):
    path = tmp_path / "sub-01_eeg.edf"
    assert not base._cached_complete(path, KEY)
    path.write_bytes(DATA[:10])
    assert not base._cached_complete(path, KEY)
    assert base._cached_complete(path, None)  # no key: existence, as before
    path.write_bytes(DATA)
    assert base._cached_complete(path, KEY)


class _FailingHandle(io.BytesIO):
    """A file handle that writes half the payload, then fails (disk full)."""

    def __init__(self, path):
        super().__init__()
        self.path = Path(path)

    def write(self, data):
        self.path.write_bytes(bytes(data[: len(data) // 2]))
        raise OSError("No space left on device")


def test_github_raw_fallback_leaves_no_partial(tmp_path, monkeypatch):
    dest = tmp_path / "sub-01" / "eeg" / "sub-01_channels.tsv"
    monkeypatch.setattr(base, "_download_via_nemar", lambda *a, **k: False)
    monkeypatch.setattr(
        base, "_resolve_nemar_pointer", lambda *a: (None, b"name\ttype\nCz\tEEG\n")
    )
    monkeypatch.setattr(
        base, "open", lambda path, mode: _FailingHandle(path), raising=False
    )
    with pytest.raises(OSError, match="No space left"):
        base._resolve_one_nemar_entry(
            dataset_id="nm000289",
            relpath="sub-01/eeg/sub-01_channels.tsv",
            base="s3://nemar/nm000289",
            dest=dest,
            stored_key=None,
            stored_sidecar=None,
            is_required=False,
        )
    assert not dest.exists()
    assert not list(dest.parent.glob("*.part"))


RAW = "sub-01/eeg/sub-01_task-rest_eeg.edf"
DEP = "sub-01/eeg/sub-01_task-rest_channels.tsv"


def _openneuro_record(**storage):
    """A ds* record served from OpenNeuro's BIDS-named objects, its key in the record."""
    return {
        "dataset": "ds001234",
        "bids_relpath": RAW,
        "storage": {
            "backend": "s3",
            "base": "s3://openneuro.org/ds001234",
            "raw_key": RAW,
            "dep_keys": [],
            "annex_keys": {RAW: KEY},
            **storage,
        },
    }


@pytest.mark.parametrize(
    "damage", [DATA[:10], b"/annex/objects/" + KEY.encode()], ids=["short", "pointer"]
)
def test_download_all_refetches_a_damaged_cached_file(tmp_path, monkeypatch, damage):
    """A short or pointer-text raw file is refetched, not kept because it exists."""
    from eegdash.dataset.dataset import EEGDashDataset

    record = _openneuro_record()
    ds = EEGDashDataset(cache_dir=tmp_path, dataset="ds001234", records=[record])
    raw = ds.datasets[0].filecache
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(damage)
    client = _client(DATA)
    monkeypatch.setattr(downloader, "get_s3_filesystem", lambda **_: client)
    ds.download_all(n_jobs=1)
    assert raw.read_bytes() == DATA
    assert client._client.gets == 1


def test_openneuro_uri_gets_the_record_key(tmp_path, monkeypatch):
    """The record key governs even when GET reports a self-consistent short body."""
    from eegdash.dataset.exceptions import DataIntegrityError

    rec = base.EEGDashRaw(_openneuro_record(), cache_dir=str(tmp_path))
    monkeypatch.setattr(downloader, "get_s3_filesystem", lambda **_: _client(DATA[:10]))
    with pytest.raises(DataIntegrityError, match="expected 20 bytes, got 10") as caught:
        rec._download_required_files()
    assert caught.value.record == rec.record
    assert not rec.filecache.exists()


def test_openneuro_sidecar_gets_the_record_key(tmp_path, monkeypatch):
    """Dependency keys are strict too."""
    from eegdash.dataset.exceptions import DataIntegrityError

    dep_key = f"SHA256E-s5--{'c' * 64}.tsv"
    record = _openneuro_record(dep_keys=[DEP], annex_keys={RAW: KEY, DEP: dep_key})
    rec = base.EEGDashRaw(record, cache_dir=str(tmp_path))
    monkeypatch.setattr(downloader, "get_s3_filesystem", lambda **_: _client(DATA))
    with pytest.raises(DataIntegrityError, match="expected 5 bytes, got 20"):
        rec._download_required_files()
    assert not (tmp_path / "ds001234" / DEP).exists()


OPENNEURO_URI = f"s3://openneuro.org/ds001234/{RAW}"


class _ServedS3(_FakeS3):
    """S3 whose HEAD and GET report *size* bytes, whatever the body holds."""

    def __init__(self, data, size):
        super().__init__(data)
        self.size = size

    def head_object(self, **_):
        return {"ContentLength": self.size}

    def get_object(self, **kwargs):
        return {**super().get_object(**kwargs), "ContentLength": self.size}


def _served(data, size):
    client = downloader.S3Client.__new__(downloader.S3Client)
    client._client = _ServedS3(data, size)
    return client


def test_twin_mismatch_is_refused(tmp_path):
    dest = tmp_path / "sub-01_eeg.edf"
    with pytest.raises(OSError, match="expected 20 bytes, got 30"):
        downloader.download_s3_file(
            OPENNEURO_URI,
            dest,
            filesystem=_served(DATA + b"0123456789", 30),
            annex_key=KEY,
        )
    assert not dest.exists()


def test_wrong_cached_twin_is_refetched_not_certified(tmp_path):
    dest = tmp_path / "sub-01_eeg.edf"
    dest.write_bytes(DATA + b"0123456789")
    client = _served(DATA, 20)
    downloader.download_s3_file(OPENNEURO_URI, dest, filesystem=client, annex_key=KEY)
    assert client._client.gets == 1
    assert dest.read_bytes() == DATA


def test_size_matching_neither_key_nor_s3_is_refused(tmp_path):
    """Key 20, S3 30, 25 received: refused, nothing published."""
    dest = tmp_path / "sub-01_eeg.edf"
    with pytest.raises(OSError, match="Incomplete download"):
        downloader.download_s3_file(
            OPENNEURO_URI, dest, filesystem=_served(DATA + b"01234", 30), annex_key=KEY
        )
    assert list(tmp_path.iterdir()) == []


def test_objects_route_refuses_a_mismatch_whatever_s3_says(tmp_path):
    """On NEMAR's objects/<key> the key names the object: S3's size never overrides it."""
    dest = tmp_path / "sub-01_eeg.edf"
    with pytest.raises(OSError, match="expected 20 bytes, got 30"):
        downloader.download_s3_file(
            URI, dest, filesystem=_served(DATA + b"0123456789", 30)
        )
    assert list(tmp_path.iterdir()) == []

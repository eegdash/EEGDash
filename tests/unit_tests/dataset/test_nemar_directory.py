"""NEMAR directory recordings (CTF ``.ds``) across repeated, concurrent and partial runs.

Batch read-check on Expanse (2026-10-18): 167 recordings of on005279, ds005279
and on000247 failed with ``OSError: [Errno 39] Directory not empty``.
"""

from unittest.mock import patch

import pytest

from eegdash.dataset import base as base_mod

REL = "sub-1/meg/sub-1_meg.ds"
TREE = (f"{REL}/sub-1_meg.meg4", f"{REL}/sub-1_meg.res4", f"{REL}/params.dsc")
POINTERS = {
    f"{REL}/sub-1_meg.meg4": ("SHA256E-s4--aa.meg4", b""),
    f"{REL}/sub-1_meg.res4": ("SHA256E-s4--bb.res4", b""),
    f"{REL}/params.dsc": (None, b"inline"),
}
RECORD = {
    "dataset": "nm999999",
    "bids_relpath": REL,
    "storage": {"backend": "nemar", "base": "s3://nemar/nm999999", "raw_key": REL},
    "entities": {"subject": "1"},
    "entities_mne": {"subject": "1"},
}


@pytest.fixture
def fetched(monkeypatch):
    """Fake NEMAR: git tree, annex pointers and S3 objects; records every GET."""
    monkeypatch.setattr(base_mod, "_nemar_tree", lambda _ds: TREE)
    monkeypatch.setattr(base_mod, "_resolve_nemar_pointer", lambda _ds, p: POINTERS[p])
    calls = []

    def fake_download_files(pairs, **_kw):
        for uri, dest in pairs:
            calls.append((uri, dest))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"data")

    monkeypatch.setattr(base_mod.downloader, "download_files", fake_download_files)
    monkeypatch.setattr(base_mod.downloader, "get_s3_filesystem", lambda **_kw: None)
    return calls


def _publish_complete(dest):
    dest.mkdir(parents=True)
    for name in ("sub-1_meg.meg4", "sub-1_meg.res4", "params.dsc"):
        (dest / name).write_bytes(b"peer")


def test_second_download_of_published_directory_is_a_noop(tmp_path, fetched):
    """``download_all`` then ``.raw`` call the download twice in one process.

    The first call leaves ``_raw_uri`` set, so the second one used to stage the
    whole recording again (1.6 GB for on000247) and fail on the rename.
    """
    dest = tmp_path / REL
    base_mod._download_nemar_directory(RECORD, dest, filesystem=None)
    n_first = len(fetched)

    base_mod._download_nemar_directory(RECORD, dest, filesystem=None)

    assert len(fetched) == n_first  # nothing fetched again
    assert sorted(p.name for p in dest.parent.iterdir()) == [dest.name]


def test_peer_publishing_first_wins_without_error(tmp_path, monkeypatch, fetched):
    """Two jobs (ds005279 and on005279) fill one cache folder at the same time.

    A peer publishes ``dest`` while we are still downloading: keep the peer's
    complete copy, discard our staging, and never write into the peer's folder.
    """
    dest = tmp_path / REL
    staging = set()
    inner = base_mod.downloader.download_files

    def racing_download(pairs, **kw):
        pairs = list(pairs)
        staging.update(local.parent for _, local in pairs)
        inner(pairs, **kw)
        if not dest.exists():
            _publish_complete(dest)  # the other process finished first

    monkeypatch.setattr(base_mod.downloader, "download_files", racing_download)

    base_mod._download_nemar_directory(RECORD, dest, filesystem=None)

    assert (dest / "sub-1_meg.meg4").read_bytes() == b"peer"
    assert dest.with_name(dest.name + ".part") not in staging  # caller-unique
    assert sorted(p.name for p in dest.parent.iterdir()) == [dest.name]


@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_incomplete_nemar_directory_is_fetched_again(_validate, tmp_path, fetched):
    """A ``.ds`` left incomplete by an older run is re-fetched, not reported forever."""
    raw = base_mod.EEGDashRaw(dict(RECORD), cache_dir=str(tmp_path))
    raw.filecache.mkdir(parents=True)
    (raw.filecache / "sub-1_meg.res4").write_bytes(b"old")  # no .meg4

    raw._ensure_ctf_directory_complete()

    assert (raw.filecache / "sub-1_meg.meg4").read_bytes() == b"data"
    assert sorted(p.name for p in raw.filecache.parent.iterdir()) == [
        raw.filecache.name
    ]


@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_failed_ctf_check_leaves_no_empty_directory(_validate, tmp_path):
    """The check must not create the ``.ds`` it reports missing.

    An empty ``.ds`` makes every later run treat the recording as cached.
    """
    from eegdash.dataset.exceptions import DataIntegrityError  # noqa: PLC0415

    record = dict(RECORD)
    record["storage"] = {
        "backend": "local",
        "base": str(tmp_path / "nm999999"),
        "raw_key": REL,
    }
    raw = base_mod.EEGDashRaw(record, cache_dir=str(tmp_path))

    with pytest.raises(DataIntegrityError, match="CTF .ds directory incomplete"):
        raw._ensure_ctf_directory_complete()
    assert not raw.filecache.exists()


@pytest.mark.parametrize(
    "key",
    [
        "MD5E-s0--d41d8cd98f00b204e9800998ecf8427e",  # BadChannels (no extension)
        "MD5E-s2595--b9e4bb001489fd2690515c08981ecc9b",  # .infods (> 4 characters)
    ],
)
def test_pointer_to_key_without_extension_is_annexed(monkeypatch, key):
    """on000247: git-annex drops the extension of ``BadChannels`` and ``.infods``.

    The pointer text was written as the file; MNE read it as a bad channel name
    (``KeyError: '../../../../.git/annex/objects/2W/kW/MD5E-s0--...'``).
    """
    pointer = f"../../../../.git/annex/objects/2W/kW/{key}/{key}".encode()
    base_mod._resolve_nemar_pointer.cache_clear()
    monkeypatch.setattr(base_mod, "_fetch_nemar_pointer", lambda _ds, _p: pointer)
    try:
        assert base_mod._resolve_nemar_pointer("on000247", key)[0] == key
    finally:
        base_mod._resolve_nemar_pointer.cache_clear()


@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_cached_directory_with_pointer_text_is_fetched_again(
    _validate, tmp_path, fetched
):
    """A complete-looking ``.ds`` whose small files hold annex pointer text."""
    raw = base_mod.EEGDashRaw(dict(RECORD), cache_dir=str(tmp_path))
    _publish_complete(raw.filecache)
    (raw.filecache / "BadChannels").write_bytes(
        b"../../../../.git/annex/objects/2W/kW/MD5E-s0--d41d/MD5E-s0--d41d"
    )

    raw._ensure_ctf_directory_complete()

    assert not (raw.filecache / "BadChannels").exists()
    assert (raw.filecache / "sub-1_meg.meg4").read_bytes() == b"data"
    assert sorted(p.name for p in raw.filecache.parent.iterdir()) == [
        raw.filecache.name
    ]

"""Shared sidecar refreshes must never remove another caller's completed file."""

import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from botocore.response import StreamingBody

from eegdash import downloader

PAYLOAD = b"name\ttype\tx\ty\tz\nS1\tsource\t0\t0\t0\n"
URI = "s3://openneuro.org/ds006377/sub-236/nirs/sub-236_optodes.tsv"


def _client():
    client = downloader.S3Client.__new__(downloader.S3Client)
    client._client = Mock()
    client._client.head_object.return_value = {"ContentLength": len(PAYLOAD)}
    client._client.get_object.side_effect = lambda **_: {
        "ContentLength": len(PAYLOAD),
        "Body": StreamingBody(BytesIO(PAYLOAD), len(PAYLOAD)),
    }
    return client


def _download(dest, api, client):
    if api == "single":
        return downloader.download_s3_file(URI, dest, filesystem=client)
    return downloader.download_files([(URI, dest)], filesystem=client)


def _publish(dest, start, published):
    assert start.wait(20)
    _download(dest, "single", _client())
    published.set()


@pytest.mark.parametrize("api", ["single", "batch"])
@pytest.mark.parametrize("worker", ["thread", "process"])
def test_shared_sidecar_stale_size_never_unlinks_completed_file(tmp_path, api, worker):
    """Pause a stale stat while a second thread/process publishes the full file."""
    dest = tmp_path / "sub-236_optodes.tsv"
    dest.write_bytes(b"old")
    ctx = multiprocessing.get_context("spawn")
    start, published = ctx.Event(), ctx.Event()
    other = (threading.Thread if worker == "thread" else ctx.Process)(
        target=_publish, args=(dest, start, published)
    )
    real_stat = Path.stat
    refresh_thread = None
    stale = True

    def stat(path, *args, **kwargs):
        nonlocal stale
        value = real_stat(path, *args, **kwargs)
        if path == dest and threading.get_ident() == refresh_thread and stale:
            stale = False
            # Return the captured size only after the other caller has
            # published a complete file.
            start.set()
            assert published.wait(20)
            return value
        return value

    client = _client()
    get_object = client._client.get_object.side_effect

    def checked_get(**kwargs):
        # Refresh must leave the published file available while GET is in flight.
        assert dest.read_bytes() == PAYLOAD
        return get_object(**kwargs)

    client._client.get_object.side_effect = checked_get

    def refresh():
        nonlocal refresh_thread
        refresh_thread = threading.get_ident()
        return _download(dest, api, client)

    # Hold the stat result rather than sleeping; this works across processes too.
    # Patch exists separately so the captured stale stat is the size check.
    real_exists = Path.exists

    def exists(path):
        if path == dest and threading.get_ident() == refresh_thread:
            return True
        return real_exists(path)

    other.start()
    try:
        with patch.object(Path, "stat", stat), patch.object(Path, "exists", exists):
            with ThreadPoolExecutor(1) as pool:
                pool.submit(refresh).result(timeout=30)
    finally:
        start.set()
        other.join(30)
    assert not other.is_alive()
    if worker == "process":
        assert other.exitcode == 0
    assert dest.read_bytes() == PAYLOAD
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("api", ["single", "batch"])
def test_shared_sidecar_disappearing_during_cache_check_is_refetched(tmp_path, api):
    dest = tmp_path / "sub-236_optodes.tsv"
    dest.write_bytes(PAYLOAD)
    real_stat = Path.stat
    missing = True

    def stat(path, *args, **kwargs):
        nonlocal missing
        if path == dest and missing:
            missing = False
            dest.unlink()
            raise FileNotFoundError(str(dest))
        return real_stat(path, *args, **kwargs)

    real_exists = Path.exists
    with (
        patch.object(Path, "exists", lambda path: path == dest or real_exists(path)),
        patch.object(Path, "stat", stat),
    ):
        _download(dest, api, _client())
    assert dest.read_bytes() == PAYLOAD


@pytest.mark.parametrize("api", ["single", "batch"])
def test_shared_sidecar_short_get_preserves_existing_file(tmp_path, api):
    dest = tmp_path / "sub-236_optodes.tsv"
    dest.write_bytes(b"existing")
    client = _client()
    body = Mock()
    body.iter_chunks.return_value = [b"short"]
    client._client.get_object.side_effect = lambda **_: {
        "ContentLength": len(PAYLOAD),
        "Body": body,
    }
    with pytest.raises(OSError, match="Incomplete download"):
        _download(dest, api, client)
    assert dest.read_bytes() == b"existing"
    assert not list(tmp_path.glob("*.part"))


def test_shared_sidecar_forced_refresh_preserves_existing_file(tmp_path):
    dest = tmp_path / "sub-236_optodes.tsv"
    dest.write_bytes(PAYLOAD)
    client = _client()
    get_object = client._client.get_object.side_effect

    def checked_get(**kwargs):
        assert dest.read_bytes() == PAYLOAD
        return get_object(**kwargs)

    client._client.get_object.side_effect = checked_get
    downloader.download_files([(URI, dest)], filesystem=client, skip_existing=False)
    assert dest.read_bytes() == PAYLOAD

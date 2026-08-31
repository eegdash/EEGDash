"""Regression coverage for immutable NM000134 BIDS stimulus pointers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from eegdash.dataset import _stimulus_viewer as stimulus_viewer
from eegdash.dataset import base as base_module

NM000134_STIMULUS_GIT_REF = "61b04adf7bca47f220b85f3744a610b44046c62f"
_POINTER_BYTES = b"/annex/objects/MD5E-s37098--21db1a0c5190bbdf111fc6ee339eee31.jpg"


class _PointerResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return _POINTER_BYTES


@pytest.fixture
def clear_nemar_pointer_cache():
    """Keep a mocked immutable pointer out of the shared resolver cache."""
    base_module._resolve_nemar_pointer.cache_clear()
    yield
    base_module._resolve_nemar_pointer.cache_clear()


@pytest.mark.parametrize(
    ("ref", "expected_url"),
    [
        (
            None,
            "https://raw.githubusercontent.com/NEMARDatasets/nm000134/HEAD/"
            "stimuli/16540.jpg",
        ),
        (
            NM000134_STIMULUS_GIT_REF,
            "https://raw.githubusercontent.com/NEMARDatasets/nm000134/"
            "61b04adf7bca47f220b85f3744a610b44046c62f/stimuli/16540.jpg",
        ),
    ],
    ids=["default-head", "immutable-ref"],
)
def test_fetch_nemar_pointer_builds_requested_github_ref(
    monkeypatch, ref, expected_url
):
    """A pointer fetch must keep HEAD by default and honor a supplied ref."""
    requests = []

    def open_url(request, *, timeout):
        requests.append((request, timeout))
        return _PointerResponse()

    monkeypatch.setattr(base_module.urllib.request, "urlopen", open_url)
    kwargs = {} if ref is None else {"ref": ref}

    assert (
        base_module._fetch_nemar_pointer(
            "nm000134", "stimuli/16540.jpg", **kwargs
        )
        == _POINTER_BYTES
    )
    assert requests[0][0].full_url == expected_url


def _recording_with_stimulus_event(tmp_path: Path) -> Path:
    recording = (
        tmp_path
        / "sub-09"
        / "ses-01"
        / "eeg"
        / "sub-09_ses-01_task-images_run-02_eeg.edf"
    )
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b"EDF")
    recording.with_name(recording.name.replace("_eeg.edf", "_events.tsv")).write_text(
        "onset\tduration\ttrial_type\n1\t0\tstim_test,16540,-1,1\n"
    )
    return recording


@pytest.mark.parametrize(
    ("dataset_id", "expected_ref"),
    [
        ("nm000134", NM000134_STIMULUS_GIT_REF),
        ("nm000999", "HEAD"),
    ],
    ids=["nm000134-is-pinned", "other-nemar-datasets-keep-head"],
)
def test_viewer_stimulus_materialization_passes_the_expected_pointer_ref(
    tmp_path, monkeypatch, clear_nemar_pointer_cache, dataset_id, expected_ref
):
    """Only NM000134 viewer images resolve through the immutable Git release."""
    recording = _recording_with_stimulus_event(tmp_path)
    raw_dataset = SimpleNamespace(
        bids_root=tmp_path,
        record={
            "dataset": dataset_id,
            "storage": {"backend": "nemar", "base": f"s3://nemar/{dataset_id}"},
        },
    )
    pointer_calls = []

    def fetch_pointer(pointer_dataset_id, relpath, *, ref="HEAD"):
        pointer_calls.append((pointer_dataset_id, relpath, ref))
        return _POINTER_BYTES

    def download_s3_file(_uri, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"jpeg-bytes")
        return destination

    def download_via_nemar(*_args):
        if dataset_id == "nm000134":
            pytest.fail("a pinned viewer asset must not consult the mutable latest manifest")
        return False

    monkeypatch.setattr(base_module, "_download_via_nemar", download_via_nemar)
    monkeypatch.setattr(base_module, "_fetch_nemar_pointer", fetch_pointer)
    monkeypatch.setattr(stimulus_viewer.downloader, "download_s3_file", download_s3_file)

    expected_image = tmp_path / "stimuli" / "16540.jpg"
    assert stimulus_viewer.stimulus_files(raw_dataset, recording) == {
        "16540": expected_image
    }
    assert pointer_calls == [
        (dataset_id, "stimuli/16540.jpg", expected_ref),
    ]

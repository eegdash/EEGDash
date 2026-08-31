import base64
import importlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module():
    return importlib.import_module("eegdash.dataset._stimulus_viewer")


def _recording_with_events(tmp_path: Path, events_text: str) -> Path:
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
        events_text
    )
    return recording


def _nemar_raw(tmp_path: Path):
    return SimpleNamespace(
        bids_root=tmp_path,
        record={
            "dataset": "nm000134",
            "storage": {"backend": "nemar", "base": "s3://nemar/nm000134"},
        },
    )


def _payload(html) -> dict:
    match = re.search(r"var payload = (.+?), origin =", html.data, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def test_stimulus_files_materialize_only_nm_event_images(tmp_path, monkeypatch):
    module = _module()
    recording = _recording_with_events(
        tmp_path,
        "onset\tduration\ttrial_type\tstim_file\timage_id\n"
        "1\t0\tstim_test,16595,-1,1\t\t\n"
        "2\t0\tordinary\tstimuli/00042.jpg\t\n"
        "3\t0\tordinary\tstimuli/00043.jpg\t42\n",
    )
    local_image = tmp_path / "stimuli" / "00042.jpg"
    local_image.parent.mkdir()
    local_image.write_bytes(b"local-jpeg")
    image_with_explicit_id = tmp_path / "stimuli" / "00043.jpg"
    image_with_explicit_id.write_bytes(b"explicit-id-jpeg")
    calls = []

    def write_fake_jpeg(recording_dataset, image_id):
        calls.append((recording_dataset.record["dataset"], image_id))
        image = recording_dataset.bids_root / "stimuli" / f"{int(image_id):05d}.jpg"
        image.parent.mkdir(exist_ok=True)
        image.write_bytes(b"downloaded-jpeg")
        return image

    monkeypatch.setattr(module, "_materialize_nemar_asset", write_fake_jpeg)

    assert module.stimulus_files(_nemar_raw(tmp_path), recording) == {
        "16595": tmp_path / "stimuli" / "16595.jpg",
        "00042": local_image,
        "42": image_with_explicit_id,
    }
    assert calls == [("nm000134", "16595")]


def test_plot_sends_stimuli_separately_and_counts_them(tmp_path, monkeypatch):
    module = _module()
    recording = _recording_with_events(
        tmp_path,
        "onset\tduration\ttrial_type\n1\t0\tstim_test,16595,-1,1\n",
    )
    image = tmp_path / "stimuli" / "16595.jpg"
    image.parent.mkdir()
    image.write_bytes(b"jpeg-bytes")
    dataset = SimpleNamespace(datasets=[_nemar_raw(tmp_path)])
    monkeypatch.setattr(module, "_recording", lambda _dataset, _index: recording)

    html = module.plot(dataset, max_bytes=1_000_000)
    payload = _payload(html)

    assert "stimuli" in payload
    assert base64.b64decode(payload["stimuli"]["16595"]) == b"jpeg-bytes"
    assert all(not entry["name"].endswith(".jpg") for entry in payload["files"])

    events = recording.with_name(recording.name.replace("_eeg.edf", "_events.tsv"))
    without_stimuli = sum(
        4 * math.ceil(path.stat().st_size / 3) for path in (recording, events)
    )
    with pytest.raises(ValueError, match="base64"):
        module.plot(dataset, max_bytes=without_stimuli)


def test_plot_without_stimuli_delegates_upstream_without_download(tmp_path, monkeypatch):
    module = _module()
    recording = _recording_with_events(
        tmp_path,
        "onset\tduration\ttrial_type\n1\t0\tordinary\n",
    )
    dataset = SimpleNamespace(datasets=[_nemar_raw(tmp_path)])
    marker = object()
    calls = []

    monkeypatch.setattr(module, "_recording", lambda _dataset, _index: recording)
    monkeypatch.setattr(module, "_upstream_plot", lambda *args, **kwargs: calls.append((args, kwargs)) or marker)
    monkeypatch.setattr(
        module,
        "_materialize_nemar_asset",
        lambda *_args: pytest.fail("a non-stimulus event must not download an image"),
    )

    assert module.plot(dataset, height=333, cdn_url="https://viewer.example", max_bytes=77) is marker
    assert calls == [
        ((dataset,), {"index": 0, "height": 333, "cdn_url": "https://viewer.example", "max_bytes": 77})
    ]

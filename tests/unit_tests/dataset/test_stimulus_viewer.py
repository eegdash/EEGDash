import base64
import importlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError


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
    match = re.search(
        r'var payload = JSON\.parse\(("(?:[^"\\]|\\.)*")\), src =',
        html.data,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(json.loads(match.group(1)))


def test_stimulus_files_preserve_event_ids_and_materialize_nm_images(
    tmp_path, monkeypatch
):
    module = _module()
    recording = _recording_with_events(
        tmp_path,
        "onset\tduration\ttrial_type\tstim_file\timage_id\n"
        "1\t0\tstim_test,16595,-1,1\t\t\n"
        "2\t0\tordinary\tstimuli/00042.jpg\t\n"
        "3\t0\tordinary\tstimuli/00043.jpg\tscene-A\n"
        "4\t0\tordinary\tstimuli/scene-B.jpg\t\n",
    )
    local_image = tmp_path / "stimuli" / "00042.jpg"
    local_image.parent.mkdir()
    local_image.write_bytes(b"local-jpeg")
    image_with_explicit_id = tmp_path / "stimuli" / "00043.jpg"
    image_with_explicit_id.write_bytes(b"explicit-id-jpeg")
    nonnumeric_stim_file = tmp_path / "stimuli" / "scene-B.jpg"
    nonnumeric_stim_file.write_bytes(b"nonnumeric-jpeg")
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
        "scene-A": image_with_explicit_id,
        "scene-B": nonnumeric_stim_file,
    }
    assert calls == [("nm000134", "16595")]


@pytest.mark.parametrize(
    ("stim_file", "local_path", "expected_id"),
    [
        ("stimuli/.hidden", "stimuli/.hidden", "stimuli/.hidden"),
        ("stimuli/foo.-", "stimuli/foo.-", "foo.-"),
        ("stimuli/foo.é", "stimuli/foo.é", "foo.é"),
        (r"stimuli\nested\scene.jpg", "stimuli/nested/scene.jpg", "scene"),
    ],
)
def test_stimulus_files_match_viewer_filename_ids(
    tmp_path, stim_file, local_path, expected_id
):
    module = _module()
    recording = _recording_with_events(
        tmp_path,
        f"onset\tduration\tstim_file\n1\t0\t{stim_file}\n",
    )
    image = tmp_path / local_path
    image.parent.mkdir(parents=True)
    image.write_bytes(b"local-jpeg")

    assert module.stimulus_files(_nemar_raw(tmp_path), recording) == {
        expected_id: image
    }


@pytest.mark.parametrize(
    "make_error",
    [
        pytest.param(
            lambda: ClientError(
                {
                    "Error": {"Code": "InternalError"},
                    "ResponseMetadata": {"HTTPStatusCode": 503},
                },
                "GetObject",
            ),
            id="client-error",
        ),
        pytest.param(
            lambda: EndpointConnectionError(endpoint_url="https://nemar.example"),
            id="endpoint-connection",
        ),
    ],
)
def test_stimulus_files_omits_optional_s3_error_and_keeps_sibling(
    tmp_path, monkeypatch, make_error
):
    module = _module()
    recording = _recording_with_events(
        tmp_path,
        "onset\tduration\ttrial_type\n"
        "1\t0\tstim_test,16595,-1,1\n"
        "2\t0\tstim_test,16596,-1,1\n",
    )
    expected = tmp_path / "stimuli" / "16596.jpg"

    def resolve(**kwargs):
        if kwargs["relpath"] == "stimuli/16595.jpg":
            raise make_error()
        return "s3://nemar/nm000134/objects/16596"

    def download(_source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"sibling-jpeg")
        return destination

    monkeypatch.setattr(module, "_resolve_one_nemar_entry", resolve)
    monkeypatch.setattr(module.downloader, "download_s3_file", download)

    assert module.stimulus_files(_nemar_raw(tmp_path), recording) == {
        "16596": expected
    }
    assert expected.read_bytes() == b"sibling-jpeg"


def test_stimulus_files_uses_local_canonical_asset_without_stim_file(
    tmp_path, monkeypatch
):
    module = _module()
    recording = _recording_with_events(
        tmp_path,
        "onset\tduration\timage_id\n1\t0\t00042\n",
    )
    image = tmp_path / "stimuli" / "00042.jpg"
    image.parent.mkdir()
    image.write_bytes(b"local-jpeg")
    local_raw = SimpleNamespace(
        bids_root=tmp_path,
        record={"dataset": "local", "storage": {"backend": "local"}},
    )
    monkeypatch.setattr(
        module,
        "_materialize_nemar_asset",
        lambda *_args: pytest.fail("a local canonical asset must win before download"),
    )

    assert module.stimulus_files(local_raw, recording) == {"00042": image}


def test_stimulus_files_does_not_substitute_a_declared_missing_stim_file(
    tmp_path, monkeypatch
):
    module = _module()
    recording = _recording_with_events(
        tmp_path,
        "onset\tduration\tstim_file\timage_id\n"
        "1\t0\tstimuli/declared.jpg\t42\n",
    )
    canonical_image = tmp_path / "stimuli" / "00042.jpg"
    canonical_image.parent.mkdir()
    canonical_image.write_bytes(b"wrong-image")
    monkeypatch.setattr(
        module,
        "_materialize_nemar_asset",
        lambda *_args: pytest.fail("a declared stimulus must not use canonical lookup"),
    )

    assert module.stimulus_files(_nemar_raw(tmp_path), recording) == {}


def test_stimulus_files_stops_remote_attempts_after_two_consecutive_misses(
    tmp_path, monkeypatch
):
    module = _module()
    recording = _recording_with_events(
        tmp_path,
        "onset\tduration\ttrial_type\tstim_file\n"
        "1\t0\tstim_test,16595,-1,1\t\n"
        "2\t0\tstim_test,16596,-1,1\t\n"
        "3\t0\tstim_test,16597,-1,1\t\n"
        "4\t0\tstim_test,16598,-1,1\t\n"
        "5\t0\tstim_test,16599,-1,1\t\n"
        "6\t0\tordinary\tstimuli/local.jpg\n",
    )
    local_image = tmp_path / "stimuli" / "local.jpg"
    local_image.parent.mkdir()
    local_image.write_bytes(b"local-jpeg")
    attempts = []
    successful_image = tmp_path / "stimuli" / "16596.jpg"

    def missing_asset(_recording_dataset, image_id):
        attempts.append(image_id)
        if image_id == "16596":
            successful_image.write_bytes(b"remote-jpeg")
            return successful_image
        return None

    monkeypatch.setattr(module, "_materialize_nemar_asset", missing_asset)

    assert module.stimulus_files(_nemar_raw(tmp_path), recording) == {
        "16596": successful_image,
        "local": local_image,
    }
    assert attempts == ["16595", "16596", "16597", "16598"]


def test_stimulus_files_skips_pathologically_long_numeric_image_ids(
    tmp_path, monkeypatch
):
    module = _module()
    image_id = "9" * 5000
    recording = _recording_with_events(
        tmp_path,
        f"onset\tduration\timage_id\n1\t0\t{image_id}\n",
    )
    monkeypatch.setattr(
        module,
        "_materialize_nemar_asset",
        lambda *_args: pytest.fail("a pathological numeric ID must not materialize"),
    )

    assert module._is_numeric_image_id("9" * 18)
    assert not module._is_numeric_image_id("9" * 19)
    assert module.stimulus_files(_nemar_raw(tmp_path), recording) == {}


@pytest.mark.parametrize("failure", ["resolve", "download"])
def test_plot_omits_optional_nemar_asset_after_remote_failure(
    tmp_path, monkeypatch, failure
):
    module = _module()
    recording = _recording_with_events(
        tmp_path,
        "onset\tduration\ttrial_type\n1\t0\tstim_test,16595,-1,1\n",
    )
    dataset = SimpleNamespace(datasets=[_nemar_raw(tmp_path)])
    marker = object()

    monkeypatch.setattr(module, "_recording", lambda _dataset, _index: recording)
    monkeypatch.setattr(module, "_upstream_plot", lambda *_args, **_kwargs: marker)
    if failure == "resolve":
        monkeypatch.setattr(
            module,
            "_resolve_one_nemar_entry",
            lambda **_kwargs: (_ for _ in ()).throw(URLError("missing image")),
        )
    else:
        monkeypatch.setattr(
            module,
            "_resolve_one_nemar_entry",
            lambda **_kwargs: "s3://nemar/nm000134/objects/missing",
        )
        monkeypatch.setattr(
            module.downloader,
            "download_s3_file",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("download failed")),
        )

    assert module.plot(dataset) is marker


def test_plot_rejects_stimulus_symlink_outside_bids_root(tmp_path, monkeypatch):
    module = _module()
    bids_root = tmp_path / "bids"
    recording = _recording_with_events(
        bids_root,
        "onset\tduration\tstim_file\n1\t0\tstimuli/00042.jpg\n",
    )
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside-image-bytes")
    stimulus = bids_root / "stimuli" / "00042.jpg"
    stimulus.parent.mkdir()
    try:
        stimulus.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    dataset = SimpleNamespace(datasets=[_nemar_raw(bids_root)])
    marker = SimpleNamespace(data="upstream viewer")
    monkeypatch.setattr(module, "_recording", lambda _dataset, _index: recording)
    monkeypatch.setattr(module, "_upstream_plot", lambda *_args, **_kwargs: marker)
    monkeypatch.setattr(
        module,
        "_materialize_nemar_asset",
        lambda *_args: pytest.fail("unsafe local path must not materialize"),
    )

    assert module.plot(dataset) is marker
    assert base64.b64encode(outside.read_bytes()).decode() not in marker.data


def test_plot_skips_unmaterializable_nonnumeric_image_id(tmp_path, monkeypatch):
    module = _module()
    recording = _recording_with_events(
        tmp_path,
        "onset\tduration\tstim_file\timage_id\n"
        "1\t0\tstimuli/00042.jpg\tscene-A\n",
    )
    dataset = SimpleNamespace(datasets=[_nemar_raw(tmp_path)])
    marker = object()
    monkeypatch.setattr(module, "_recording", lambda _dataset, _index: recording)
    monkeypatch.setattr(module, "_upstream_plot", lambda *_args, **_kwargs: marker)
    monkeypatch.setattr(
        module,
        "_materialize_nemar_asset",
        lambda *_args: pytest.fail("a nonnumeric image ID must not resolve remotely"),
    )

    assert module.plot(dataset) is marker


def test_plot_sends_stimuli_separately_and_counts_them(tmp_path, monkeypatch):
    module = _module()
    recording = _recording_with_events(
        tmp_path,
        "onset\tduration\tstim_file\timage_id\n"
        "1\t0\tstimuli/00042.jpg\tscene-A\n",
    )
    image = tmp_path / "stimuli" / "00042.jpg"
    image.parent.mkdir()
    image.write_bytes(b"jpeg-bytes")
    dataset = SimpleNamespace(datasets=[_nemar_raw(tmp_path)])
    monkeypatch.setattr(module, "_recording", lambda _dataset, _index: recording)

    html = module.plot(dataset, max_bytes=1_000_000)
    payload = _payload(html)

    assert "stimuli" in payload
    assert base64.b64decode(payload["stimuli"]["scene-A"]) == b"jpeg-bytes"
    assert all(not entry["name"].endswith(".jpg") for entry in payload["files"])

    events = recording.with_name(recording.name.replace("_eeg.edf", "_events.tsv"))
    without_stimuli = sum(
        4 * math.ceil(path.stat().st_size / 3) for path in (recording, events)
    )
    with pytest.raises(ValueError, match="base64"):
        module.plot(dataset, max_bytes=without_stimuli)


def test_plot_requires_the_configured_viewer_origin_before_sending_payload(
    tmp_path, monkeypatch
):
    module = _module()
    recording = _recording_with_events(
        tmp_path,
        "onset\tduration\tstim_file\n1\t0\tstimuli/00042.jpg\n",
    )
    image = tmp_path / "stimuli" / "00042.jpg"
    image.parent.mkdir()
    image.write_bytes(b"jpeg-bytes")
    dataset = SimpleNamespace(datasets=[_nemar_raw(tmp_path)])
    monkeypatch.setattr(module, "_recording", lambda _dataset, _index: recording)

    html = module.plot(dataset, max_bytes=1_000_000)

    assert "e.source === frame.contentWindow && e.origin === origin" in html.data
    assert "send(origin);" in html.data
    assert "send(e.origin)" not in html.data


def test_plot_derives_origin_from_exact_viewer_source_url(tmp_path, monkeypatch):
    module = _module()
    recording = _recording_with_events(
        tmp_path,
        "onset\tduration\tstim_file\n1\t0\tstimuli/00042.jpg\n",
    )
    image = tmp_path / "stimuli" / "00042.jpg"
    image.parent.mkdir()
    image.write_bytes(b"jpeg-bytes")
    dataset = SimpleNamespace(datasets=[_nemar_raw(tmp_path)])
    monkeypatch.setattr(module, "_recording", lambda _dataset, _index: recording)

    html = module.plot(
        dataset,
        cdn_url="https://EEGDASH.github.io:443/eegdash-viewer",
        max_bytes=1_000_000,
    )

    assert (
        'src = "https://EEGDASH.github.io:443/eegdash-viewer/index.html?embed=1", '
        "origin = new URL(src).origin"
    ) in html.data
    assert "frame.src = src;" in html.data
    assert 'origin = "https://EEGDASH.github.io:443"' not in html.data


def test_plot_preserves_proto_stimulus_in_json_payload(tmp_path, monkeypatch):
    module = _module()
    recording = _recording_with_events(
        tmp_path,
        "onset\tduration\tstim_file\n1\t0\tstimuli/__proto__.jpg\n",
    )
    image = tmp_path / "stimuli" / "__proto__.jpg"
    image.parent.mkdir()
    image.write_bytes(b"proto-jpeg")
    dataset = SimpleNamespace(datasets=[_nemar_raw(tmp_path)])
    monkeypatch.setattr(module, "_recording", lambda _dataset, _index: recording)

    html = module.plot(dataset, max_bytes=1_000_000)
    assert "stimuli = Object.create(null);" in html.data
    payload = _payload(html)

    assert base64.b64decode(payload["stimuli"]["__proto__"]) == b"proto-jpeg"


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

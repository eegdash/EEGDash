"""BIDS stimulus assets for the serverless in-notebook EEGDash viewer."""

from __future__ import annotations

import base64
import csv
import json
import re
import uuid
from pathlib import Path, PurePosixPath
from urllib.error import URLError
from urllib.parse import urlsplit

from botocore.exceptions import BotoCoreError, ClientError
from mne.utils import _soft_import

from braindecode.datasets._notebook_viewer import (
    _BIDS,
    CDN,
    MAX_BYTES,
    _recording,
    recording_files,
)
from braindecode.datasets._notebook_viewer import plot as _upstream_plot

from .. import downloader
from .base import _resolve_one_nemar_entry

_NM_STIMULUS = re.compile(r"^stim_(?:train|test),(\d+),")

_SCRIPT = """<iframe id=%(id)s title="eegdash trace viewer" style="width:100%%;height:%(height)spx;
border:1px solid var(--jp-border-color1,#d9dce1);border-radius:6px;background:transparent"></iframe>
<script>
(function () {
  var self = document.currentScript, id = %(id)s;
  var frame = (self && self.previousElementSibling && self.previousElementSibling.tagName === "IFRAME")
    ? self.previousElementSibling : document.getElementById(id);
  if (!frame) { console.error("eegdash viewer: output iframe " + id + " not found"); return; }
  var payload = JSON.parse(%(payload)s), origin = %(origin)s, files = null, pose = null, stimuli = null;
  function decode(b64) {
    if (Uint8Array.fromBase64) return Uint8Array.fromBase64(b64);
    var bin = atob(b64), out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  function send(target) {
    try {
      if (!files) {
        files = payload.files.map(function (f) { return new File([decode(f.b64)], f.name); });
        pose = payload.pose ? "data:application/json;base64," + payload.pose : null;
        stimuli = Object.create(null);
        Object.keys(payload.stimuli).forEach(function (id) {
          stimuli[id] = new Blob([decode(payload.stimuli[id])], { type: "image/jpeg" });
        });
        payload = null;
      }
      frame.contentWindow.postMessage(
        { type: "eegdash-viewer:open", files: files, pose: pose, stimuli: stimuli },
        target || origin
      );
    } catch (err) {
      frame.insertAdjacentHTML("afterend", '<div style="font:12px system-ui;color:#b3261e">eegdash viewer: '
        + String(err.message).replace(/</g, "&lt;") + "</div>");
    }
  }
  window.addEventListener("message", function onMessage(e) {
    if (e.source === frame.contentWindow && e.data && e.data.type === "eegdash-viewer:ready") send(e.origin);
    else if (!frame.isConnected) window.removeEventListener("message", onMessage);
  });
  frame.src = %(src)s;
})();
</script>"""


def _event_stimuli(recording: Path) -> dict[str, str | None]:
    """Return viewer image IDs and optional BIDS ``stim_file`` values."""
    files, _ = recording_files(recording)
    events = next((path for path in files if path.name.endswith("_events.tsv")), None)
    if events is None:
        return {}

    stimuli: dict[str, str | None] = {}
    with events.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            stim_file = _bids_cell(row.get("stim_file"))
            image_id = _stimulus_id(
                stim_file,
                _bids_cell(row.get("image_id")),
                _bids_cell(row.get("trial_type")),
            )
            if image_id is not None:
                stimuli.setdefault(image_id, stim_file)
    return stimuli


def _bids_cell(value: str | None) -> str | None:
    """Return a nonempty BIDS TSV cell, treating ``n/a`` as absent."""
    if value is None:
        return None
    value = value.strip()
    return value if value and value.lower() != "n/a" else None


def _stimulus_id(
    stim_file: str | None, image_id: str | None, trial_type: str | None
) -> str | None:
    """Extract the image ID exactly as the viewer will read it from events."""
    if image_id:
        return image_id
    if stim_file:
        filename = re.split(r"[\\/]", stim_file)[-1] or stim_file
        return re.sub(r"\.[A-Za-z0-9]+$", "", filename) or stim_file
    match = _NM_STIMULUS.match(trial_type or "")
    return match.group(1) if match else None


def _safe_bids_path(bids_root: Path, path: Path) -> Path | None:
    """Return a resolved path only when it remains within ``bids_root``."""
    try:
        root = bids_root.resolve()
        resolved = path.resolve()
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


def _local_stimulus_path(bids_root: Path, stim_file: str | None) -> Path | None:
    """Return a safe local ``stimuli/`` path named by an event row."""
    if not stim_file:
        return None
    candidate = PurePosixPath(stim_file.replace("\\", "/"))
    if not candidate.parts or candidate.parts[0] != "stimuli" or ".." in candidate.parts:
        return None
    return _safe_bids_path(bids_root, bids_root.joinpath(*candidate.parts))


def _is_numeric_image_id(image_id: str) -> bool:
    return image_id.isascii() and image_id.isdigit()


def _materialize_nemar_asset(recording_dataset, image_id: str) -> Path | None:
    """Resolve and cache one NEMAR image by its canonical BIDS path."""
    record = getattr(recording_dataset, "record", {})
    storage = record.get("storage") or {}
    if storage.get("backend") != "nemar":
        return None
    dataset_id = record.get("dataset")
    base = str(storage.get("base") or "").rstrip("/")
    bids_root = getattr(recording_dataset, "bids_root", None)
    if not dataset_id or not base or bids_root is None or not _is_numeric_image_id(image_id):
        return None

    relpath = f"stimuli/{int(image_id):05d}.jpg"
    destination = _safe_bids_path(Path(bids_root), Path(bids_root) / relpath)
    if destination is None:
        return None
    if destination.is_file():
        return destination
    annex_keys = storage.get("annex_keys") or {}
    sidecar_inline = storage.get("sidecar_inline") or {}
    try:
        object_uri = _resolve_one_nemar_entry(
            dataset_id=dataset_id,
            relpath=relpath,
            base=base,
            dest=destination,
            stored_key=annex_keys.get(relpath),
            stored_sidecar=sidecar_inline.get(relpath),
            is_required=False,
        )
        if object_uri:
            downloader.download_s3_file(object_uri, destination)
    except (BotoCoreError, ClientError, OSError, URLError):
        return None
    return destination if destination.is_file() else None


def stimulus_files(recording_dataset, recording_path) -> dict[str, Path]:
    """Return local BIDS image files referenced by one recording's events."""
    bids_root = getattr(recording_dataset, "bids_root", None)
    if bids_root is None:
        return {}
    root = Path(bids_root)
    files: dict[str, Path] = {}
    for image_id, stim_file in _event_stimuli(Path(recording_path)).items():
        local = _local_stimulus_path(root, stim_file)
        if local is None and stim_file is None and _is_numeric_image_id(image_id):
            local = _local_stimulus_path(root, f"stimuli/{int(image_id):05d}.jpg")
        if stim_file is not None and local is None:
            continue
        if local is not None and local.is_file():
            files[image_id] = local
            continue
        if not _is_numeric_image_id(image_id):
            continue
        materialized = _materialize_nemar_asset(recording_dataset, image_id)
        if materialized is not None and materialized.is_file():
            files[image_id] = materialized
    return files


def _base64_size(path: Path) -> int:
    return 4 * -(-path.stat().st_size // 3)


def _viewer_names(files: list[Path]) -> list[str]:
    recording = files[0]
    head = (
        recording.name
        if recording.stem.rpartition("_")[2] in _BIDS
        else f"{recording.stem}_eeg{recording.suffix.lower()}"
    )
    return [head] + [
        head.rsplit("_", 1)[0] + "_eeg.fdt"
        if path.suffix.lower() == ".fdt"
        else path.name
        for path in files[1:]
    ]


def _build_html(
    recording: Path,
    stimuli: dict[str, Path],
    *,
    height: int,
    cdn_url: str,
    max_bytes: int,
) -> str:
    """Build the upstream bridge payload with a separate image mapping."""
    url = urlsplit(cdn_url)
    if (
        url.scheme not in ("http", "https")
        or not url.netloc
        or url.username
        or url.query
        or url.fragment
        or url.path.endswith(("index.html", "index.htm"))
    ):
        raise ValueError(f"cdn_url must be the viewer's base http(s) URL, got {cdn_url!r}")

    files, pose = recording_files(recording)
    encoded = sum(_base64_size(path) for path in files)
    if pose:
        encoded += _base64_size(pose)
    encoded += sum(_base64_size(path) for path in stimuli.values())
    if encoded > max_bytes:
        raise ValueError(
            f"{files[0].name}: {encoded / 2**20:.1f} MiB of base64 would be inlined into the "
            f"notebook output (max_bytes={max_bytes / 2**20:.1f} MiB); crop/downsample or raise it"
        )

    literals = {
        "id": f"eegdash-viewer-{uuid.uuid4().hex[:8]}",
        "height": int(height),
        "payload": {
            "files": [
                {"name": name, "b64": base64.b64encode(path.read_bytes()).decode()}
                for name, path in zip(_viewer_names(files), files, strict=True)
            ],
            "pose": base64.b64encode(pose.read_bytes()).decode() if pose else None,
            "stimuli": {
                image_id: base64.b64encode(path.read_bytes()).decode()
                for image_id, path in stimuli.items()
            },
        },
        "origin": f"{url.scheme}://{url.netloc}",
        "src": f"{url.geturl().rstrip('/')}/index.html?embed=1",
    }
    literals["payload"] = json.dumps(literals["payload"]).replace("<", "\\u003c")
    return _SCRIPT % {
        key: json.dumps(value).replace("<", "\\u003c")
        for key, value in literals.items()
    }


def plot(
    dataset,
    index: int = 0,
    *,
    height: int = 520,
    cdn_url: str = CDN,
    max_bytes: int = MAX_BYTES,
):
    """Show a recording with its BIDS image stimuli in eegdash-viewer."""
    recording = _recording(dataset, index)
    stimuli = stimulus_files(dataset.datasets[index], recording)
    if not stimuli:
        return _upstream_plot(
            dataset, index=index, height=height, cdn_url=cdn_url, max_bytes=max_bytes
        )
    ipython = _soft_import("IPython", purpose=f"{type(dataset).__name__}.plot()")
    return ipython.display.HTML(
        _build_html(
            recording,
            stimuli,
            height=height,
            cdn_url=cdn_url,
            max_bytes=max_bytes,
        )
    )


__all__ = ["plot", "stimulus_files"]

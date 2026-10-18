"""EEGDashRaw reads deposit quirks without writing a byte into the cached dataset.

EEGDash used to fix these by rewriting the cached file: stale BrainVision
``DataFile=``/``MarkerFile=`` (plus stub ``.vmrk`` files), an EEGLAB ``.set``
naming another recording's ``.fdt`` or a truncated ``.fdt`` (``savemat``), and
non-UTF-8 TSVs. The rewritten file no longer matched its S3 size, so each new
session downloaded it again and rewrote it, and a second process sharing the
cache could read it half written. Now the right files are chosen at read time
(MNE 1.13 ``overrides``, ``eegdash.io.eeglab``, ``eegdash.io.bids``), and every
file of the dataset must stay byte-identical, with no file added.
"""

import json
import warnings
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, write_raw_bids
from scipy import io

from eegdash.dataset.base import EEGDashRaw
from eegdash.dataset.io import _load_raw_direct

SFREQ = 100.0
CH_NAMES = ["Fz", "Cz", "Pz"]
N_TIMES = 1000
DATA = np.random.default_rng(0).standard_normal((3, N_TIMES)).astype(np.float32)
ANNEX_EEG = "SHA256E-s9999--" + "ab" * 32 + ".eeg"


def _snapshot(root):
    return {
        p.relative_to(root): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _eegdashraw(root, relpath, task, cache_dir):
    record = {
        "dataset": root.name,
        "bids_relpath": relpath,
        "datatype": "eeg",
        "suffix": "eeg",
        "extension": Path(relpath).suffix,
        "storage": {"backend": "local", "base": str(root), "raw_key": relpath},
        "entities": {"subject": "01", "task": task},
        "entities_mne": {"subject": "01", "task": task},
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        return EEGDashRaw(record, cache_dir=str(cache_dir))


# BrainVision


def _write_brainvision(root):
    pytest.importorskip("pybv")
    info = mne.create_info(CH_NAMES, SFREQ, "eeg")
    raw = mne.io.RawArray(DATA.astype(float) * 1e-5, info)
    raw.set_annotations(mne.Annotations([2.0, 5.0], [0.0, 0.0], ["go", "stop"]))
    bp = BIDSPath(subject="01", task="rest", datatype="eeg", root=root)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_raw_bids(
            raw, bp, format="BrainVision", allow_preload=True, verbose="ERROR"
        )
    return root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.vhdr", raw


def _break_pointers(vhdr):
    """DataFile= an annex key (ds002158-style), no MarkerFile= at all."""
    lines = []
    for line in vhdr.read_text().splitlines():
        if line.startswith("DataFile="):
            line = f"DataFile={ANNEX_EEG}"
        if line.startswith("MarkerFile="):
            continue
        lines.append(line)
    vhdr.write_text("\n".join(lines) + "\n")


def test_brainvision_stale_pointers_read_without_rewrite(tmp_path):
    root = tmp_path / "ds-bv"
    vhdr, ref = _write_brainvision(root)
    _break_pointers(vhdr)
    before = _snapshot(root)

    raw = _eegdashraw(
        root, "sub-01/eeg/sub-01_task-rest_eeg.vhdr", "rest", tmp_path
    ).raw

    assert _snapshot(root) == before  # header untouched, no stub .vmrk
    np.testing.assert_allclose(raw.get_data(), ref.get_data(), rtol=1e-5, atol=1e-11)
    # the markers of the .vmrk the header no longer names (events.tsv agrees)
    assert {"go", "stop"} <= set(raw.annotations.description)


def test_brainvision_direct_fallback_gets_overrides(tmp_path):
    """The direct reader (every _load_raw_direct fallback) finds the same files."""
    vhdr, ref = _write_brainvision(tmp_path / "ds-bv")
    markers = mne.io.read_raw_brainvision(vhdr, verbose="ERROR").annotations
    assert len(markers) >= 2  # Stimulus/S  1, Stimulus/S  2
    _break_pointers(vhdr)
    before = _snapshot(tmp_path / "ds-bv")
    raw = _load_raw_direct(vhdr)
    np.testing.assert_allclose(raw.get_data(), ref.get_data(), rtol=1e-5, atol=1e-11)
    # the .vmrk markers, although the header has no MarkerFile= any more
    assert list(raw.annotations.description) == list(markers.description)
    np.testing.assert_array_equal(raw.annotations.onset, markers.onset)
    assert _snapshot(tmp_path / "ds-bv") == before


# EEGLAB


def _write_set(path, data, dataname):
    data.T.astype("<f4").tofile(path.with_suffix(".fdt"))
    chanlocs = np.array([(n.encode(),) for n in CH_NAMES], dtype=[("labels", "S10")])
    eeg = dict(
        nbchan=len(CH_NAMES),
        pnts=data.shape[1],
        trials=1,
        srate=SFREQ,
        xmin=0.0,
        data=dataname,
        chanlocs=chanlocs,
        event=[],
        epoch=[],
    )
    io.savemat(path, {"EEG": eeg}, appendmat=False, oned_as="row")


def _write_eeglab_bids(root):
    """sub-01 with a wake and a sleep recording; wake's EEG.data names sleep's .fdt."""
    eeg = root / "sub-01" / "eeg"
    eeg.mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.9.0"})
    )
    for task, data in (("sleep", DATA + 1000), ("wake", DATA)):
        stem = f"sub-01_task-{task}_eeg"
        _write_set(eeg / f"{stem}.set", data, "sub-01_task-sleep_eeg.fdt")
        (eeg / f"{stem}.json").write_text(
            json.dumps(
                {"TaskName": task, "SamplingFrequency": SFREQ, "EEGReference": "Cz"}
            )
        )
        (eeg / f"{stem.replace('_eeg', '_channels')}.tsv").write_text(
            "name\ttype\tunits\n" + "".join(f"{n}\tEEG\tuV\n" for n in CH_NAMES)
        )
    return eeg


def test_eeglab_own_fdt_read_without_rewrite(tmp_path):
    """on004348: the wake recording reads its own samples, the .set stays as is."""
    root = tmp_path / "ds-set"
    _write_eeglab_bids(root)
    before = _snapshot(root)

    raw = _eegdashraw(root, "sub-01/eeg/sub-01_task-wake_eeg.set", "wake", tmp_path).raw

    np.testing.assert_allclose(raw.get_data(), DATA * 1e-6, rtol=1e-6)
    assert _snapshot(root) == before


def test_eeglab_truncated_fdt_read_without_rewrite(tmp_path):
    """ds006554: the complete samples are read and the loss reported; the .set stays."""
    root = tmp_path / "ds-set"
    eeg = _write_eeglab_bids(root)
    fdt = eeg / "sub-01_task-sleep_eeg.fdt"
    n_complete = 800
    fdt.write_bytes(fdt.read_bytes()[: (n_complete * len(CH_NAMES) + 2) * 4 + 1])
    before = _snapshot(root)

    with pytest.warns(RuntimeWarning, match="runs past the end of the file"):
        raw = _eegdashraw(
            root, "sub-01/eeg/sub-01_task-sleep_eeg.set", "sleep", tmp_path
        ).raw

    assert raw.n_times == n_complete
    np.testing.assert_allclose(
        raw.get_data(), (DATA[:, :n_complete] + 1000) * 1e-6, rtol=1e-6
    )
    assert f"{n_complete} of {N_TIMES} samples" in raw.info["description"]
    assert _snapshot(root) == before

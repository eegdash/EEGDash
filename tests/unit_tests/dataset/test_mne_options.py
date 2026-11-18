"""BIDS sidecar quirks that MNE-BIDS 0.20 reads itself, with on_ch_mismatch="rename".

EEGDash used to rewrite these files in the download cache before reading them.
The rewritten copy no longer matched the S3 size, so every new session downloaded
it again and repaired it again, and threads or jobs sharing a cache could read a
half-written events.tsv (on004841: "the number of columns changed from 4 to 1",
on005342: KeyError 'onset'). MNE-BIDS 0.20 reads them as they are, so the files
must stay byte-identical.
"""

import os
import warnings
from datetime import UTC, datetime
from unittest.mock import patch

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, write_raw_bids

from eegdash.dataset.base import EEGDashRaw

pytest.importorskip("pybv")

SFREQ = 100.0
MEAS_DATE = datetime(2020, 1, 1, 10, 0, 0, tzinfo=UTC)
RELPATH = "sub-01/eeg/sub-01_task-rest_eeg.vhdr"


def _write_bids(root):
    info = mne.create_info(["Fz", "Cz", "Pz"], SFREQ, "eeg")
    data = np.random.default_rng(0).standard_normal((3, int(10 * SFREQ))) * 1e-5
    raw = mne.io.RawArray(data, info)
    raw.set_meas_date(MEAS_DATE)
    bp = BIDSPath(subject="01", task="rest", datatype="eeg", root=root)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_raw_bids(
            raw, bp, format="BrainVision", allow_preload=True, verbose="ERROR"
        )
    return root / "sub-01" / "eeg"


def _eegdashraw(root, cache_dir):
    record = {
        "dataset": root.name,
        "bids_relpath": RELPATH,
        "datatype": "eeg",
        "suffix": "eeg",
        "extension": ".vhdr",
        "storage": {"backend": "local", "base": str(root), "raw_key": RELPATH},
        "entities": {"subject": "01", "task": "rest"},
        "entities_mne": {"subject": "01", "task": "rest"},
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        return EEGDashRaw(record, cache_dir=str(cache_dir))


def _tsv_bytes(root):
    return {p: p.read_bytes() for p in sorted(root.rglob("*.tsv"))}


def test_sidecar_quirks_are_read_without_rewriting_them(tmp_path):
    root = tmp_path / "ds-quirks"
    eeg = _write_bids(root)
    # Decimal comma (on005342, nm000376), padded n/a (on004381), n/a and NaN
    # onsets (on004841), all in one events.tsv.
    (eeg / "sub-01_task-rest_events.tsv").write_text(
        "onset\tduration\ttrial_type\n"
        "1,5\tn/a   \tgo  \n"
        "n/a\t0\tlost\n"
        "NaN\t0\tlost\n"
        "3.0\t0,25\tstop\n"
    )
    # An acq_time with 60 seconds (nm000364): MNE-BIDS keeps the file's date.
    scans = root / "sub-01" / "sub-01_scans.tsv"
    scans.write_text(
        "filename\tacq_time\neeg/sub-01_task-rest_eeg.vhdr\t2020-01-01T10:47:60\n"
    )
    # The same name twice in channels.tsv: on_ch_mismatch="rename" numbers them.
    channels = eeg / "sub-01_task-rest_channels.tsv"
    lines = channels.read_text().splitlines()
    lines[2] = "Fz" + lines[2][2:]  # Cz -> Fz
    channels.write_text("\n".join(lines) + "\n")
    before = _tsv_bytes(root)

    raw = _eegdashraw(root, tmp_path).raw

    assert _tsv_bytes(root) == before  # nothing in the deposit copy is rewritten
    np.testing.assert_allclose(raw.annotations.onset, [1.5, 3.0])
    np.testing.assert_allclose(raw.annotations.duration, [0.0, 0.25])
    assert list(raw.annotations.description) == ["go", "stop"]
    # meas_date is the recording's own (MNE-BIDS leaves it), not a made-up 10:48:00
    vhdr = eeg / "sub-01_task-rest_eeg.vhdr"
    own = mne.io.read_raw_brainvision(vhdr, verbose="ERROR").info["meas_date"]
    assert raw.info["meas_date"] == own
    assert raw.ch_names == ["Fz-0", "Fz-1", "Pz"]


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root ignores modes"
)
def test_read_only_dataset_reads_without_a_lock_patch(tmp_path):
    """MNE-BIDS 0.20 skips a lock it cannot take; no filelock patch is needed."""
    root = tmp_path / "ds-ro"
    _write_bids(root)
    dirs = [root, *[p for p in root.rglob("*") if p.is_dir()]]
    for d in dirs:
        d.chmod(0o555)
    try:
        raw = _eegdashraw(root, tmp_path).raw
        assert raw.n_times == int(10 * SFREQ)
        assert not list(root.rglob("*.lock"))
    finally:
        for d in dirs:
            d.chmod(0o755)

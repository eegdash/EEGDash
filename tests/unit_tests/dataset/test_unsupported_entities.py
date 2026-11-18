"""A parsed entity unsupported by BIDSPath must not select another recording."""

from unittest.mock import patch

import mne
import numpy as np

from eegdash.dataset.base import EEGDashRaw


def test_part_entity_reads_exact_file_with_warning(tmp_path, caplog):
    """on002312: part-02 is parsed, but MNE-BIDS 0.20 cannot represent it."""
    folder = tmp_path / "sub-01" / "meg"
    folder.mkdir(parents=True)
    name = "sub-01_task-rest_part-02_meg.fif"
    info = mne.create_info(["MEG001"], 100.0, "mag")
    for filename, value in ((name, 2e-12), (name.replace("_part-02", ""), 1e-12)):
        mne.io.RawArray(np.full((1, 100), value), info).save(folder / filename)
    record = {
        "dataset": "test",
        "bids_relpath": f"sub-01/meg/{name}",
        "datatype": "meg",
        "suffix": "meg",
        "extension": ".fif",
        "entities_mne": {"subject": "01", "task": "rest"},
        "storage": {
            "backend": "local",
            "base": str(tmp_path),
            "raw_key": f"sub-01/meg/{name}",
        },
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=tmp_path)
    with patch.object(ds, "_read_raw_bids", side_effect=AssertionError("wrong file")):
        raw = ds._load_raw()
    np.testing.assert_allclose(raw.get_data(), 2e-12)
    assert "part" in caplog.text
    assert "sidecars" in caplog.text
    assert name in caplog.text

from pathlib import Path

from eegdash.bids_eeg_metadata import load_eeg_attrs_from_bids_file


class _DummyBIDSDataset:
    def __init__(self, root: Path, bids_file: Path):
        self.bidsdir = root
        self.dataset = root.name
        self.files = [str(bids_file)]

    def _get_relative_bidspath(self, filename: str | Path) -> str:
        return str(Path(filename).relative_to(self.bidsdir.parent))

    get_relative_bidspath = _get_relative_bidspath

    def subject_participant_tsv(self, bids_file):  # noqa: ARG002 - unused in tests
        return None

    def eeg_json(self, bids_file):  # noqa: ARG002 - unused in tests
        return None

    def get_bids_metadata_files(self, bids_file, extension):  # noqa: ARG002 - unused
        return []

    def get_bids_file_attribute(self, attribute: str, bids_file: str):
        defaults = {
            "subject": "01",
            "session": "01",
            "task": "test",
            "run": "01",
            "modality": "eeg",
            "sfreq": 100,
            "nchans": 1,
            "ntimes": 1,
        }
        return defaults.get(attribute)


# --- tests -----------------------------------------------------------------

def test_sidecar_dependencies_filtered(tmp_path: Path):
    root = tmp_path / "ds000001"
    eeg_dir = root / "sub-01" / "ses-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    base = "sub-01_ses-01_task-test_run-01_eeg"
    vhdr = eeg_dir / f"{base}.vhdr"
    vhdr.write_text("")
    eeg = eeg_dir / f"{base}.eeg"
    eeg.write_text("")
    vmrk = eeg_dir / f"{base}.vmrk"
    vmrk.write_text("")

    bids_ds = _DummyBIDSDataset(root, vhdr)
    attrs = load_eeg_attrs_from_bids_file(bids_ds, str(vhdr))

    expected = {
        bids_ds._get_relative_bidspath(eeg),
        bids_ds._get_relative_bidspath(vmrk),
    }
    assert expected.issubset(set(attrs["bidsdependencies"]))
    assert not any(dep.endswith(ext) for ext in [".dat", ".raw", ".vhdr"] for dep in attrs["bidsdependencies"])

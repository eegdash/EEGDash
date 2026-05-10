from pathlib import Path

from eegdash.paths import get_default_cache_dir


def is_bids_dataset_available() -> tuple[bool, str]:
    """Check if the shared mini BIDS dataset is available and valid."""
    cache_dir = Path(get_default_cache_dir())
    path = cache_dir / "ds005509-bdf-mini"

    if not path.exists():
        return False, f"BIDS dataset not found at {path}"

    if not (path / "dataset_description.json").exists():
        return False, "Not a valid BIDS dataset (missing dataset_description.json)"

    bdf_files = list(path.rglob("*.bdf"))
    edf_files = list(path.rglob("*.edf"))
    if not bdf_files and not edf_files:
        return False, "No BDF/EDF data files found in dataset"

    return True, ""

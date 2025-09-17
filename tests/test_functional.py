from pathlib import Path

import pytest

from eegdash.dataset.dataset import EEGDashDataset


@pytest.fixture(scope="module")
def cache_dir():
    """Provide a shared cache directory for tests that need to cache datasets."""
    from pathlib import Path

    from eegdash.paths import get_default_cache_dir

    cache_dir = Path(get_default_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def test_progress_bar_output(capsys, cache_dir: Path):
    eegdash_dataset = EEGDashDataset(
        query={
            "dataset": "ds005514",
            "task": "RestingState",
            "subject": "NDARDB033FW5",
        },
        cache_dir=cache_dir,
    )
    _ = eegdash_dataset.datasets[0].raw

    out = capsys.readouterr()
    # tqdm uses carriage returns; just assert a stable fragment:
    assert "Downloading" in out.err
    assert out.err  # non-empty

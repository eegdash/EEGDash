from pathlib import Path

from eegdash.dataset.dataset import EEGDashDataset


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

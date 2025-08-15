import shutil
from pathlib import Path

from eegdash.dataset import EEGDashDataset



def test_progress_bar_output(capsys, tmp_path):
    
    temporary = (Path.home() / "mne_data" / "temp").resolve()
    
    if temporary.exists():
        shutil.rmtree(temporary)
    
    temporary.mkdir(parents=True, exist_ok=True)

    eegdash_dataset = EEGDashDataset(
            query={
                "dataset": "ds005514",
                "task": "RestingState",
                "subject": "NDARDB033FW5",
            },
            cache_dir=temporary,
        )
    raw = eegdash_dataset.datasets[0].raw

    out = capsys.readouterr()
    # tqdm uses carriage returns; just assert a stable fragment:
    assert "Downloading" in out.err
    assert out.err  # non-empty

# %% [markdown]
""".. _scratch-tutorial:

Scratch Tutorial
================

A minimal exploration script using EEGDash utilities.
"""
# raw = mne.io.read_raw_eeglab('./.eegdash_cache/sub-NDARDB033FW5_task-RestingState_eeg.set', preload=True)
# for preprocessor in preprocessors:
#     raw = preprocessor.apply(raw)

from pathlib import Path
import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(Path.cwd()))
(Path(os.environ["_MNE_FAKE_HOME_DIR"]) / ".mne").mkdir(exist_ok=True)

from eegdash import EEGDash, EEGDashDataset
from mne_bids import BIDSPath, read_raw_bids
import mne

CACHE_DIR = Path(os.getenv("EEGDASH_CACHE_DIR", Path.cwd() / "eegdash_cache")).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

eegdash = EEGDash()
record_limit = int(os.getenv("EEGDASH_RECORD_LIMIT", "5"))
records = eegdash.find({"dataset": "ds002718"}, limit=record_limit)
if not records:
    raise RuntimeError("No records found for dataset ds002718.")

record = records[0]
print("Record:", record)

dataset = EEGDashDataset(cache_dir=CACHE_DIR, records=records)
raw = None
loaded_record = None
for base_ds in dataset.datasets:
    try:
        raw = base_ds.raw
        loaded_record = base_ds.record
        break
    except RuntimeError as exc:
        raw_path = CACHE_DIR / base_ds.record["bidspath"]
        if raw_path.suffix == ".set":
            try:
                raw = mne.io.read_raw_eeglab(raw_path, preload=False, verbose="ERROR")
                loaded_record = base_ds.record
                break
            except Exception as fallback_exc:
                print(f"Skipping record due to read error: {fallback_exc}")
        else:
            print(f"Skipping record due to read error: {exc}")

if raw is None or loaded_record is None:
    raise RuntimeError("Unable to load any recordings from the dataset.")

events, event_id = mne.events_from_annotations(raw)
print("Event IDs:", event_id)

bidspath = BIDSPath(
    root=CACHE_DIR / loaded_record["dataset"],
    datatype="eeg",
    task=loaded_record.get("task"),
    subject=loaded_record.get("subject"),
    suffix="eeg",
)
try:
    raw_bids = read_raw_bids(bidspath, verbose=False)
    print("BIDS raw duration (s):", raw_bids.times[-1])
except RuntimeError as exc:
    print(f"BIDS read failed (missing coordsystem): {exc}")

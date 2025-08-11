"""Tutorial Scratch."""
# raw = mne.io.read_raw_eeglab('./.eegdash_cache/sub-NDARDB033FW5_task-RestingState_eeg.set', preload=True)
# for preprocessor in preprocessors:
#     raw = preprocessor.apply(raw)

# %%
# %load_ext autoreload
# %autoreload 2
from eegdash import EEGDash, EEGDashDataset

eegdashdata = EEGDash()

# %%
import mne_bids
from mne_bids import (
    BIDSPath,
)

# %%
records = eegdashdata.find({"dataset": "ds002718", "subject": "012"})
record = records[0]

print(record)

# Downloading with eegdash Dataset
eegdashdata = EEGDashDataset(
    query={"dataset": "ds002718", "subject": "012"}, cache_dir=".eegdash_cache"
)

eeg = eegdashdata.load_data()

# %%
bidspath = BIDSPath(
    root=".eegdash_cache/ds002718",
    datatype="eeg",
    task=record["task"],
    subject=record["subject"],
    suffix="eeg",
)

# %%
EEG = mne_bids.read_raw_bids(bidspath)

# %%
from braindecode.datasets import BaseDataset

BaseDataset(None)

# %%
EEG.annotations

# %%
from eegdash.data_utils import BIDSDataset

bids = BIDSDataset("/mnt/nemar/openneuro/ds002718", "ds002718")
bids.files

# %%
from eegdash import EEGDash

eegdashObj = EEGDash()
record = eegdashObj.load_eeg_attrs_from_bids_file(
    bids,
    "/mnt/nemar/openneuro/ds002718/sub-018/eeg/sub-018_task-FaceRecognition_eeg.set",
)

# %%
from eegdash.data_utils import EEGDashBaseDataset

eegdashDataset = EEGDashBaseDataset(record, ".eegdash_cache")

# %%
len(eegdashDataset.raw)

# %%
import mne

raw = eegdashDataset.raw
events, event_id = mne.events_from_annotations(raw)

print(event_id)

# %%
ds = EEGDashDataset(query={"dataset": "ds005507"}, target_name="sex")

# %%
ds.datasets[2][0]

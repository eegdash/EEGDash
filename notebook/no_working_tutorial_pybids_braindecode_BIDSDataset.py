# %% [markdown]
""".. _pybids-bidsdataset-demo:

Exploring Braindecode's BIDSDataset
===================================

Tests showing BIDSDataset not able to handle example EEGLAB dataset and slower than pybids
"""

# %%
from pathlib import Path
import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(Path.cwd()))
(Path(os.environ["_MNE_FAKE_HOME_DIR"]) / ".mne").mkdir(exist_ok=True)

from bids import BIDSLayout
from braindecode.datasets import BIDSDataset
from eegdash import EEGDash, EEGDashDataset

CACHE_DIR = Path(os.getenv("EEGDASH_CACHE_DIR", Path.cwd() / "eegdash_cache")).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATASET_ID = os.getenv("EEGDASH_DATASET_ID", "ds002718")

eegdash = EEGDash()
records = eegdash.find({"dataset": DATASET_ID}, limit=3)
if not records:
    raise RuntimeError(f"No records found for dataset {DATASET_ID}.")

dataset = EEGDashDataset(cache_dir=CACHE_DIR, records=records)
try:
    _ = dataset.datasets[0].raw
except RuntimeError as exc:
    print(f"Raw read failed (likely missing coordsystem.json): {exc}")
root = CACHE_DIR / DATASET_ID

bids = BIDSDataset(root=str(root), preload=False)
# Can't import regular EEGLAB dataset

# %% [markdown]
# Tests showing pybids utilities as well as limitations
#
# - Recording files can be retrieved fast
# - File path can be mapped to BIDS file using simple additional parsing
# - Needed info such as duration and channel count can be retrieved easily
# - Not all file level metadata files can be retrieved even though they exist
# - Top level json associated with a file can't be retrieved from file level


# %%
def get_recordings(layout: BIDSLayout):
    extensions = {
        ".set": [".set", ".fdt"],  # eeglab
        ".edf": [".edf"],  # european
        ".vhdr": [".eeg", ".vhdr", ".vmrk", ".dat", ".raw"],  # brainvision
        ".bdf": [".bdf"],  # biosemi
    }
    files = []
    for ext, exts in extensions.items():
        files = layout.get(extension=ext, return_type="filename")
        if files:
            break
    return files


print(get_recordings(BIDSLayout(str(root))))

# %%
layout = BIDSLayout(str(root))
# get file from path
recordings = get_recordings(layout)
if not recordings:
    raise RuntimeError(f"No EEG recordings found under {root}.")
example_file = recordings[0]
entities = layout.parse_file_entities(example_file)
bidsfile = layout.get(**entities)[0]
print(bidsfile)

# %%
import pprint

# get general info of a recording
pprint.pprint(bidsfile.get_entities(metadata="all"))

# %%
# get associations doesn't give us all desired bids dependencies
bidsfile.get_associations()

# %%
# top level events.json can't be retrieved from a file level
file_entities = bidsfile.get_entities()
# remove 'datatype'
file_entities.pop("datatype")

file_entities["suffix"] = "events"
file_entities["extension"] = ".json"
print(file_entities)
print(layout.get(**file_entities))
print(layout.get(suffix="events", extension=".json"))

# not all file level metadata files can be retrieved even though they exist
file_entities["suffix"] = "events"
file_entities["extension"] = "tsv"
print(file_entities)
print(layout.get(**file_entities))

file_entities["suffix"] = "electrodes"
file_entities["extension"] = "tsv"
print(file_entities)
print(layout.get(**file_entities))

file_entities["suffix"] = "coordsystem"
file_entities["extension"] = "json"
print(file_entities)
print(layout.get(**file_entities))

# %%

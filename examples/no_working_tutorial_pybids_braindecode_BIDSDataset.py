"""Tests showing BIDSDataset not able to handle example EEGLAB dataset and slower than pybids"""

# %%
from braindecode.datasets import BIDSDataset
from bids import BIDSLayout

# %%
bids = BIDSDataset(root="/mnt/nemar/openneuro/ds003645", preload=False)
# Can't import regular EEGLAB dataset

# %%
# %time bids = BIDSDataset(root='/mnt/nemar/openneuro/ds002718', preload=False)

# %%
# %time layout = BIDSLayout('/mnt/nemar/openneuro/ds003645')
# %time layout = BIDSLayout('/mnt/nemar/openneuro/ds002718')

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


print(get_recordings(BIDSLayout("/mnt/nemar/openneuro/ds002718")))
print(get_recordings(BIDSLayout("/mnt/nemar/openneuro/ds004770")))
print(get_recordings(BIDSLayout("/mnt/nemar/openneuro/ds004561")))

# %%
layout = BIDSLayout("/mnt/nemar/openneuro/ds002718")
# get file from path
entities = layout.parse_file_entities(
    "/mnt/nemar/openneuro/ds002718/sub-002/eeg/sub-002_task-FaceRecognition_eeg.set"
)
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

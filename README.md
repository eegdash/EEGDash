# EEG-Dash

[![PyPI version](https://img.shields.io/pypi/v/eegdash)](https://pypi.org/project/eegdash/)
[![Docs](https://img.shields.io/badge/docs-stable-brightgreen.svg)](https://sccn.github.io/eegdash)

[![License: GPL-2.0-or-later](https://img.shields.io/badge/License-GPL--2.0--or--later-blue.svg)](LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/eegdash.svg)](https://pypi.org/project/eegdash/)
[![Downloads](https://pepy.tech/badge/eegdash)](https://pepy.tech/project/eegdash)
<!-- [![Coverage](https://img.shields.io/codecov/c/github/sccn/eegdash)](https://codecov.io/gh/sccn/eegdash) -->

To leverage recent and ongoing advancements in large-scale computational methods and to ensure the preservation of scientific data generated from publicly funded research, the EEG-DaSh data archive will create a data-sharing resource for MEEG (EEG, MEG) data contributed by collaborators for machine learning (ML) and deep learning (DL) applications. 

## Data source

The data in EEG-DaSh originates from a collaboration involving 25 laboratories, encompassing 27,053 participants. This extensive collection includes MEEG data, which is a combination of EEG and MEG signals. The data is sourced from various studies conducted by these labs, involving both healthy subjects and clinical populations with conditions such as ADHD, depression, schizophrenia, dementia, autism, and psychosis. Additionally, data spans different mental states like sleep, meditation, and cognitive tasks. In addition, EEG-DaSh will incorporate a subset of the data converted from NEMAR, which includes 330 MEEG BIDS-formatted datasets, further expanding the archive with well-curated, standardized neuroelectromagnetic data.

## Featured data

The following HBN datasets are currently featured on EEGDash. Documentation about these datasets is available [here](https://neuromechanist.github.io/data/hbn/).

| DatasetID | Participants | Files | Sessions | Population | Channels | Is 10-20? | Modality | Size |
|---|---|---|---|---|---|---|---|---|
| [ds005505](https://nemar.org/dataexplorer/detail?dataset_id=ds005505) | 136 | 5393 | 1 | Healthy | 129 | other | Visual | 103 GB |
| [ds005506](https://nemar.org/dataexplorer/detail?dataset_id=ds005506) | 150 | 5645 | 1 | Healthy | 129 | other | Visual | 112 GB |
| [ds005507](https://nemar.org/dataexplorer/detail?dataset_id=ds005507) | 184 | 7273 | 1 | Healthy | 129 | other | Visual | 140 GB |
| [ds005508](https://nemar.org/dataexplorer/detail?dataset_id=ds005508) | 324 | 13393 | 1 | Healthy | 129 | other | Visual | 230 GB |
| [ds005510](https://nemar.org/dataexplorer/detail?dataset_id=ds005510) | 135 | 4933 | 1 | Healthy | 129 | other | Visual | 91 GB |
| [ds005512](https://nemar.org/dataexplorer/detail?dataset_id=ds005512) | 257 | 9305 | 1 | Healthy | 129 | other | Visual | 157 GB |
| [ds005514](https://nemar.org/dataexplorer/detail?dataset_id=ds005514) | 295 | 11565 | 1 | Healthy | 129 | other | Visual | 185 GB |

A total of [246 other datasets](datasets.md) are also available through EEGDash. 

## Data format

EEGDash queries return a **Pytorch Dataset** formatted to facilitate machine learning (ML) and deep learning (DL) applications. PyTorch Datasets are the best format for EEGDash queries because they provide an efficient, scalable, and flexible structure for machine learning (ML) and deep learning (DL) applications. They allow seamless integration with PyTorch’s DataLoader, enabling efficient batching, shuffling, and parallel data loading, which is essential for training deep learning models on large EEG datasets.

## Data preprocessing

EEGDash datasets are processed using the popular [BrainDecode](https://braindecode.org/stable/index.html) library. In fact, EEGDash datasets are BrainDecode datasets, which are themselves PyTorch datasets. This means that any preprocessing possible on BrainDecode datasets is also possible on EEGDash datasets. Refer to [BrainDecode](https://braindecode.org/stable/index.html) tutorials for guidance on preprocessing EEG data.

## EEG-Dash usage

### Install
Use your preferred Python environment manager with Python > 3.9 to install the package.
* To install the eegdash package, use the following command: `pip install eegdash`
* To verify the installation, start a Python session and type: `from eegdash import EEGDash`

### Data access

To use the data from a single subject, enter:

```python
from eegdash import EEGDashDataset

ds_NDARDB033FW5 = EEGDashDataset(
    {"dataset": "ds005514", "task":
     "RestingState", "subject": "NDARDB033FW5"}, 
     cache_dir="."
)
```

This will search and download the metadata for the task **RestingState** for subject **NDARDB033FW5** in BIDS dataset **ds005514**. The actual data will not be downloaded at this stage. Following standard practice, data is only downloaded once it is processed. The **ds_NDARDB033FW5** object is a fully functional BrainDecode dataset, which is itself a PyTorch dataset. This [tutorial](https://github.com/sccn/EEGDash/blob/develop/notebooks/tutorial_eoec.ipynb) shows how to preprocess the EEG data, extracting portions of the data containing eyes-open and eyes-closed segments, then perform eyes-open vs. eyes-closed classification using a (shallow) deep-learning model. 

To use the data from multiple subjects, enter:

```python
from eegdash import EEGDashDataset

ds_ds005505rest = EEGDashDataset(
    {"dataset": "ds005505", "task": "RestingState"}, target_name="sex", cache_dir=".
)
```

This will search and download the metadata for the task 'RestingState' for all subjects in BIDS dataset 'ds005505' (a total of 136). As above, the actual data will not be downloaded at this stage so this command is quick to execute. Also, the target class for each subject is assigned using the target_name parameter. This means that this object is ready to be directly fed to a deep learning model, although the [tutorial script](https://github.com/sccn/EEGDash/blob/develop/notebooks/tutorial_sex_classification.ipynb) performs minimal processing on it, prior to training a deep-learning model. Because 14 gigabytes of data are downloaded, this tutorial takes about 10 minutes to execute.

### Automatic caching

EEGDash automatically caches the downloaded data in the .eegdash_cache folder of the current directory from which the script is called. This means that if you run the tutorial [scripts](https://github.com/sccn/EEGDash/tree/develop/notebooks), the data will only be downloaded the first time the script is executed.

## Education -- Coming soon...

We organize workshops and educational events to foster cross-cultural education and student training, offering both online and in-person opportunities in collaboration with US and Israeli partners. Events for 2025 will be announced via the EEGLABNEWS mailing list. Be sure to [subscribe](https://sccn.ucsd.edu/mailman/listinfo/eeglabnews).

## About EEG-DaSh

EEG-DaSh is a collaborative initiative between the United States and Israel, supported by the National Science Foundation (NSF). The partnership brings together experts from the Swartz Center for Computational Neuroscience (SCCN) at the University of California San Diego (UCSD) and Ben-Gurion University (BGU) in Israel. 

![Screenshot 2024-10-03 at 09 14 06](https://github.com/user-attachments/assets/327639d3-c3b4-46b1-9335-37803209b0d3)





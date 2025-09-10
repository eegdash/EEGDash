# EEG-Dash

[![PyPI version](https://img.shields.io/pypi/v/eegdash)](https://pypi.org/project/eegdash/)
[![Docs](https://img.shields.io/badge/docs-stable-brightgreen.svg)](https://sccn.github.io/eegdash)
[![License: GPL-2.0-or-later](https://img.shields.io/badge/License-GPL--2.0--or--later-blue.svg)](LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/eegdash.svg)](https://pypi.org/project/eegdash/)
[![Downloads](https://pepy.tech/badge/eegdash)](https://pepy.tech/project/eegdash)

EEG-Dash is a Python package designed to simplify access to and analysis of large-scale MEEG (EEG, MEG) datasets. It provides a high-level interface for querying and retrieving data from the EEG-DaSh data archive, as well as tools for feature extraction and analysis.

## Core Concepts

The `eegdash` package is built around a few core concepts:

-   **`EEGDash`**: The main entry point for interacting with the EEG-DaSh database. It provides methods for finding, adding, and updating metadata records.
-   **`EEGDashDataset`**: A PyTorch-compatible dataset class that represents a collection of EEG recordings. It allows for easy integration with machine learning and deep learning pipelines.
-   **Feature Extractors**: A set of classes and functions for extracting features from EEG signals. These are designed to be modular and extensible, allowing you to easily create and apply custom feature extraction pipelines.

## Getting Started

### Installation

To get started with `eegdash`, you need to have Python > 3.9 installed. You can install the package using `pip`:

```bash
pip install eegdash
```

To verify the installation, open a Python interpreter and run:

```python
from eegdash import EEGDash
```

### Basic Usage

Here's a simple example of how to query the database and load a dataset:

```python
from eegdash import EEGDashDataset

# Find all recordings for a specific subject and task
ds = EEGDashDataset(
    dataset="ds005505",
    subject="NDARCA153NKE",
    task="RestingState",
    cache_dir="."
)

# The `ds` object is a braindecode dataset, which is a PyTorch dataset.
# You can now use it with a PyTorch DataLoader:
from torch.utils.data import DataLoader

loader = DataLoader(ds, batch_size=32, shuffle=True)
```

## Data Access

### Querying the Database

You can query the database using keyword arguments to `EEGDashDataset`, or by passing a raw MongoDB query.

**Keyword Arguments:**

```python
# Find all recordings for a list of subjects and a specific task
subjects = ["NDARCA153NKE", "NDARXT792GY8"]
ds = EEGDashDataset(
    dataset="ds005505",
    subject=subjects,
    task="RestingState",
    cache_dir="."
)
```

**Raw MongoDB Query:**

```python
# Use a raw MongoDB query for more advanced filtering
raw_query = {
    "dataset": "ds005505",
    "subject": {"$in": ["NDARCA153NKE", "NDARXT792GY8"]},
    "task": "RestingState"
}
ds = EEGDashDataset(query=raw_query, cache_dir=".")
```

### Automatic Caching

By default, EEGDash caches downloaded data under a single, consistent folder:

-   If `EEGDASH_CACHE_DIR` is set in your environment, that path is used.
-   Else, if MNE’s `MNE_DATA` config is set, that path is used to align with other EEG tooling.
-   Otherwise, `.eegdash_cache` in the current working directory is used.

This means that data is only downloaded the first time it is accessed and is reused in subsequent runs.

## Feature Extraction

`eegdash` provides a powerful and flexible feature extraction pipeline. You can use the built-in feature extractors or create your own.

Here's an example of how to extract features from a dataset:

```python
from eegdash.features import extract_features
from eegdash.features.feature_bank import signal_mean, signal_std

# Define the features to extract
features = {
    "mean": signal_mean,
    "std": signal_std,
}

# Extract the features
feature_ds = extract_features(ds, features)

# The `feature_ds` object is a FeaturesConcatDataset, which can be converted
# to a pandas DataFrame for further analysis.
df = feature_ds.to_dataframe()
```

## Contributing

We welcome contributions to `eegdash`! If you'd like to contribute, please follow these steps:

1.  Fork the repository on GitHub.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and write tests for them.
4.  Run the tests to ensure everything is working correctly.
5.  Submit a pull request with a clear description of your changes.

## About EEG-DaSh

EEG-DaSh is a collaborative initiative between the United States and Israel, supported by the National Science Foundation (NSF). The partnership brings together experts from the Swartz Center for Computational Neuroscience (SCCN) at the University of California San Diego (UCSD) and Ben-Gurion University (BGU) in Israel.

![Screenshot 2024-10-03 at 09 14 06](https://github.com/user-attachments/assets/327639d3-c3b4-46b1-9335-37803209b0d3)

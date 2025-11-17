# EEG-Dash

[![PyPI version](https://img.shields.io/pypi/v/eegdash)](https://pypi.org/project/eegdash/)
[![Docs](https://img.shields.io/badge/docs-stable-brightgreen.svg)](https://sccn.github.io/eegdash)

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/eegdash.svg)](https://pypi.org/project/eegdash/)
[![Downloads](https://pepy.tech/badge/eegdash)](https://pepy.tech/project/eegdash)
<!-- [![Coverage](https://img.shields.io/codecov/c/github/sccn/eegdash)](https://codecov.io/gh/sccn/eegdash) -->

To leverage recent and ongoing advancements in large-scale computational methods and to ensure the preservation of scientific data generated from publicly funded research, the EEG-DaSh data archive will create a data-sharing resource for MEEG (EEG, MEG) data contributed by collaborators for machine learning (ML) and deep learning (DL) applications. 

## Data source

The data in EEG-DaSh originates from a collaboration involving 25 laboratories, encompassing 27,053 participants. This extensive collection includes MEEG data, which is a combination of EEG and MEG signals. The data is sourced from various studies conducted by these labs, involving both healthy subjects and clinical populations with conditions such as ADHD, depression, schizophrenia, dementia, autism, and psychosis. Additionally, data spans different mental states like sleep, meditation, and cognitive tasks. In addition, EEG-DaSh will incorporate a subset of the data converted from NEMAR, which includes 330 MEEG BIDS-formatted datasets, further expanding the archive with well-curated, standardized neuroelectromagnetic data.

## Data format

EEGDash queries return a **Pytorch Dataset** formatted to facilitate machine learning (ML) and deep learning (DL) applications. PyTorch Datasets are the best format for EEGDash queries because they provide an efficient, scalable, and flexible structure for machine learning (ML) and deep learning (DL) applications. They allow seamless integration with PyTorch’s DataLoader, enabling efficient batching, shuffling, and parallel data loading, which is essential for training deep learning models on large EEG datasets.

## Data preprocessing

EEGDash datasets are processed using the popular [braindecode](https://braindecode.org/stable/index.html) library. In fact, EEGDash datasets are braindecode datasets, which are themselves PyTorch datasets. This means that any preprocessing possible on braindecode datasets is also possible on EEGDash datasets. Refer to [braindecode](https://braindecode.org/stable/index.html) tutorials for guidance on preprocessing EEG data.

## EEG-Dash usage

### Install
Use your preferred Python environment manager with Python > 3.10 to install the package.
* To install the eegdash package, use the following command: `pip install eegdash`
* To verify the installation, start a Python session and type: `from eegdash import EEGDash`

Please check our tutorial webpages to explore what you can do with [eegdash](https://eegdash.org/)! 

## Education -- Coming soon...

We organize workshops and educational events to foster cross-cultural education and student training, offering both online and in-person opportunities in collaboration with US and Israeli partners. Events for 2025 will be announced via the EEGLABNEWS mailing list. Be sure to [subscribe](https://sccn.ucsd.edu/mailman/listinfo/eeglabnews).

## About EEG-DaSh

EEG-DaSh is a collaborative initiative between the United States and Israel, supported by the National Science Foundation (NSF). The partnership brings together experts from the Swartz Center for Computational Neuroscience (SCCN) at the University of California San Diego (UCSD) and Ben-Gurion University (BGU) in Israel. 

![Screenshot 2024-10-03 at 09 14 06](https://github.com/user-attachments/assets/327639d3-c3b4-46b1-9335-37803209b0d3)




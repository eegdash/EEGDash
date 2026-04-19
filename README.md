# EEG-Dash

[![PyPI version](https://img.shields.io/pypi/v/eegdash)](https://pypi.org/project/eegdash/)
[![Docs](https://img.shields.io/badge/docs-stable-brightgreen.svg)](https://sccn.github.io/eegdash)

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/eegdash.svg)](https://pypi.org/project/eegdash/)
[![Downloads](https://pepy.tech/badge/eegdash)](https://pepy.tech/project/eegdash)
[![Coverage](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Feegdash%2FEEGDash%2Fmain%2Fcoverage.json&query=%24.totals.percent_covered_display&suffix=%25&label=coverage)](https://github.com/eegdash/EEGDash/blob/main/coverage.json)

EEG-DaSh is a data-sharing archive for MEEG (EEG, MEG) recordings contributed by collaborating labs. It preserves publicly funded research data and exposes it in a form that machine learning and deep learning workflows can use directly.

## Data source

The archive draws on 25 labs and 27,053 participants, with recordings covering both EEG and MEG. Subjects include healthy controls and clinical groups: ADHD, depression, schizophrenia, dementia, autism, and psychosis. Tasks range across sleep, meditation, and cognitive paradigms. EEG-DaSh also pulls in 330 BIDS-formatted MEEG datasets converted from NEMAR.

## Data format

EEGDash queries return a **PyTorch Dataset**. The format plugs directly into PyTorch's `DataLoader` for batching, shuffling, and parallel loading, which matters when training models on large EEG corpora.

## Data preprocessing

EEGDash datasets are [braindecode](https://braindecode.org/stable/index.html) datasets, which are themselves PyTorch datasets. Any preprocessing that works on a braindecode dataset works on an EEGDash dataset. See the braindecode tutorials for the available options.

## EEG-Dash usage

### Install
Use your preferred Python environment manager with Python > 3.10 to install the package.
* To install the eegdash package, use the following command: `pip install eegdash`
* To verify the installation, start a Python session and type: `from eegdash import EEGDash`

See the tutorials at [eegdash.org](https://eegdash.org/) for end-to-end examples.

## Education (coming soon)

We run workshops and student training events with US and Israeli partners, online and in person. 2025 dates will go out on the EEGLABNEWS mailing list. [Subscribe here](https://sccn.ucsd.edu/mailman/listinfo/eeglabnews).

## About EEG-DaSh

EEG-DaSh is a collaborative initiative between the United States and Israel, supported by the National Science Foundation (NSF). The partnership brings together experts from the Swartz Center for Computational Neuroscience (SCCN) at the University of California San Diego (UCSD) and Ben-Gurion University (BGU) in Israel. 

![Screenshot 2024-10-03 at 09 14 06](https://github.com/user-attachments/assets/327639d3-c3b4-46b1-9335-37803209b0d3)




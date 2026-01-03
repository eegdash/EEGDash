"""Shared search keywords for ingestion fetchers.

This module centralizes keyword lists used across different ingestion sources to
reduce per-file verbosity and make maintenance easier.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Zenodo
# -----------------------------------------------------------------------------

# Target modalities (focused on surface/non-invasive recordings)
ZENODO_TARGET_MODALITIES = ["eeg", "meg", "fnirs"]

# Modalities to exclude from results
ZENODO_EXCLUDED_MODALITIES = [
    "ieeg",
    "ecog",
    "seeg",
    "intracranial",
    "lfp",
    "spike",
    "spiking",
    "neuropixels",
    "mea",
]

# Simple keyword searches - more reliable than complex Elasticsearch syntax
# Each search combines modality + BIDS keywords
ZENODO_SIMPLE_SEARCHES = [
    # EEG variants
    "EEG BIDS",
    "electroencephalography BIDS",
    "electroencephalogram BIDS",
    # MEG variants
    "MEG BIDS",
    "magnetoencephalography BIDS",
    # fNIRS variants
    "fNIRS BIDS",
    "NIRS BIDS",
    "near-infrared spectroscopy BIDS",
    # General neural + BIDS
    "BIDS brain imaging",
    "BIDS neuroimaging dataset",
    "ERP BIDS",
]

# -----------------------------------------------------------------------------
# Figshare
# -----------------------------------------------------------------------------

# Multiple search queries to find more EEG/neural recording datasets
FIGSHARE_MULTI_QUERIES = [
    "EEG BIDS",
    "electroencephalography BIDS",
    "EEG dataset",
    "ERP EEG",
    "MEG BIDS",
    "magnetoencephalography",
    "iEEG BIDS",
    "intracranial EEG",
    "ECoG dataset",
    "EMG dataset",
    "electromyography",
    "fNIRS dataset",
    "brain signals dataset",
    "neural recording dataset",
    "BCI dataset",
    "brain-computer interface",
    "sleep EEG",
    "epilepsy EEG",
]

# -----------------------------------------------------------------------------
# OSF
# -----------------------------------------------------------------------------

# License ID to name mapping (common OSF licenses)
OSF_LICENSE_NAMES = {
    "563c1cf88c5e4a3877f9e96a": "CC-BY-4.0",
    "563c1cf88c5e4a3877f9e965": "CC0-1.0",
    "563c1cf88c5e4a3877f9e968": "MIT",
    "563c1cf88c5e4a3877f9e96c": "GPL-3.0",
    "563c1cf88c5e4a3877f9e96e": "Apache-2.0",
    "563c1cf88c5e4a3877f9e967": "BSD-2-Clause",
    "563c1cf88c5e4a3877f9e969": "BSD-3-Clause",
    "563c1cf88c5e4a3877f9e96b": "CC-BY-NC-4.0",
}

# Categories we're interested in (datasets/data, not posters/presentations)
OSF_DATA_CATEGORIES = {"data", "project", "analysis", "software", "other"}

# Modality tags to search for - optimized for speed (core keywords only)
# Less common variants are covered by title search
OSF_MODALITY_TAGS = {
    "eeg": ["eeg", "electroencephalography", "erp"],  # erp = event-related potential
    "meg": ["meg", "magnetoencephalography"],
    "emg": ["emg", "electromyography"],
    "fnirs": ["fnirs", "fNIRS", "nirs"],
    "lfp": ["lfp", "local field potential"],
    "spike": ["spike", "single unit", "multi-unit"],
    "mea": ["mea", "microelectrode array", "neuropixels"],
    "ieeg": ["ieeg", "intracranial eeg", "seeg", "ecog"],
}

# Title search keywords - for datasets without tags
# These are searched via filter[title][icontains]
OSF_TITLE_SEARCH_KEYWORDS = {
    "eeg": [
        "EEG",
        "electroencephalography",
        "electroencephalogram",
        "event-related potential",
        "ERP CORE",
    ],
    "meg": ["MEG", "magnetoencephalography"],
    "emg": ["EMG", "electromyography"],
    "fnirs": ["fNIRS", "NIRS", "near-infrared spectroscopy"],
    "ieeg": ["iEEG", "intracranial EEG", "ECoG", "sEEG"],
    "bids": ["BIDS"],
}

# -----------------------------------------------------------------------------
# SciDB
# -----------------------------------------------------------------------------

# Modality keywords for searching SciDB - comprehensive keyword coverage
SCIDB_MODALITY_KEYWORDS = {
    "eeg": [
        "eeg",
        "electroencephalography",
        "electroencephalogram",
        "scalp eeg",
        "scalp-eeg",
    ],
    "meg": ["meg", "magnetoencephalography", "magnetoencephalogram"],
    "emg": ["emg", "electromyography", "electromyogram"],
    "fnirs": [
        "fnirs",
        "fNIRS",
        "nirs",
        "near-infrared spectroscopy",
        "near infrared spectroscopy",
        "functional near-infrared",
    ],
    "lfp": [
        "lfp",
        "local field potential",
        "local field potentials",
        "field potential",
        "field potentials",
    ],
    "spike": [
        "single unit",
        "single-unit",
        "multi-unit",
        "multiunit",
        "spike",
        "spike train",
        "neuronal firing",
        "unit activity",
        "single unit activity",
        "multi-unit activity",
    ],
    "mea": [
        "mea",
        "microelectrode array",
        "microelectrode arrays",
        "utah array",
        "neuropixels",
        "depth electrode",
    ],
    "ieeg": [
        "ieeg",
        "intracranial eeg",
        "intracranial electroencephalography",
        "intracranial electroencephalogram",
        "seeg",
        "stereoelectroencephalography",
        "ecog",
        "electrocorticography",
        "corticography",
        "subdural electrode",
        "subdural grid",
        "subdural strip",
    ],
    "bids": ["bids", "brain imaging data structure", "brain imaging data structures"],
}

# -----------------------------------------------------------------------------
# data.ru.nl
# -----------------------------------------------------------------------------

DATARN_MODALITY_SEARCHES = {
    "eeg": "EEG",
    "meg": "MEG",
    "emg": "EMG",
    "fnirs": "fNIRS",
    "lfp": "LFP",
    "ieeg": "iEEG",
}

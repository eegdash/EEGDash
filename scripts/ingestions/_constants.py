"""Constants for BIDS dataset digestion.

This module centralizes constants used across the ingestion scripts,
particularly for modality detection and format-specific file handling.
"""

# Semantic mapping to canonical BIDS modalities
# Maps variant names to their canonical BIDS equivalents
MODALITY_CANONICAL_MAP: dict[str, str] = {
    "nirs": "fnirs",
    "fnirs": "fnirs",
    "spike": "ieeg",
    "lfp": "ieeg",
    "mea": "ieeg",
}

# Supported canonical neurophysiology modalities
# These are the standard BIDS modality names we recognize
NEURO_MODALITIES: tuple[str, ...] = ("eeg", "meg", "ieeg", "emg", "fnirs")

# Modalities we care about detecting (including aliases/variants)
# This includes non-canonical names that should be mapped via MODALITY_CANONICAL_MAP
MODALITY_DETECTION_TARGETS: tuple[str, ...] = (
    "eeg",
    "meg",
    "ieeg",
    "emg",
    "nirs",
    "fnirs",
    "spike",
    "lfp",
    "mea",
)

# CTF MEG uses .ds directories containing multiple files
# We should only match the .ds directory path, not files inside
CTF_INTERNAL_EXTENSIONS: set[str] = {
    ".meg4",
    ".res4",
    ".hc",
    ".infods",
    ".acq",
    ".hist",
    ".newds",
}

# MEF3 (Multiscale Electrophysiology Format) uses .mefd directories
# Internal files should be skipped - only the .mefd directory is the data file
MEF3_INTERNAL_EXTENSIONS: set[str] = {
    ".tmet",  # Time series metadata
    ".tidx",  # Time index
    ".tdat",  # Time series data
}

# MEF3 internal directory patterns (inside .mefd)
MEF3_INTERNAL_DIRS: set[str] = {".timd", ".segd"}

# Dataset IDs explicitly excluded from digestion AND injection. Single
# source of truth shared by 3_digest.py and 5_inject.py — keep both
# stages in sync. Add new IDs here, not in either stage.
EXCLUDED_DATASETS: set[str] = {
    "test",
    "ds003380",
    # OpenNeuro IDs that now redirect to other datasets on openneuro.org.
    # ds004929 and ds005930 are fNIRS-only (no EEG); ds005407 redirects.
    "ds004929",
    "ds005407",
    "ds005930",
}

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

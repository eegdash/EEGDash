"""Plot generation utilities for EEGDash documentation."""

from .bubble import generate_dataset_bubble  # noqa: F401
from .colours import (  # noqa: F401
    CANONICAL_MAP,
    COLUMN_COLOR_MAPS,
    MODALITY_COLOR_MAP,
    PATHOLOGY_COLOR_MAP,
    TYPE_COLOR_MAP,
    hex_to_rgba,
)
from .ridgeline import generate_modality_ridgeline  # noqa: F401

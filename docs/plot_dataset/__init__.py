"""Plot generation utilities for EEGDash documentation."""

from .bubble import generate_dataset_bubble  # noqa: F401
from .clinical_breakdown import generate_clinical_stacked_bar  # noqa: F401
from .colours import (  # noqa: F401
    CANONICAL_MAP,
    COLUMN_COLOR_MAPS,
    MODALITY_COLOR_MAP,
    PATHOLOGY_COLOR_MAP,
    TYPE_COLOR_MAP,
    hex_to_rgba,
)
from .growth import generate_dataset_growth  # noqa: F401
from .plot_sankey import generate_dataset_sankey  # noqa: F401
from .ridgeline import generate_modality_ridgeline  # noqa: F401
from .treemap import generate_dataset_treemap  # noqa: F401

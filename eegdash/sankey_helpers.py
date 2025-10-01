"""Helpers for Sankey diagram generation."""

# Color mappings consistent with prepare_summary_tables.py and custom.css
PATHOLOGY_COLOR_MAP = {
    "Healthy": "#16a34a",
    "Clinical": "#ef4444",  # Use a red consistent with the table styles
    "Unknown": "#94a3b8",
}

MODALITY_COLOR_MAP = {
    "Visual": "#2563eb",
    "Auditory": "#0ea5e9",
    "Tactile": "#10b981",
    "Somatosensory": "#10b981",
    "Multisensory": "#ec4899",
    "Motor": "#f59e0b",
    "Resting State": "#6366f1",
    "Rest": "#6366f1",
    "Sleep": "#7c3aed",
    "Other": "#14b8a6",
    "Unknown": "#94a3b8",
}

TYPE_COLOR_MAP = {
    "Perception": "#3b82f6",
    "Decision-making": "#eab308",
    "Rest": "#16a34a",
    "Resting-state": "#16a34a",
    "Sleep": "#8b5cf6",
    "Cognitive": "#6366f1",
    "Clinical": "#f87171",  # Lighter red to match table
    "Memory": "#c4b5fd",  # Lighter purple to match table
    "Attention": "#c4b5fd",  # Lighter purple to match table
    "Intervention": "#c4b5fd",  # Lighter purple to match table
    "Learning": "#c4b5fd",  # Lighter purple to match table
    "Other": "#c4b5fd",  # Lighter purple to match table
    "Unknown": "#94a3b8",
}

# Canonical mappings to normalize values
CANONICAL_MAP = {
    "Type Subject": {
        "healthy controls": "Healthy",
        "healthy": "Healthy",
        "control": "Healthy",
        "clinical": "Clinical",
        "patient": "Clinical",
    },
    "modality of exp": {
        "visual": "Visual",
        "auditory": "Auditory",
        "tactile": "Tactile",
        "somatosensory": "Tactile",
        "multisensory": "Multisensory",
        "motor": "Motor",
        "rest": "Resting State",
        "resting state": "Resting State",
        "resting-state": "Resting State",
        "sleep": "Sleep",
        "other": "Other",
    },
    "type of exp": {
        "perception": "Perception",
        "decision making": "Decision-making",
        "decision-making": "Decision-making",
        "rest": "Rest",
        "resting state": "Resting-state",
        "resting-state": "Resting-state",
        "sleep": "Sleep",
        "cognitive": "Cognitive",
        "clinical": "Clinical",
    },
}

# Map column names to their color maps
COLUMN_COLOR_MAPS = {
    "Type Subject": PATHOLOGY_COLOR_MAP,
    "modality of exp": MODALITY_COLOR_MAP,
    "type of exp": TYPE_COLOR_MAP,
}


def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert hex color to rgba with given alpha."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError("Invalid hex color format")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

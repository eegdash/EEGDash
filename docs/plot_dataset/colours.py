"""Helpers for Sankey diagram generation."""

# Color mappings consistent with prepare_summary_tables.py and custom.css
PATHOLOGY_COLOR_MAP = {
    "Healthy": "#22c55e",  # green
    "Clinical": "#f87171",  # Lighter red to match table
    "Unknown": "#94a3b8",  # grey
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
    "EEG": "#3b82f6",  # blue-500
    "iEEG": "#1d4ed8",  # blue-700
    "MEG": "#a855f7",  # purple-500
    "fNIRS": "#ef4444",  # red-500
    "EMG": "#f97316",  # orange-500
    "fMRI": "#06b6d4",  # cyan-500
    "MRI": "#0891b2",  # cyan-600
    "ECG": "#be123c",  # rose-700
    "Behavior": "#84cc16",  # lime-500
}

MODALITY_EMOJI = {
    "Visual": "👁️",
    "Auditory": "👂",
    "Sleep": "🌙",
    "Multisensory": "🧩",
    "Tactile": "✋",
    "Motor": "🏃",
    "Resting State": "🧘",
    "Rest": "🧘",
    "Other": "🧭",
    "Unknown": "❔",
    "EEG": "🧠",
    "iEEG": "⚡",
    "MEG": "🧲",
    "fNIRS": "💡",
    "EMG": "💪",
    "fMRI": "🧠",
    "MRI": "📷",
    "ECG": "❤️",
    "Behavior": "📝",
}

PATHOLOGY_PASTEL_OVERRIDES = {
    # Healthy / Control
    "Healthy": "#86efac",  # green-300
    "healthy": "#86efac",
    # Unknown / Unspecified
    "Unknown": "#cbd5e1",  # slate-300
    "unknown": "#cbd5e1",
    "Unspecified Clinical": "#fda4af",  # rose-300 - more visible clinical indicator
    "unspecified clinical": "#fda4af",
    # Neurological conditions - warm colors
    "Epilepsy": "#fdba74",  # orange-300
    "epilepsy": "#fdba74",
    "Parkinson's": "#f9a8d4",  # pink-300
    "parkinson's": "#f9a8d4",
    "parkinson": "#f9a8d4",
    "Alzheimer": "#c4b5fd",  # violet-300
    "alzheimer": "#c4b5fd",
    "Dementia": "#ddd6fe",  # violet-200
    "dementia": "#ddd6fe",
    "TBI": "#fca5a5",  # red-300
    "tbi": "#fca5a5",
    # Psychiatric conditions - cool colors
    "Schizophrenia": "#7dd3fc",  # sky-300
    "schizophrenia": "#7dd3fc",
    "Psychosis": "#67e8f9",  # cyan-300
    "psychosis": "#67e8f9",
    "Depression": "#a5b4fc",  # indigo-300
    "depression": "#a5b4fc",
    "ADHD": "#fcd34d",  # amber-300
    "adhd": "#fcd34d",
    # Other clinical
    "Surgery": "#fed7aa",  # orange-200
    "surgery": "#fed7aa",
    "Clinical": "#fecaca",  # red-200
    "clinical": "#fecaca",
    "Other": "#e5e7eb",  # gray-200
    "other": "#e5e7eb",
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
        "resting_state": "Resting State",
        "sleep": "Sleep",
        "other": "Other",
        "eeg": "EEG",
        "ieeg": "iEEG",
        "meg": "MEG",
        "fnirs": "fNIRS",
        "emg": "EMG",
        "fmri": "fMRI",
        "mri": "MRI",
        "ecg": "ECG",
        "behavior": "Behavior",
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
        "other": "Other",
    },
}

# Map column names to their color maps
# Use PATHOLOGY_PASTEL_OVERRIDES for Type Subject to have consistent colors
# for specific conditions (epilepsy, schizophrenia, etc.) across all plots
COLUMN_COLOR_MAPS = {
    "Type Subject": PATHOLOGY_PASTEL_OVERRIDES,
    "modality of exp": MODALITY_COLOR_MAP,
    "type of exp": TYPE_COLOR_MAP,
}


def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert hex color to rgba with given alpha."""
    if not isinstance(hex_color, str) or not hex_color.startswith("#"):
        # This is not a valid hex color, return a default color
        return "rgba(148, 163, 184, 0.2)"  # Default grey
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "rgba(148, 163, 184, 0.2)"  # Default grey for invalid length
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError:
        return "rgba(148, 163, 184, 0.2)"  # Default grey for conversion error
    return f"rgba({r}, {g}, {b}, {alpha})"

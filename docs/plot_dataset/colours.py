"""Unified color palette for EEGDash visualizations and CSS.

This module provides a single source of truth for colors used across:
- Plotly charts (bubble, sankey, ridgeline, growth, treemap, clinical)
- HTML table tags in custom.css
- Tag palette styling via tag-palette.js

Color Philosophy:
- Recording modalities (EEG, MEG, etc.): Distinct, saturated colors
- Experimental modalities (Visual, Auditory, etc.): Medium saturation
- Pathology: Green for healthy, red tones for clinical, grey for unknown
- Type of experiment: Blue/purple tones for cognitive, warm for clinical
"""


def _create_color_map_with_aliases(base_map: dict) -> dict:
    """Create color map with automatic case aliases (lower, upper, title).

    This avoids manual duplication like:
        "Epilepsy": "#fdba74",
        "epilepsy": "#fdba74",

    Instead, define the canonical form once and aliases are auto-generated.
    """
    result = {}
    for key, color in base_map.items():
        result[key] = color  # Original
        result[key.lower()] = color  # lowercase
        result[key.upper()] = color  # UPPERCASE
        title = key.title()
        if title != key and title != key.lower() and title != key.upper():
            result[title] = color  # Title Case
    return result


# =============================================================================
# RECORDING MODALITY COLORS (EEG, MEG, iEEG, fNIRS, EMG, etc.)
# =============================================================================
# These are the primary data types - distinct, easily distinguishable colors
RECORDING_MODALITY_COLORS = {
    "EEG": "#3b82f6",  # blue-500 - primary EEG color
    "eeg": "#3b82f6",
    "iEEG": "#06b6d4",  # cyan-500 - clearly distinct from MEG
    "ieeg": "#06b6d4",
    "MEG": "#a855f7",  # purple-500 - magnetic, purple theme
    "meg": "#a855f7",
    "fNIRS": "#f97316",  # orange-500 - optical/light, warm color
    "fnirs": "#f97316",
    "EMG": "#10b981",  # emerald-500 - muscle, green
    "emg": "#10b981",
    "fMRI": "#06b6d4",  # cyan-500
    "fmri": "#06b6d4",
    "MRI": "#0891b2",  # cyan-600
    "mri": "#0891b2",
    "ECG": "#ef4444",  # red-500 - heart
    "ecg": "#ef4444",
    "Behavior": "#84cc16",  # lime-500
    "behavior": "#84cc16",
}

# =============================================================================
# EXPERIMENTAL MODALITY COLORS (Visual, Auditory, Motor, etc.)
# =============================================================================
EXPERIMENTAL_MODALITY_COLORS = {
    "Visual": "#2563eb",  # blue-600
    "visual": "#2563eb",
    "Auditory": "#0ea5e9",  # sky-500
    "auditory": "#0ea5e9",
    "Tactile": "#14b8a6",  # teal-500
    "tactile": "#14b8a6",
    "Somatosensory": "#14b8a6",  # same as tactile
    "somatosensory": "#14b8a6",
    "Multisensory": "#ec4899",  # pink-500
    "multisensory": "#ec4899",
    "Motor": "#f59e0b",  # amber-500
    "motor": "#f59e0b",
    "Resting State": "#6366f1",  # indigo-500
    "resting state": "#6366f1",
    "resting-state": "#6366f1",
    "Rest": "#6366f1",
    "rest": "#6366f1",
    "Sleep": "#7c3aed",  # violet-600
    "sleep": "#7c3aed",
    "Other": "#64748b",  # slate-500
    "other": "#64748b",
    "Unknown": "#94a3b8",  # slate-400
    "unknown": "#94a3b8",
}

# Combined modality map for backward compatibility
MODALITY_COLOR_MAP = {**RECORDING_MODALITY_COLORS, **EXPERIMENTAL_MODALITY_COLORS}

# =============================================================================
# PATHOLOGY COLORS
# =============================================================================
# Simple pathology map for basic healthy/clinical/unknown
PATHOLOGY_COLOR_MAP = {
    "Healthy": "#22c55e",  # green-500
    "healthy": "#22c55e",
    "Clinical": "#ef4444",  # red-500
    "clinical": "#ef4444",
    "Unknown": "#94a3b8",  # slate-400
    "unknown": "#94a3b8",
}

# Detailed pathology colors (pastel) for specific conditions
_BASE_PATHOLOGY_COLORS = {
    # Healthy / Control - green tones
    "Healthy": "#86efac",  # green-300
    # Unknown / Unspecified - grey tones
    "Unknown": "#cbd5e1",  # slate-300
    "Unspecified Clinical": "#fda4af",  # rose-300
    # Neurological conditions - warm colors (orange/red family)
    "Epilepsy": "#fdba74",  # orange-300
    "Parkinson's": "#f9a8d4",  # pink-300
    "Alzheimer": "#c4b5fd",  # violet-300
    "Dementia": "#ddd6fe",  # violet-200
    "TBI": "#fca5a5",  # red-300
    # Psychiatric conditions - cool colors (blue/cyan family)
    "Schizophrenia": "#7dd3fc",  # sky-300
    "Psychosis": "#67e8f9",  # cyan-300
    "Depression": "#a5b4fc",  # indigo-300
    "ADHD": "#fcd34d",  # amber-300
    # Developmental
    "Development": "#c4b5fd",  # violet-300
    "Neurodevelopmental": "#c4b5fd",  # violet-300
    # Other clinical
    "Surgery": "#fed7aa",  # orange-200
    "Clinical": "#fecaca",  # red-200
    "Other": "#e5e7eb",  # gray-200
}

# Create the full map with case aliases
PATHOLOGY_PASTEL_OVERRIDES = _create_color_map_with_aliases(_BASE_PATHOLOGY_COLORS)
PATHOLOGY_PASTEL_OVERRIDES["parkinson"] = PATHOLOGY_PASTEL_OVERRIDES["Parkinson's"]

# =============================================================================
# EXPERIMENT TYPE COLORS
# =============================================================================
TYPE_COLOR_MAP = {
    "Perception": "#3b82f6",  # blue-500
    "perception": "#3b82f6",
    "Decision-making": "#eab308",  # yellow-500
    "decision-making": "#eab308",
    "Rest": "#16a34a",  # green-600
    "rest": "#16a34a",
    "Resting-state": "#16a34a",
    "resting-state": "#16a34a",
    "Sleep": "#8b5cf6",  # violet-500
    "sleep": "#8b5cf6",
    "Cognitive": "#6366f1",  # indigo-500
    "cognitive": "#6366f1",
    "Clinical": "#ef4444",  # red-500 - matches pathology clinical
    "clinical": "#ef4444",
    "Clinical/Intervention": "#ef4444",
    "Memory": "#a78bfa",  # violet-400
    "memory": "#a78bfa",
    "Attention": "#818cf8",  # indigo-400
    "attention": "#818cf8",
    "Intervention": "#f472b6",  # pink-400
    "intervention": "#f472b6",
    "Learning": "#c084fc",  # purple-400
    "learning": "#c084fc",
    "Motor": "#f59e0b",  # amber-500 - matches modality motor
    "motor": "#f59e0b",
    "Other": "#94a3b8",  # slate-400
    "other": "#94a3b8",
    "Unknown": "#94a3b8",
    "unknown": "#94a3b8",
}

# =============================================================================
# EMOJI MAPPINGS
# =============================================================================
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

# =============================================================================
# CANONICAL MAPPINGS (for normalizing input values)
# =============================================================================
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

# =============================================================================
# COLUMN COLOR MAPS (for sankey and other multi-column visualizations)
# =============================================================================
COLUMN_COLOR_MAPS = {
    "Type Subject": PATHOLOGY_PASTEL_OVERRIDES,
    "modality of exp": MODALITY_COLOR_MAP,
    "type of exp": TYPE_COLOR_MAP,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert hex color to rgba with given alpha."""
    if not isinstance(hex_color, str) or not hex_color.startswith("#"):
        return "rgba(148, 163, 184, 0.2)"  # Default grey
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "rgba(148, 163, 184, 0.2)"
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError:
        return "rgba(148, 163, 184, 0.2)"
    return f"rgba({r}, {g}, {b}, {alpha})"


def hex_to_light_gradient(hex_color: str) -> tuple[str, str, str]:
    """Convert hex to a light gradient suitable for tag backgrounds.

    Returns (gradient_start, gradient_end, border_color).
    """
    if not isinstance(hex_color, str) or not hex_color.startswith("#"):
        return ("#f1f5f9", "#e2e8f0", "#cbd5e1")  # Default slate

    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return ("#f1f5f9", "#e2e8f0", "#cbd5e1")

    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError:
        return ("#f1f5f9", "#e2e8f0", "#cbd5e1")

    # Create lighter versions for gradient
    # lighten factor 0.85, medium factor 0.7
    r1 = int(r + (255 - r) * 0.85)
    g1 = int(g + (255 - g) * 0.85)
    b1 = int(b + (255 - b) * 0.85)

    r2 = int(r + (255 - r) * 0.7)
    g2 = int(g + (255 - g) * 0.7)
    b2 = int(b + (255 - b) * 0.7)

    gradient_start = f"#{r1:02x}{g1:02x}{b1:02x}"
    gradient_end = f"#{r2:02x}{g2:02x}{b2:02x}"
    border_color = f"#{hex_color}"  # Re-add # prefix

    return (gradient_start, gradient_end, border_color)


def generate_css_tag_styles() -> str:
    """Generate CSS styles for tags based on the color definitions.

    This ensures CSS tags match plot colors exactly.
    """
    css_lines = [
        "/* ==========================================================================",
        "   AUTO-GENERATED TAG COLORS - matches colours.py",
        "   Do not edit manually - regenerate with: python -c 'from plot_dataset.colours import generate_css_tag_styles; print(generate_css_tag_styles())'",
        "   ========================================================================== */",
        "",
    ]

    # Recording modality tags
    css_lines.append("/* Recording Modality Tags */")
    for name, color in RECORDING_MODALITY_COLORS.items():
        if name == name.lower():
            continue  # Skip lowercase aliases
        slug = name.lower()
        start, end, border = hex_to_light_gradient(color)
        css_lines.append(f'.tag[data-tag-value="{slug}"] {{')
        css_lines.append(f"  background: linear-gradient(135deg, {start}, {end});")
        css_lines.append(f"  border: 1px solid {border} !important;")
        # Calculate text color (darker version of the color)
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        text_color = f"#{max(0, r - 80):02x}{max(0, g - 80):02x}{max(0, b - 80):02x}"
        css_lines.append(f"  color: {text_color};")
        css_lines.append("}")
        css_lines.append("")

    # Pathology tags
    css_lines.append("/* Pathology Tags */")
    for name, color in [
        ("healthy", PATHOLOGY_COLOR_MAP["Healthy"]),
        ("clinical", PATHOLOGY_COLOR_MAP["Clinical"]),
    ]:
        start, end, border = hex_to_light_gradient(color)
        css_lines.append(f".tag-pathology-{name} {{")
        css_lines.append(f"  background: linear-gradient(135deg, {start}, {end});")
        css_lines.append(f"  border: 1px solid {border} !important;")
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        text_color = f"#{max(0, r - 80):02x}{max(0, g - 80):02x}{max(0, b - 80):02x}"
        css_lines.append(f"  color: {text_color};")
        css_lines.append("}")
        css_lines.append("")

    # Experimental modality tags
    css_lines.append("/* Experimental Modality Tags */")
    for name in [
        "Visual",
        "Auditory",
        "Tactile",
        "Multisensory",
        "Motor",
        "Resting State",
        "Sleep",
    ]:
        color = EXPERIMENTAL_MODALITY_COLORS.get(name, "#94a3b8")
        slug = name.lower().replace(" ", "-")
        start, end, border = hex_to_light_gradient(color)
        css_lines.append(f".tag-modality-{slug} {{")
        css_lines.append(f"  background: linear-gradient(135deg, {start}, {end});")
        css_lines.append(f"  border: 1px solid {border} !important;")
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        text_color = f"#{max(0, r - 80):02x}{max(0, g - 80):02x}{max(0, b - 80):02x}"
        css_lines.append(f"  color: {text_color};")
        css_lines.append("}")
        css_lines.append("")

    # Type tags
    css_lines.append("/* Experiment Type Tags */")
    for name in [
        "Perception",
        "Decision-making",
        "Rest",
        "Sleep",
        "Cognitive",
        "Clinical",
    ]:
        color = TYPE_COLOR_MAP.get(name, "#94a3b8")
        slug = name.lower()
        start, end, border = hex_to_light_gradient(color)
        css_lines.append(f".tag-type-{slug} {{")
        css_lines.append(f"  background: linear-gradient(135deg, {start}, {end});")
        css_lines.append(f"  border: 1px solid {border} !important;")
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        text_color = f"#{max(0, r - 80):02x}{max(0, g - 80):02x}{max(0, b - 80):02x}"
        css_lines.append(f"  color: {text_color};")
        css_lines.append("}")
        css_lines.append("")

    return "\n".join(css_lines)

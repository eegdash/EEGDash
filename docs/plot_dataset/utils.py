from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:  # Allow import both as package and script
    from .colours import CANONICAL_MAP, MODALITY_COLOR_MAP
except ImportError:  # pragma: no cover - fallback for direct script execution
    from colours import CANONICAL_MAP, MODALITY_COLOR_MAP  # type: ignore

__all__ = [
    "build_and_export_html",
    "detect_modality_column",
    "get_dataset_url",
    "human_readable_size",
    "normalize_modality_string",
    "primary_modality",
    "primary_recording_modality",
    "read_dataset_csv",
    "safe_int",
]

_SEPARATORS = ("/", "|", ";")


def primary_modality(value: Any) -> str:
    """Return the canonical modality label for a record."""
    if value is None:
        return "Unknown"
    if isinstance(value, float) and pd.isna(value):
        return "Unknown"

    text = str(value).strip()
    if not text:
        return "Unknown"

    # normalise separators, keep order of appearance
    for sep in _SEPARATORS:
        text = text.replace(sep, ",")
    tokens = [tok.strip() for tok in text.split(",") if tok.strip()]
    if not tokens:
        return "Unknown"

    first = tokens[0]
    canonical_map = CANONICAL_MAP.get("modality of exp", {})
    lowered = first.lower()
    canonical = canonical_map.get(lowered)
    if canonical:
        return canonical

    if first in MODALITY_COLOR_MAP:
        return first

    title_variant = first.title()
    if title_variant in MODALITY_COLOR_MAP:
        return title_variant

    return "Other"


# Canonical recording modality order and mapping
RECORDING_MODALITY_MAP = {
    "eeg": "EEG",
    "ieeg": "iEEG",
    "meg": "MEG",
    "fnirs": "fNIRS",
    "nirs": "fNIRS",  # Also accept 'nirs' without 'f' prefix
    "emg": "EMG",
    "ecg": "ECG",
    "fmri": "fMRI",
    "mri": "MRI",
}


def primary_recording_modality(value: Any) -> str:
    """Return the canonical recording modality label (EEG, MEG, iEEG, etc.)."""
    if value is None:
        return "Unknown"
    if isinstance(value, float) and pd.isna(value):
        return "Unknown"

    text = str(value).strip().lower()
    if not text:
        return "Unknown"

    # Handle multi-modality entries (e.g., "eeg, meg") - take the first
    for sep in _SEPARATORS:
        text = text.replace(sep, ",")
    tokens = [tok.strip() for tok in text.split(",") if tok.strip()]
    if not tokens:
        return "Unknown"

    first = tokens[0].lower()

    # Map to canonical form
    canonical = RECORDING_MODALITY_MAP.get(first)
    if canonical:
        return canonical

    # Try direct match in color map
    if first.upper() in MODALITY_COLOR_MAP:
        return first.upper()

    return "Other"


def safe_int(value: Any, default: int | None = None) -> int | None:
    """Convert *value* to ``int`` when possible; otherwise return *default*."""
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return int(round(float(value)))
    except Exception:
        return default


def human_readable_size(num_bytes: int | float | None) -> str:
    """Format bytes using the closest unit among MB, GB, TB (fallback to KB/B)."""
    if num_bytes is None:
        return "0 B"

    try:
        size = float(num_bytes)
    except Exception:
        return "0 B"

    units = [
        (1024**4, "TB"),
        (1024**3, "GB"),
        (1024**2, "MB"),
        (1024**1, "KB"),
        (1, "B"),
    ]

    for factor, unit in units:
        if size >= factor:
            value = size / factor
            if unit in {"B", "KB"}:
                return f"{int(round(value))} {unit}"
            return f"{value:.2f} {unit}"
    return "0 B"


def get_dataset_url(name: str) -> str:
    """Generate dataset URL for plots (relative to dataset summary page)."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = str(name).strip()
    if not text:
        return ""
    return f"api/dataset/eegdash.dataset.{text.upper()}.html"


def ensure_directory(path: str | Path) -> Path:
    """Create *path* directory if required and return ``Path`` instance."""
    dest = Path(path)
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def build_and_export_html(
    fig,
    out_path: str | Path,
    div_id: str,
    height: int = 550,
    extra_style: str = "",
    pre_html: str = "",
    extra_html: str = "",
    config: dict | None = None,
    include_default_style: bool = True,
    html_content: str | None = None,
) -> Path:
    """Build styled HTML from a Plotly figure and write it to *out_path*.

    This consolidates the common HTML wrapping and export logic used
    across all chart generation files.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure or None
        The Plotly figure to export. Can be None if *html_content* is provided.
    out_path : str | Path
        Destination file path for the exported HTML.
    div_id : str
        The HTML ``id`` attribute for the plot container div.
    height : int, optional
        Desired chart height in pixels (default 550).
    extra_style : str, optional
        Additional CSS rules to include inside the ``<style>`` block.
    pre_html : str, optional
        HTML content to insert before the plot container (e.g., loading divs).
    extra_html : str, optional
        Additional HTML/JS to append after the plot container.
    config : dict, optional
        Plotly config options. Defaults to responsive mode with no logo.
    include_default_style : bool, optional
        Whether to include the default CSS styling block. Set to False for
        charts with completely custom HTML structure (default True).
    html_content : str, optional
        Pre-generated HTML content for the plot. If provided, *fig* is ignored
        and this content is used directly (useful for custom rendering).

    Returns
    -------
    Path
        The path to the written HTML file.

    """
    if html_content is None:
        if config is None:
            config = {"responsive": True, "displaylogo": False}
        html_content = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config=config,
            div_id=div_id,
        )

    if include_default_style:
        styled_html = f"""
<style>
#{div_id} {{
    width: 100% !important;
    height: {height}px !important;
    min-height: {height}px;
    margin: 0 auto;
}}
#{div_id} .plotly-graph-div {{
    width: 100% !important;
    height: 100% !important;
}}
{extra_style}
</style>
{pre_html}{html_content}
{extra_html}
"""
    else:
        # Custom HTML structure without default styling
        styled_html = f"{extra_style}{pre_html}{html_content}{extra_html}"

    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(styled_html, encoding="utf-8")
    return dest


def normalize_modality_string(val: Any) -> str:
    """Normalize modality string to standard format.

    Handles various input formats and maps them to canonical modality names
    like EEG, iEEG, MEG, fNIRS, EMG, fMRI, MRI, ECG, Behavior.

    Parameters
    ----------
    val : Any
        The raw modality value (typically a string from a DataFrame column).

    Returns
    -------
    str
        The normalized modality string, or "Unknown" if not recognized.

    """
    if not isinstance(val, str) or pd.isna(val):
        return "Unknown"

    lowered = val.lower().strip()
    if lowered in ("nan", "none", ""):
        return "Unknown"

    # Priority checks - order matters (e.g., ieeg before eeg)
    if "ieeg" in lowered or "intracranial" in lowered:
        return "iEEG"
    if "meg" in lowered:
        return "MEG"
    if "fnirs" in lowered:
        return "fNIRS"
    if "emg" in lowered:
        return "EMG"
    if "fmri" in lowered or "functional magnetic resonance" in lowered:
        return "fMRI"
    if "mri" in lowered:
        return "MRI"
    if "eeg" in lowered:
        return "EEG"
    if "ecg" in lowered:
        return "ECG"
    if "behavior" in lowered:
        return "Behavior"

    # Fallback: clean up the string (remove list brackets) and title-case
    cleaned = (
        val.replace("['", "").replace("']", "").replace('["', "").replace('"]', "")
    )
    return cleaned.title() if cleaned else "Unknown"


# Default candidates for modality column detection
_DEFAULT_MODALITY_CANDIDATES = (
    "recording_modality",
    "record_modality",
    "experimental_modality",
    "modality of exp",
    "modality",
    "record modality",
)


def detect_modality_column(
    df: pd.DataFrame,
    candidates: tuple[str, ...] | list[str] | None = None,
) -> str | None:
    """Detect the modality column from a DataFrame.

    Searches through a list of candidate column names and returns
    the first one found in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to search for a modality column.
    candidates : tuple[str, ...] | list[str] | None, optional
        Ordered list of candidate column names to check. If None, uses
        the default candidates: recording_modality, record_modality,
        experimental_modality, modality of exp, modality, record modality.

    Returns
    -------
    str | None
        The name of the first matching column, or None if no match is found.

    """
    if candidates is None:
        candidates = _DEFAULT_MODALITY_CANDIDATES

    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def read_dataset_csv(path: str | Path) -> pd.DataFrame:
    """Read a dataset summary CSV file into a DataFrame.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.

    """
    return pd.read_csv(path, index_col=False, header=0, skipinitialspace=True)

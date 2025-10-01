from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:  # Allow import both as package and script
    from .colours import CANONICAL_MAP, MODALITY_COLOR_MAP
except ImportError:  # pragma: no cover - fallback for direct script execution
    from colours import CANONICAL_MAP, MODALITY_COLOR_MAP  # type: ignore

__all__ = [
    "get_dataset_url",
    "human_readable_size",
    "primary_modality",
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
    return f"../../api/dataset/eegdash.dataset.{text.upper()}.html"


def ensure_directory(path: str | Path) -> Path:
    """Create *path* directory if required and return ``Path`` instance."""
    dest = Path(path)
    dest.mkdir(parents=True, exist_ok=True)
    return dest
